"""
Phase 5 — Edit Executor
========================
Executes targeted edits based on classified intent.

audio      → re-synthesise TTS with new voice parameters
video_frame→ apply OpenCV filters (darken/brighten/sepia/grayscale/etc.)
video      → recompose with updated parameters (speed, subtitle toggle)
script     → re-invoke Phase 1 and cascade through phases 2+3
"""

import json
import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.registry import mcp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(path: str = "outputs/scene_manifest.json") -> Dict:
    with open(path) as f:
        return json.load(f)


def _load_char_db(path: str = "outputs/character_db.json") -> Dict:
    with open(path) as f:
        return json.load(f)


def _scene_ids_from_scope(scope: str, manifest: Dict) -> List[int]:
    """Parse scope → list of scene IDs to act on."""
    if scope.startswith("scene:"):
        try:
            return [int(scope.split(":")[1])]
        except ValueError:
            pass
    return [s["scene_id"] for s in manifest.get("scenes", [])]


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO executor
# ─────────────────────────────────────────────────────────────────────────────

class AudioEditor:
    """Handles audio-targeted edits: voice tone, speed, BGM add/remove."""

    TONE_VOICE_MAP = {
        "whispered": {"voice": "en-US-AriaNeural",  "rate": "-20%", "pitch": "-5Hz"},
        "whisper":   {"voice": "en-US-AriaNeural",  "rate": "-20%", "pitch": "-5Hz"},
        "angry":     {"voice": "en-US-GuyNeural",   "rate": "+15%", "pitch": "+10Hz"},
        "dramatic":  {"voice": "en-US-EricNeural",  "rate": "-10%", "pitch": "-2Hz"},
        "calm":      {"voice": "en-US-AriaNeural",  "rate": "-5%",  "pitch": "0Hz"},
        "excited":   {"voice": "en-US-JennyNeural", "rate": "+20%", "pitch": "+5Hz"},
        "sad":       {"voice": "en-US-AriaNeural",  "rate": "-15%", "pitch": "-8Hz"},
        "nervous":   {"voice": "en-US-GuyNeural",   "rate": "+10%", "pitch": "+3Hz"},
        "serious":   {"voice": "en-GB-RyanNeural",  "rate": "-5%",  "pitch": "-3Hz"},
        "cheerful":  {"voice": "en-US-JennyNeural", "rate": "+10%", "pitch": "+5Hz"},
    }

    def apply(self, intent: Dict) -> Dict:
        scope      = intent.get("scope", "all")
        params     = intent.get("parameters", {})
        action     = intent.get("intent", "")
        manifest   = _load_manifest()
        char_db    = _load_char_db()
        scene_ids  = _scene_ids_from_scope(scope, manifest)

        results = []

        if action == "change_voice_tone":
            results = self._change_voice_tone(
                scene_ids, scope, params, manifest, char_db
            )
        elif action == "change_voice_speed":
            results = self._change_voice_speed(scene_ids, params, manifest)
        elif action == "add_background_music":
            results = self._add_bgm(scene_ids, params)
        elif action == "remove_background_music":
            results = self._remove_bgm(scene_ids)

        return {"target": "audio", "action": action, "affected_scenes": results}

    def _change_voice_tone(self, scene_ids, scope, params, manifest, char_db):
        tone        = params.get("tone", "calm")
        voice_cfg   = self.TONE_VOICE_MAP.get(tone, {"voice": "en-US-AriaNeural",
                                                       "rate": "0%", "pitch": "0Hz"})
        target_char = None
        if scope.startswith("character:"):
            target_char = scope.split(":", 1)[1].upper()

        results = []
        for sid in scene_ids:
            scene = next((s for s in manifest["scenes"] if s["scene_id"] == sid), None)
            if not scene:
                continue
            for i, dlg in enumerate(scene.get("dialogue", [])):
                speaker = dlg["speaker"].upper()
                if target_char and speaker != target_char:
                    continue
                # Re-synthesise with new voice
                out_path = f"audio_tracks/scene_{sid}_dlg_{i:02d}_{speaker.lower().replace(' ','_')}_edited.mp3"
                result = mcp.invoke("voice_cloning_synthesizer", {
                    "text":        dlg["line"],
                    "voice":       voice_cfg["voice"],
                    "output_path": out_path,
                })
                results.append({
                    "scene_id": sid,
                    "speaker":  speaker,
                    "tone":     tone,
                    "file":     out_path,
                    "status":   result["status"],
                })
            print(f"    [AudioEditor] Scene {sid}: voice tone → {tone}")
        return results

    def _change_voice_speed(self, scene_ids, params, manifest):
        factor  = float(params.get("factor", 1.5))
        results = []
        for sid in scene_ids:
            audio_path = f"audio_tracks/scene_{sid}_merged.mp3"
            out_path   = f"audio_tracks/scene_{sid}_speed_{factor}x.mp3"
            if Path(audio_path).exists():
                self._stretch_audio(audio_path, out_path, factor)
                results.append({"scene_id": sid, "factor": factor, "file": out_path})
                print(f"    [AudioEditor] Scene {sid}: speed → {factor}x")
        return results

    def _stretch_audio(self, inp: str, out: str, factor: float):
        """Simple audio speed change via sample-rate manipulation."""
        try:
            with wave.open(inp.replace(".mp3", ".wav")
                           if not inp.endswith(".wav") else inp, "rb") as wf:
                params  = wf.getparams()
                frames  = wf.readframes(wf.getnframes())
        except Exception:
            shutil.copy(inp, out)
            return
        data = np.frombuffer(frames, dtype=np.int16)
        new_len = int(len(data) / factor)
        indices = (np.arange(new_len) * factor).astype(int)
        indices = np.clip(indices, 0, len(data) - 1)
        new_data = data[indices]
        with wave.open(out.replace(".mp3", ".wav"), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(new_data.tobytes())
        shutil.copy(out.replace(".mp3", ".wav"), out)

    def _add_bgm(self, scene_ids, params):
        mood = params.get("mood", "mysterious")
        from phase3.agents import BGMSelectorAgent
        clips  = [{"scene_id": sid, "mood": mood, "duration_s": 5.0,
                   "video_path":"","audio_path":"","location":"","dialogue":[],"action":""}
                  for sid in scene_ids]
        paths  = BGMSelectorAgent().run(clips)
        return [{"scene_id": sid, "mood": mood, "file": p}
                for sid, p in zip(scene_ids, paths)]

    def _remove_bgm(self, scene_ids):
        removed = []
        for sid in scene_ids:
            p = Path(f"outputs/bgm/bgm_scene_{sid}.wav")
            if p.exists():
                p.rename(str(p) + ".bak")
                removed.append({"scene_id": sid, "status": "removed"})
        return removed


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO_FRAME executor (OpenCV filters)
# ─────────────────────────────────────────────────────────────────────────────

class VideoFrameEditor:
    """
    Applies OpenCV-based filters to scene videos frame-by-frame.
    Filters: darken, brighten, sepia, grayscale, warm, cool, blur, sharpen, vintage
    """

    FILTER_REGISTRY = {
        "darken":          "_filter_darken",
        "brighter":        "_filter_brighten",
        "make_darker":     "_filter_darken",
        "make_brighter":   "_filter_brighten",
        "sepia":           "_filter_sepia",
        "grayscale":       "_filter_grayscale",
        "black_and_white": "_filter_grayscale",
        "warm":            "_filter_warm",
        "cool":            "_filter_cool",
        "vintage":         "_filter_vintage",
        "blur":            "_filter_blur",
        "sharpen":         "_filter_sharpen",
    }

    def apply(self, intent: Dict) -> Dict:
        import cv2

        scope      = intent.get("scope", "all")
        params     = intent.get("parameters", {})
        action     = intent.get("intent", "")
        manifest   = _load_manifest()
        scene_ids  = _scene_ids_from_scope(scope, manifest)

        # Determine which filter to apply
        if action == "make_scene_darker":
            filter_name = "darken"
            intensity   = params.get("intensity", 50) / 100.0
        elif action == "make_scene_brighter":
            filter_name = "brighter"
            intensity   = params.get("intensity", 50) / 100.0
        elif action == "apply_filter":
            filter_name = params.get("filter", "sepia")
            intensity   = params.get("intensity", 80) / 100.0
        elif action == "change_character_design":
            return self._regenerate_character(scope, params)
        else:
            filter_name = "sepia"
            intensity   = 0.8

        results = []
        for sid in scene_ids:
            in_path  = f"raw_scenes/scene_{sid:02d}.mp4"
            out_path = f"raw_scenes/scene_{sid:02d}_{filter_name}.mp4"
            if not Path(in_path).exists():
                in_path = f"raw_scenes/scene_{sid:02d}_raw.mp4"
            if not Path(in_path).exists():
                print(f"    [VideoFrameEditor] Scene {sid}: source not found")
                continue

            self._apply_filter_to_video(in_path, out_path,
                                        filter_name, intensity, cv2)
            results.append({"scene_id": sid, "filter": filter_name,
                            "intensity": intensity, "output": out_path})
            print(f"    [VideoFrameEditor] Scene {sid}: filter={filter_name} → {out_path}")

        return {"target": "video_frame", "action": action,
                "filter": filter_name, "affected_scenes": results}

    def _apply_filter_to_video(self, in_path: str, out_path: str,
                                filter_name: str, intensity: float, cv2):
        import imageio, gc
        reader     = imageio.get_reader(in_path)
        meta       = reader.get_meta_data()
        fps        = float(meta.get("fps", 24))
        total_f    = reader.count_frames()

        method_name = self.FILTER_REGISTRY.get(filter_name, "_filter_sepia")
        filter_fn   = getattr(self, method_name)

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for fi in range(total_f):
            try:
                frame = reader.get_data(fi)
            except Exception:
                break
            filtered = filter_fn(frame, intensity, cv2)
            writer.append_data(filtered)
        try:
            writer.close()
        except Exception:
            pass
        try:
            reader.close()
        except Exception:
            pass
        gc.collect()

    # ── Filter implementations ────────────────────────────────────────────────

    def _filter_darken(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        factor = max(0.1, 1.0 - intensity * 0.7)
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _filter_brighten(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        factor = 1.0 + intensity * 0.8
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _filter_sepia(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        img = frame.astype(np.float32)
        r = np.clip(img[:,:,0]*0.393 + img[:,:,1]*0.769 + img[:,:,2]*0.189, 0, 255)
        g = np.clip(img[:,:,0]*0.349 + img[:,:,1]*0.686 + img[:,:,2]*0.168, 0, 255)
        b = np.clip(img[:,:,0]*0.272 + img[:,:,1]*0.534 + img[:,:,2]*0.131, 0, 255)
        sepia = np.stack([r, g, b], axis=2).astype(np.uint8)
        return cv2.addWeighted(frame, 1 - intensity, sepia, intensity, 0)

    def _filter_grayscale(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        gray3 = cv2.cvtColor(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
        )
        return cv2.addWeighted(frame, 1 - intensity, gray3, intensity, 0)

    def _filter_warm(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        f = frame.astype(np.float32)
        f[:,:,0] = np.clip(f[:,:,0] * (1 + 0.3 * intensity), 0, 255)  # R up
        f[:,:,2] = np.clip(f[:,:,2] * (1 - 0.2 * intensity), 0, 255)  # B down
        return f.astype(np.uint8)

    def _filter_cool(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        f = frame.astype(np.float32)
        f[:,:,2] = np.clip(f[:,:,2] * (1 + 0.3 * intensity), 0, 255)  # B up
        f[:,:,0] = np.clip(f[:,:,0] * (1 - 0.2 * intensity), 0, 255)  # R down
        return f.astype(np.uint8)

    def _filter_vintage(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        # Sepia + vignette
        sepia   = self._filter_sepia(frame, intensity * 0.6, cv2)
        H, W    = frame.shape[:2]
        Y, X    = np.mgrid[0:H, 0:W]
        cx, cy  = W / 2, H / 2
        dist    = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        vignette = np.clip(1.0 - dist * intensity * 0.5, 0, 1)
        vignette = np.stack([vignette] * 3, axis=2)
        return (sepia.astype(np.float32) * vignette).astype(np.uint8)

    def _filter_blur(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        k = max(1, int(intensity * 15))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _filter_sharpen(self, frame: np.ndarray, intensity: float, cv2) -> np.ndarray:
        import numpy as np
        kernel = np.array([
            [0,   -intensity,        0],
            [-intensity, 1 + 4*intensity, -intensity],
            [0,   -intensity,        0],
        ], dtype=np.float32)
        sharpened = cv2.filter2D(frame.astype(np.float32), -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _regenerate_character(self, scope: str, params: Dict) -> Dict:
        """Regenerate character portrait with updated description."""
        char_name = scope.replace("character:", "") if "character:" in scope else "unknown"
        char_db   = _load_char_db()
        char = next((c for c in char_db["characters"]
                     if c["name"].upper() == char_name.upper()), None)
        if not char:
            return {"target": "video_frame", "error": f"Character {char_name} not found"}
        if params.get("description"):
            char["appearance"] = params["description"]
        from phase1.agents import ImageSynthesizerAgent
        paths = ImageSynthesizerAgent().run([char], output_dir="image_assets")
        return {"target": "video_frame", "action": "change_character_design",
                "character": char_name, "new_portrait": paths[0] if paths else None}


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO executor (speed / subtitle toggle)
# ─────────────────────────────────────────────────────────────────────────────

class VideoEditor:
    """Handles full-video edits: speed adjustment, subtitle on/off."""

    def apply(self, intent: Dict) -> Dict:
        action    = intent.get("intent", "")
        params    = intent.get("parameters", {})
        scope     = intent.get("scope", "all")
        manifest  = _load_manifest()
        scene_ids = _scene_ids_from_scope(scope, manifest)
        results   = []

        if action in ("speed_up_scene", "slow_down_scene"):
            factor = float(params.get("factor", 1.5 if "up" in action else 0.75))
            for sid in scene_ids:
                out = self._adjust_speed(sid, factor)
                results.append({"scene_id": sid, "factor": factor, "output": out})
                print(f"    [VideoEditor] Scene {sid}: speed {factor}x → {out}")

        elif action == "remove_subtitle":
            # Re-compose without subtitle by using raw clips
            for sid in scene_ids:
                raw = f"raw_scenes/scene_{sid:02d}_raw.mp4"
                dst = f"raw_scenes/scene_{sid:02d}_no_sub.mp4"
                if Path(raw).exists():
                    shutil.copy(raw, dst)
                    results.append({"scene_id": sid, "subtitle": False, "output": dst})
                    print(f"    [VideoEditor] Scene {sid}: subtitles removed")

        elif action == "add_subtitle":
            from phase3.agents import SubtitleBurnerAgent
            clips = [{"scene_id": sid,
                      "video_path": f"raw_scenes/scene_{sid:02d}.mp4",
                      "dialogue": next(
                          (s["dialogue"] for s in manifest["scenes"]
                           if s["scene_id"] == sid), [])}
                     for sid in scene_ids]
            paths = SubtitleBurnerAgent().run(clips)
            results = [{"scene_id": sid, "subtitle": True, "output": p}
                       for sid, p in zip(scene_ids, paths)]

        # Recompose final video after any video-level change
        if results:
            self._recompose_final()

        return {"target": "video", "action": action, "affected_scenes": results}

    def _adjust_speed(self, scene_id: int, factor: float) -> str:
        """Adjust video speed using imageio frame duplication/dropping."""
        import imageio, gc
        in_path  = f"raw_scenes/scene_{scene_id:02d}.mp4"
        out_path = f"raw_scenes/scene_{scene_id:02d}_speed_{factor}x.mp4"
        if not Path(in_path).exists():
            return in_path

        reader  = imageio.get_reader(in_path)
        meta    = reader.get_meta_data()
        fps     = float(meta.get("fps", 24))
        n       = reader.count_frames()
        frames  = []
        for i in range(n):
            try:
                frames.append(reader.get_data(i))
            except Exception:
                break
        try:
            reader.close()
        except Exception:
            pass

        # New frame indices at adjusted speed
        new_n   = int(n / factor)
        indices = (np.arange(new_n) * factor).astype(int)
        indices = np.clip(indices, 0, len(frames) - 1)

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for idx in indices:
            writer.append_data(frames[idx])
        try:
            writer.close()
        except Exception:
            pass
        gc.collect()
        return out_path

    def _recompose_final(self):
        """Re-run Phase 3 compositor to update final_output.mp4."""
        try:
            from phase3.workflow import run_phase3
            run_phase3(
                scene_manifest_path="outputs/scene_manifest.json",
                character_db_path="outputs/character_db.json",
                phase2_log_path="task_logs/phase2_log.json",
            )
        except Exception as e:
            print(f"    [VideoEditor] Recompose warning: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT executor (re-invoke Phase 1 + cascade)
# ─────────────────────────────────────────────────────────────────────────────

class ScriptEditor:
    """Handles script-level edits: full regeneration or mood change."""

    def apply(self, intent: Dict) -> Dict:
        action = intent.get("intent", "")
        params = intent.get("parameters", {})

        if action == "regenerate_script":
            return self._regenerate(params)
        elif action == "change_scene_mood":
            return self._change_mood(intent.get("scope", "all"), params)

        return {"target": "script", "error": f"Unknown script action: {action}"}

    def _regenerate(self, params: Dict) -> Dict:
        """Re-run Phase 1 with same or new prompt, then cascade."""
        from phase1.workflow import run_phase1
        from phase2.workflow import run_phase2
        from phase3.workflow import run_phase3

        prompt = params.get("prompt")
        if not prompt:
            # Read existing prompt from memory
            mem = Path("outputs/memory/scripts.json")
            if mem.exists():
                with open(mem) as f:
                    data = json.load(f)
                script = data.get("current_script", {})
                prompt = f"Remake this story: {script.get('title', 'A dramatic story')}"
            else:
                prompt = "A detective investigates a mysterious disappearance"

        p1 = run_phase1(prompt=prompt)
        if p1["status"] != "complete":
            return {"target": "script", "error": p1.get("error")}

        p2 = run_phase2(scene_manifest_path=p1["scene_manifest_path"],
                        character_db_path=p1["character_db_path"])
        if p2["status"] != "complete":
            return {"target": "script", "error": p2.get("error")}

        p3 = run_phase3(scene_manifest_path=p1["scene_manifest_path"],
                        character_db_path=p1["character_db_path"],
                        phase2_log_path="task_logs/phase2_log.json")

        return {
            "target":       "script",
            "action":       "regenerate_script",
            "new_manifest": p1["scene_manifest_path"],
            "final_video":  p3.get("final_video_path"),
            "status":       p3.get("status"),
        }

    def _change_mood(self, scope: str, params: Dict) -> Dict:
        """Patch mood description in scene manifest and re-run phase 3."""
        mood = params.get("mood", "darker")
        manifest = _load_manifest()

        scene_ids = _scene_ids_from_scope(scope, manifest)
        for scene in manifest["scenes"]:
            if scene["scene_id"] in scene_ids:
                scene["action_description"] += f" [mood: {mood}]"

        with open("outputs/scene_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        from phase3.workflow import run_phase3
        p3 = run_phase3(
            scene_manifest_path="outputs/scene_manifest.json",
            character_db_path="outputs/character_db.json",
            phase2_log_path="task_logs/phase2_log.json",
        )
        return {"target": "script", "action": "change_scene_mood",
                "mood": mood, "scenes": scene_ids,
                "final_video": p3.get("final_video_path")}


# ─────────────────────────────────────────────────────────────────────────────
# Unified Edit Executor
# ─────────────────────────────────────────────────────────────────────────────

class EditExecutor:
    """Routes classified intent to the correct editor."""

    def __init__(self):
        self._audio        = AudioEditor()
        self._video_frame  = VideoFrameEditor()
        self._video        = VideoEditor()
        self._script       = ScriptEditor()

    def execute(self, intent: Dict) -> Dict:
        target = intent.get("target", "unknown")

        if target == "audio":
            return self._audio.apply(intent)
        elif target == "video_frame":
            return self._video_frame.apply(intent)
        elif target == "video":
            return self._video.apply(intent)
        elif target == "script":
            return self._script.apply(intent)
        else:
            return {"error": f"Unknown target: {target}",
                    "raw_intent": intent}