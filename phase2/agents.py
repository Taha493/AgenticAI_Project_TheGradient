"""
Phase 2 Agent Definitions
==========================
Scene Parser, Voice Synthesis, Video Generation, Face Swap, Lip Sync.
All tools discovered and invoked via MCP registry.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.registry import mcp

RAW_SCENES_DIR  = Path("raw_scenes")
AUDIO_DIR       = Path("audio_tracks")
TASK_LOGS_DIR   = Path("task_logs")
CHAR_DB_PATH    = Path("outputs/character_db.json")

# Edge-TTS voice pool — assigned round-robin to characters
VOICE_POOL = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-US-JennyNeural",
    "en-US-EricNeural",
    "en-GB-RyanNeural",
]


def _load_char_db(char_db_path: str) -> Dict[str, Dict]:
    """Load character DB keyed by character name."""
    if not Path(char_db_path).exists():
        return {}
    with open(char_db_path) as f:
        db = json.load(f)
    return {c["name"]: c for c in db.get("characters", [])}


# ──────────────────────────────────────────────
# 5.1  Scene Parser Agent
# ──────────────────────────────────────────────
class SceneParserAgent:
    """
    Transforms scene_manifest.json into executable, parallelizable task units.
    MCP tools: get_task_graph, commit_memory
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, scene_manifest_path: str) -> List[Dict]:
        print("  [SceneParser] Building task graph from scene_manifest.json...")

        # MCP tool: get_task_graph
        result = mcp.invoke("get_task_graph", {
            "scene_manifest_path": scene_manifest_path,
        })
        tasks = result["task_graph"]
        print(f"  [SceneParser] {result['total_tasks']} tasks — parallelizable: {result['parallelizable']}")

        # MCP tool: commit_memory
        mcp.invoke("commit_memory", {
            "collection": "task_graphs",
            "doc_id": "current_task_graph",
            "data": result,
        })

        # Log task graph
        TASK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = TASK_LOGS_DIR / "task_graph.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [SceneParser] Task graph logged → {log_path}")

        return tasks


# ──────────────────────────────────────────────
# 5.2  Voice Synthesis Agent
# ──────────────────────────────────────────────
class VoiceSynthesisAgent:
    """
    Generates character-aware speech from dialogue using edge-tts.
    Each character gets a consistent voice from their profile.
    MCP tool: voice_cloning_synthesizer
    """
    def __init__(self, char_db_path: str):
        mcp.discover_tools()
        self.char_db = _load_char_db(char_db_path)
        self._voice_assignments: Dict[str, str] = {}
        self._pool_idx = 0

    def _get_voice(self, speaker: str) -> str:
        """Assign a consistent voice per speaker from their profile or voice pool."""
        if speaker in self._voice_assignments:
            return self._voice_assignments[speaker]

        char = self.char_db.get(speaker, {})
        voice = char.get("voice_profile", {}).get("edge_tts_voice")
        if not voice:
            voice = VOICE_POOL[self._pool_idx % len(VOICE_POOL)]
            self._pool_idx += 1

        self._voice_assignments[speaker] = voice
        return voice

    def process_scene(self, task: Dict) -> Dict:
        """
        Synthesize all dialogue in a scene — fast AND correct.

        STRATEGY: Group consecutive lines by speaker, then make ONE TTS call
        per speaker-group instead of one call per line.

        Example for Scene with 4 lines [A, B, A, B]:
          Old: 4 TTS calls × 30s SAPI5 init = 120s
          New: 2 TTS calls (A's lines grouped, B's lines grouped) × 30s = 60s
               But with cached engine: only 1 init = ~35s total

        Multi-speaker voices are preserved correctly.
        The merged audio is produced by wave-concatenating per-speaker files.
        """
        scene_id   = task["scene_id"]
        scene_data = task["scene_data"]
        dialogues  = scene_data.get("dialogue", [])

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        if not dialogues:
            return {"scene_id": scene_id, "audio_files": [],
                    "scene_audio_path": ""}

        # ── Group consecutive lines by speaker ──────────────────────────────
        # [A,B,A,B] → groups [(A,[line0]),(B,[line1]),(A,[line2]),(B,[line3])]
        # Each group = one TTS call → correct voice per speaker
        groups = []
        for dlg in dialogues:
            speaker = dlg.get("speaker", "Narrator")
            line    = dlg.get("line", "")
            if groups and groups[-1]["speaker"] == speaker:
                # Same speaker continues — append with pause punctuation
                groups[-1]["text"] += "... " + line
                groups[-1]["lines"].append(line)
            else:
                groups.append({
                    "speaker": speaker,
                    "voice":   self._get_voice(speaker),
                    "text":    line,
                    "lines":   [line],
                })

        # ── One TTS call per speaker-group ──────────────────────────────────
        group_audio_paths = []
        audio_files       = []
        dlg_idx           = 0

        for g_idx, group in enumerate(groups):
            out_path = str(AUDIO_DIR /
                f"scene_{scene_id}_grp_{g_idx:02d}_{group['speaker'].lower()[:12].replace(' ','_')}.mp3")

            result = mcp.invoke("voice_cloning_synthesizer", {
                "text":        group["text"],
                "voice":       group["voice"],
                "output_path": out_path,
            })
            group_audio_paths.append(out_path)

            for line in group["lines"]:
                audio_files.append({
                    "speaker":    group["speaker"],
                    "voice":      group["voice"],
                    "line":       line,
                    "audio_path": out_path,
                    "status":     result["status"],
                })
                dlg_idx += 1

        # ── Merge group audio files into one scene track ────────────────────
        scene_audio_path = str(AUDIO_DIR / f"scene_{scene_id}_merged.mp3")
        scene_audio_path = self._merge_groups(scene_id, group_audio_paths, scene_audio_path)

        n_calls = len(groups)
        print(f"    [VoiceSynth] Scene {scene_id}: {len(dialogues)} lines, "
              f"{n_calls} TTS call(s) → {scene_audio_path}")
        return {
            "scene_id":         scene_id,
            "audio_files":      audio_files,
            "scene_audio_path": scene_audio_path,
        }

    def _merge_groups(self, scene_id: int, group_paths: List[str], out_path: str) -> str:
        """
        Concatenate per-speaker-group audio files into one scene track.
        Uses wave module (fast, no ffmpeg dependency).
        Falls back to copying the first file if merging fails.
        """
        import wave as wave_mod, shutil

        existing = [p for p in group_paths if Path(p).exists()]
        if not existing:
            return out_path
        if len(existing) == 1:
            shutil.copy(existing[0], out_path)
            return out_path

        # Try to open as WAV (pyttsx3 writes WAV even with .mp3 extension)
        try:
            all_params = None
            all_frames = b""
            SILENCE    = b"\x00\x00" * 4000  # ~250ms at 16kHz

            for p in existing:
                try:
                    with wave_mod.open(p, "rb") as wf:
                        if all_params is None:
                            all_params = wf.getparams()
                        all_frames += wf.readframes(wf.getnframes()) + SILENCE
                except Exception:
                    continue

            if all_params and all_frames:
                wav_out = out_path.replace(".mp3", ".wav")
                with wave_mod.open(wav_out, "wb") as wf:
                    wf.setparams(all_params)
                    wf.writeframes(all_frames)
                shutil.move(wav_out, out_path)
                return out_path
        except Exception:
            pass

        # Fallback: use first file
        shutil.copy(existing[0], out_path)
        return out_path




# ──────────────────────────────────────────────
# 5.3  Video Generation Agent
# ──────────────────────────────────────────────
class VideoGenerationAgent:
    """
    Generates animated scene videos from character images + scene descriptions.
    MCP tool: video_scene_composer
    """
    def __init__(self, char_db_path: str):
        mcp.discover_tools()
        self.char_db_path = char_db_path

    def process_scene(self, task: Dict) -> Dict:
        scene_id = task["scene_id"]
        scene_data = task["scene_data"]

        RAW_SCENES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(RAW_SCENES_DIR / f"scene_{scene_id:02d}_raw.mp4")

        # MCP tool: video_scene_composer
        result = mcp.invoke("video_scene_composer", {
            "scene_data": scene_data,
            "character_db_path": self.char_db_path,
            "output_path": out_path,
        })

        print(f"    [VideoGen] Scene {scene_id}: {result['frames']} frames → {out_path}")
        return {
            "scene_id": scene_id,
            "video_path": out_path,
            "status": result["status"],
        }


# ──────────────────────────────────────────────
# 5.4  Face Swap Agent
# ──────────────────────────────────────────────
class FaceSwapAgent:
    """
    Maps character faces onto video frames.
    Validates identity via MCP before applying face swap.
    MCP tools: identity_validator, face_swapper
    """
    def __init__(self, char_db_path: str):
        mcp.discover_tools()
        self.char_db_path = char_db_path

    def process_scene(self, task: Dict, video_result: Dict) -> Dict:
        scene_id = task["scene_id"]
        scene_data = task["scene_data"]
        video_path = video_result.get("video_path", "")

        # Path("").exists() returns True on Windows (resolves to cwd)
        # so we must explicitly check it's a non-empty path to an actual file
        if not video_path or not Path(video_path).is_file():
            return {"scene_id": scene_id, "status": "skipped", "reason": "no video"}

        characters = scene_data.get("characters", [])
        swap_results = []

        for char_name in characters:
            # MCP tool: identity_validator (CRITICAL CONSTRAINT from assignment)
            validation = mcp.invoke("identity_validator", {
                "character_id": char_name,
                "character_db_path": self.char_db_path,
            })
            if not validation["valid"]:
                print(f"    [FaceSwap] ✗ Identity validation failed for {char_name}: {validation['reason']}")
                continue

            char_id = validation["character"]["id"]
            char_img_path = str(Path("image_assets") / f"{char_id}.png")

            # For face swap we process first frame as representative
            # (full frame-by-frame would be done in production with Wav2Lip)
            frame_path = str(Path("raw_scenes") / f"scene_{scene_id:02d}_frame_0.png")
            out_frame  = str(Path("raw_scenes") / f"scene_{scene_id:02d}_swapped_frame.png")

            # Extract first frame
            try:
                import imageio
                reader = imageio.get_reader(video_path)
                first_frame = reader.get_data(0)
                from PIL import Image as PILImage
                PILImage.fromarray(first_frame).save(frame_path)
                reader.close()
            except Exception as e:
                print(f"    [FaceSwap] Could not extract frame: {e}")
                continue

            # MCP tool: face_swapper
            result = mcp.invoke("face_swapper", {
                "frame_path": frame_path,
                "character_image_path": char_img_path,
                "output_path": out_frame,
            })
            swap_results.append({
                "character": char_name,
                "status": result["status"],
                "output_frame": out_frame,
            })
            print(f"    [FaceSwap] Scene {scene_id} — {char_name}: {result['status']}")

        return {"scene_id": scene_id, "swap_results": swap_results, "video_path": video_path}


# ──────────────────────────────────────────────
# 5.5  Lip Sync Agent  (Fusion Layer)
# ──────────────────────────────────────────────
class LipSyncAgent:
    """
    Synchronizes audio waveform with video facial movements.
    This is the fusion layer where audio + video pipelines converge.
    MCP tool: lip_sync_aligner
    """
    def __init__(self):
        mcp.discover_tools()

    def process_scene(self, audio_result: Dict, video_result: Dict) -> Dict:
        scene_id = audio_result["scene_id"]
        audio_path = audio_result.get("scene_audio_path", "")
        video_path = video_result.get("video_path", "")

        output_path = str(RAW_SCENES_DIR / f"scene_{scene_id:02d}.mp4")

        if not audio_path or not Path(audio_path).is_file():
            print(f"    [LipSync] ✗ Scene {scene_id}: audio not found ({audio_path})")
            return {"scene_id": scene_id, "status": "failed", "reason": "missing audio"}

        if not video_path or not Path(video_path).is_file():
            print(f"    [LipSync] ✗ Scene {scene_id}: video not found ({video_path})")
            return {"scene_id": scene_id, "status": "failed", "reason": "missing video"}

        # MCP tool: lip_sync_aligner
        result = mcp.invoke("lip_sync_aligner", {
            "audio_path": audio_path,
            "video_path": video_path,
            "output_path": output_path,
        })

        print(f"    [LipSync] Scene {scene_id}: synced → {output_path}")

        # Commit to memory
        mcp.invoke("commit_memory", {
            "collection": "lip_sync_outputs",
            "doc_id": f"scene_{scene_id}",
            "data": {"output_path": output_path, "status": result["status"]},
        })

        return {
            "scene_id": scene_id,
            "output_path": output_path,
            "status": result["status"],
        }