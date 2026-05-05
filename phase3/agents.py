"""
Phase 3 Agent Definitions
==========================
ClipValidatorAgent    — validates Phase 2 outputs exist and are readable
BGMSelectorAgent      — selects/generates per-scene background music
SubtitleBurnerAgent   — burns dialogue subtitles onto each scene video
TransitionCompositorAgent — stitches scenes with cinematic transitions
TimingManifestAgent   — produces timing_manifest.json for Phase 4/5

All tools discovered and invoked via MCP registry.
"""

import json
import math
import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.registry import mcp

OUTPUTS_DIR  = Path("outputs")
RAW_DIR      = Path("raw_scenes")
AUDIO_DIR    = Path("audio_tracks")
BGM_DIR      = Path("outputs/bgm")
FINAL_DIR    = Path("outputs")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _font(size: int, bold: bool = False):
    import platform
    if platform.system() == "Windows":
        base = "C:/Windows/Fonts/"
        candidates = [
            base + ("arialbd.ttf" if bold else "arial.ttf"),
            base + ("calibrib.ttf" if bold else "calibri.ttf"),
        ]
    else:
        base = "/usr/share/fonts/truetype/dejavu/"
        candidates = [
            base + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _get_audio_duration_ms(audio_path: str) -> int:
    """Return audio duration in milliseconds. Works for WAV and MP3."""
    try:
        import wave as wave_mod
        with wave_mod.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate   = wf.getframerate()
            return int(frames / rate * 1000)
    except Exception:
        pass
    try:
        # Fallback: use imageio/ffmpeg to probe
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             audio_path],
            capture_output=True, text=True, timeout=10
        )
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return 5000   # 5 second default


def _get_video_duration_s(video_path: str) -> float:
    """Return video duration in seconds using imageio."""
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        meta   = reader.get_meta_data()
        reader.close()
        return float(meta.get("duration", 5.0))
    except Exception:
        return 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Clip Validator Agent
# ─────────────────────────────────────────────────────────────────────────────
class ClipValidatorAgent:
    """
    Validates that all Phase 2 output clips exist and are readable.
    Builds a clean clip_metadata list for downstream agents.
    MCP tool: commit_memory
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, phase2_log_path: str, scene_manifest_path: str) -> List[Dict]:
        print("  [ClipValidator] Validating Phase 2 scene clips...")

        with open(phase2_log_path) as f:
            phase2_log = json.load(f)
        with open(scene_manifest_path) as f:
            manifest = json.load(f)

        scene_map = {s["scene_id"]: s for s in manifest.get("scenes", [])}
        clips = []

        for entry in phase2_log:
            sid        = entry["scene_id"]
            video_path = entry.get("output_path", "")
            audio_path = entry.get("audio_path", "")

            # Normalise path separators (Windows backslashes → forward)
            video_path = video_path.replace("\\", "/")
            audio_path = audio_path.replace("\\", "/")

            if not Path(video_path).exists():
                print(f"    ✗ Scene {sid}: video not found at {video_path}")
                continue

            scene_data = scene_map.get(sid, {})
            duration_s = _get_video_duration_s(video_path)

            clips.append({
                "scene_id":    sid,
                "video_path":  video_path,
                "audio_path":  audio_path,
                "duration_s":  duration_s,
                "location":    scene_data.get("location", ""),
                "dialogue":    scene_data.get("dialogue", []),
                "action":      scene_data.get("action_description", ""),
                "mood":        _infer_mood(scene_data.get("location", "")),
            })
            print(f"    ✓ Scene {sid}: {duration_s:.1f}s  [{scene_data.get('location','')}]")

        mcp.invoke("commit_memory", {
            "collection": "phase3",
            "doc_id":     "validated_clips",
            "data":       {"clips": clips},
        })
        return clips


def _infer_mood(location: str) -> str:
    loc = location.lower()
    if any(w in loc for w in ("warehouse", "abandoned", "dark", "alley", "cellar")):
        return "tense"
    if any(w in loc for w in ("bar", "club", "jazz", "lounge", "tavern")):
        return "mysterious"
    if any(w in loc for w in ("office", "precinct", "station", "interrogation")):
        return "serious"
    if any(w in loc for w in ("apartment", "home", "house", "bedroom")):
        return "dramatic"
    if any(w in loc for w in ("street", "city", "outside", "park")):
        return "ambient"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  BGM Selector Agent
# ─────────────────────────────────────────────────────────────────────────────
class BGMSelectorAgent:
    """
    Generates per-scene background music as a synthesised WAV tone layer.
    In production: replace with MusicGen or royalty-free library API call.
    MCP tool: commit_memory
    """
    # Mood → (base_freq_hz, rhythm_bpm, harmonic_profile)
    MOOD_PARAMS = {
        "tense":      (55,  70,  [1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2]),
        "mysterious": (65,  80,  [1.0, 0.3, 0.0, 0.4, 0.0, 0.2, 0.0]),
        "serious":    (80,  90,  [1.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0]),
        "dramatic":   (70,  85,  [1.0, 0.2, 0.4, 0.1, 0.2, 0.0, 0.1]),
        "ambient":    (110, 100, [1.0, 0.5, 0.0, 0.3, 0.0, 0.1, 0.0]),
        "neutral":    (90,  95,  [1.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
    }

    def __init__(self):
        mcp.discover_tools()

    def run(self, clips: List[Dict]) -> List[str]:
        print("  [BGMSelector] Generating scene background music...")
        BGM_DIR.mkdir(parents=True, exist_ok=True)
        bgm_paths = []

        for clip in clips:
            sid      = clip["scene_id"]
            mood     = clip["mood"]
            duration = clip["duration_s"]
            out_path = str(BGM_DIR / f"bgm_scene_{sid}.wav")

            self._generate_bgm(mood, duration, out_path)
            bgm_paths.append(out_path)
            print(f"    ✓ Scene {sid} BGM ({mood}): {out_path}")

        mcp.invoke("commit_memory", {
            "collection": "phase3",
            "doc_id":     "bgm_paths",
            "data":       {"bgm_paths": bgm_paths},
        })
        return bgm_paths

    def _generate_bgm(self, mood: str, duration_s: float, out_path: str):
        """Synthesise a multi-harmonic ambient score using NumPy."""
        SAMPLE_RATE = 22050
        base_freq, bpm, harmonics = self.MOOD_PARAMS.get(mood, self.MOOD_PARAMS["neutral"])

        n_samples = int(SAMPLE_RATE * duration_s)
        t         = np.linspace(0, duration_s, n_samples, endpoint=False)

        # Build harmonic series
        wave_data = np.zeros(n_samples, dtype=np.float32)
        for i, amp in enumerate(harmonics):
            if amp > 0:
                freq = base_freq * (i + 1)
                wave_data += amp * np.sin(2 * np.pi * freq * t)

        # Add slow rhythmic pulse (LFO)
        lfo_rate  = bpm / 60.0
        lfo       = 0.6 + 0.4 * np.sin(2 * np.pi * lfo_rate * t * 0.25)
        wave_data *= lfo

        # Apply fade-in / fade-out (2 seconds each)
        fade_len = min(int(SAMPLE_RATE * 2), n_samples // 4)
        wave_data[:fade_len]  *= np.linspace(0, 1, fade_len)
        wave_data[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalise + scale to int16 at low volume (BGM should be quiet under dialogue)
        peak = np.max(np.abs(wave_data))
        if peak > 0:
            wave_data = wave_data / peak * 0.18   # 18% volume — sits under dialogue

        pcm = (wave_data * 32767).astype(np.int16)

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Subtitle Burner Agent
# ─────────────────────────────────────────────────────────────────────────────
class SubtitleBurnerAgent:
    """
    Burns timed dialogue subtitles directly onto each scene video.
    Uses imageio frame-by-frame processing — no ffmpeg binary required.
    MCP tool: commit_memory
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, clips: List[Dict]) -> List[str]:
        """
        Burn subtitles onto scene clips using MoviePy.
        MoviePy preserves the original audio track — imageio strips it.
        """
        print("  [SubtitleBurner] Burning subtitles onto scene clips...")
        import gc
        from moviepy import VideoFileClip
        from moviepy.video.VideoClip import ImageClip
        import tempfile, imageio

        subtitled_paths = []

        for clip in clips:
            sid        = clip["scene_id"]
            video_path = clip["video_path"]
            dialogues  = clip["dialogue"]
            out_path   = str(RAW_DIR / f"scene_{sid:02d}_subtitled.mp4")

            if not Path(video_path).exists():
                subtitled_paths.append(video_path)
                continue

            try:
                mv_clip    = VideoFileClip(video_path)
                fps        = mv_clip.fps or 24.0
                duration_s = mv_clip.duration
                total_f    = int(duration_s * fps)

                # Build subtitle schedule
                subs = self._build_subtitle_schedule(dialogues, duration_s, fps, scene_id=sid)

                # Process frames via MoviePy's fl_image (preserves audio)
                def _add_subtitle(get_frame, t):
                    frame     = get_frame(t)
                    frame_idx = int(t * fps)
                    img       = Image.fromarray(frame)
                    img       = self._burn_subtitle(img, frame_idx, subs)
                    return np.array(img)

                subtitled_clip = mv_clip.transform(_add_subtitle)

                # Write with audio preserved
                subtitled_clip.write_videofile(
                    out_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=fps,
                    logger=None,
                )
            except Exception as e:
                print(f"    ✗ Scene {sid} subtitle error: {e} — using original")
                out_path = video_path
            finally:
                try:
                    subtitled_clip.close()
                except Exception:
                    pass
                try:
                    mv_clip.close()
                except Exception:
                    pass
                gc.collect()

            subtitled_paths.append(out_path)
            print(f"    ✓ Scene {sid}: subtitles burned → {out_path}")

        mcp.invoke("commit_memory", {
            "collection": "phase3",
            "doc_id":     "subtitled_clips",
            "data":       {"subtitled_paths": subtitled_paths},
        })
        return subtitled_paths

    def _build_subtitle_schedule(
        self, dialogues: List[Dict], duration_s: float, fps: float,
        scene_id: int = 0
    ) -> List[Dict]:
        """
        Assign each dialogue line a start/end frame range.

        Derives timing from actual per-group audio files so subtitles
        stay in sync with the audio track.
        Falls back to equal division if audio files are not found.
        """
        if not dialogues:
            return []

        import glob as _glob, subprocess as _sp, json as _j

        def _audio_dur(path: str) -> float:
            try:
                r = _sp.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                     '-show_format', path],
                    capture_output=True, text=True, timeout=5
                )
                return float(_j.loads(r.stdout)['format']['duration'])
            except Exception:
                return None

        # Try audio-derived timing (same logic as video composer)
        grp_files = sorted(
            _glob.glob(f"audio_tracks/scene_{scene_id}_grp_*.mp3") +
            _glob.glob(f"audio_tracks/scene_{scene_id}_grp_*.wav")
        )

        PADDING_S = 0.3   # same padding as video composer

        if grp_files:
            grp_durations = [_audio_dur(f) or 3.0 for f in grp_files]
            groups = []
            for dlg in dialogues:
                sp = dlg.get("speaker", "")
                if groups and groups[-1]["speaker"] == sp:
                    groups[-1]["count"] += 1
                else:
                    groups.append({"speaker": sp, "count": 1})

            per_line_dur = []
            for i, grp in enumerate(groups):
                dur = grp_durations[i] if i < len(grp_durations) else 3.0
                per_dlg = dur / grp["count"]
                per_line_dur.extend([per_dlg] * grp["count"])
        else:
            # Fallback: equal division using total duration
            per_dlg_dur = duration_s / max(len(dialogues), 1)
            per_line_dur = [per_dlg_dur] * len(dialogues)

        schedule = []
        cursor   = 0
        for i, dlg in enumerate(dialogues):
            hold_f = max(int((per_line_dur[i] + PADDING_S) * fps), int(fps * 1.5))
            sf     = cursor
            ef     = cursor + hold_f
            schedule.append({
                "start_frame": sf,
                "end_frame":   ef,
                "speaker":     dlg.get("speaker", ""),
                "line":        dlg.get("line", ""),
            })
            cursor = ef

        return schedule

    def _burn_subtitle(
        self, img: Image.Image, frame_idx: int, subs: List[Dict]
    ) -> Image.Image:
        """Draw the current subtitle line onto the frame."""
        current = next(
            (s for s in subs if s["start_frame"] <= frame_idx < s["end_frame"]),
            None,
        )
        if not current or not current["line"]:
            return img

        W, H   = img.size
        draw   = ImageDraw.Draw(img)
        f_body = _font(16)
        f_name = _font(14, bold=True)

        line    = current["line"]
        speaker = current["speaker"]

        # Truncate long lines
        if len(line) > 85:
            line = line[:82] + "…"

        # Background pill
        BAR_H = 52
        bar   = Image.new("RGBA", (W, BAR_H), (0, 0, 0, 175))
        img.paste(bar, (0, H - BAR_H - 4), bar)

        draw = ImageDraw.Draw(img)

        # Speaker label
        if speaker:
            draw.text((12, H - BAR_H + 2), f"{speaker}:", font=f_name, fill=(100, 200, 255))

        # Dialogue text
        draw.text((12, H - BAR_H + 20), line, font=f_body, fill=(255, 255, 255))

        return img


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Transition Compositor Agent
# ─────────────────────────────────────────────────────────────────────────────
class TransitionCompositorAgent:
    """
    Stitches all subtitled scene clips into a single final_output.mp4.
    Adds cinematic transitions between scenes:
      - Fade-to-black (out) then fade-from-black (in)
      - Mixable with cross-dissolve for adjacent scenes
    Also mixes BGM under the dialogue audio.
    MCP tool: commit_memory
    """
    TRANSITION_FRAMES = 18   # ~0.75s at 24fps

    def __init__(self):
        mcp.discover_tools()

    def run(
        self,
        subtitled_clips: List[str],
        bgm_paths: List[str],
        clips_meta: List[Dict],
        output_path: str,
    ) -> str:
        """
        Stitch scenes into final_output.mp4 using MoviePy throughout.
        MoviePy preserves the audio track from each scene clip, so dialogue
        audio flows all the way through to the final output.
        imageio is NOT used here — it writes video-only (no audio stream).
        """
        print("  [Compositor] Compositing final video with transitions + BGM...")
        import gc
        from moviepy import (
            VideoFileClip, AudioFileClip, CompositeAudioClip,
            concatenate_videoclips, ColorClip,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fps     = 24.0
        TF      = self.TRANSITION_FRAMES
        TF_s    = TF / fps          # transition duration in seconds

        composited_clips = []
        timing_entries   = []
        frame_cursor     = 0

        for clip_path, meta in zip(subtitled_clips, clips_meta):
            if not Path(clip_path).exists():
                print(f"    ✗ Missing clip: {clip_path} — skipping")
                continue

            try:
                clip = VideoFileClip(clip_path)
            except Exception as e:
                print(f"    ✗ Cannot open {clip_path}: {e}")
                continue

            fps = clip.fps or 24.0
            W, H = clip.size

            scene_start_frame = frame_cursor

            # ── Fade-in: black clip that cross-dissolves into scene ───────
            black_in = ColorClip(size=(W, H), color=[0, 0, 0], duration=TF_s)
            fade_in  = (
                black_in
                .with_effects([])      # no effects needed — we use opacity
            )
            # Use clip's own fade-in effect via moviepy
            from moviepy.video.fx import FadeIn, FadeOut
            clip_with_fades = clip.with_effects([
                FadeIn(TF_s),
                FadeOut(TF_s),
            ])

            composited_clips.append(clip_with_fades)
            frame_cursor += int(clip.duration * fps) + 2 * TF

            scene_end_frame = frame_cursor
            timing_entries.append({
                "scene_id":    meta["scene_id"],
                "location":    meta["location"],
                "start_frame": scene_start_frame,
                "end_frame":   scene_end_frame,
                "start_ms":    int(scene_start_frame / fps * 1000),
                "end_ms":      int(scene_end_frame   / fps * 1000),
                "audio_path":  meta.get("audio_path", ""),
                "mood":        meta.get("mood", "neutral"),
            })
            print(f"    ✓ Scene {meta['scene_id']} composited  "
                  f"[{timing_entries[-1]['start_ms']}ms → {timing_entries[-1]['end_ms']}ms]")

        if not composited_clips:
            raise RuntimeError("No valid scene clips found for compositing.")

        # ── Concatenate all scenes ────────────────────────────────────────
        final_video = concatenate_videoclips(composited_clips, method="compose")

        # ── Mix BGM under existing dialogue audio ─────────────────────────
        bgm_audio_clips = []
        cursor_s = 0.0
        for bgm_path, meta in zip(bgm_paths, clips_meta):
            scene_dur = meta.get("duration_s", 5.0) + 2 * TF_s
            if Path(bgm_path).exists():
                try:
                    bgm = (
                        AudioFileClip(bgm_path)
                        .subclipped(0, min(AudioFileClip(bgm_path).duration, scene_dur))
                        .with_start(cursor_s)
                        .multiply_volume(0.18)   # BGM at 18% — quiet under dialogue
                    )
                    bgm_audio_clips.append(bgm)
                except Exception:
                    pass
            cursor_s += scene_dur

        if bgm_audio_clips:
            total_dur = final_video.duration
            bgm_track = CompositeAudioClip(bgm_audio_clips).subclipped(0, total_dur)
            if final_video.audio is not None:
                # Dialogue audio + BGM together
                mixed = CompositeAudioClip([final_video.audio, bgm_track])
            else:
                mixed = bgm_track
            final_video = final_video.with_audio(mixed)

        # ── Write output ──────────────────────────────────────────────────
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            logger=None,
        )

        # Clean up
        try:
            final_video.close()
            for c in composited_clips:
                c.close()
            gc.collect()
        except Exception:
            pass

        # Commit
        mcp.invoke("commit_memory", {
            "collection": "phase3",
            "doc_id":     "final_video",
            "data":       {"path": output_path, "timing": timing_entries},
        })

        return output_path, timing_entries

    def _mix_bgm(
        self,
        video_path: str,
        bgm_paths: List[str],
        clips_meta: List[Dict],
        output_path: str,
        fps: float,
    ):
        """Mix per-scene BGM under the final video using MoviePy."""
        try:
            from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
            import gc

            video = VideoFileClip(video_path)
            total_dur = video.duration

            # Build concatenated BGM track matching total video duration
            bgm_clips  = []
            cursor_s   = 0.0
            TF_s       = self.TRANSITION_FRAMES / fps

            for bgm_path, meta in zip(bgm_paths, clips_meta):
                if not Path(bgm_path).exists():
                    cursor_s += meta["duration_s"] + 2 * TF_s
                    continue
                try:
                    bgm  = AudioFileClip(bgm_path).with_start(cursor_s)
                    bgm_clips.append(bgm)
                except Exception:
                    pass
                cursor_s += meta["duration_s"] + 2 * TF_s

            if not bgm_clips:
                video.close()
                return

            bgm_track = CompositeAudioClip(bgm_clips).subclipped(0, total_dur)

            # Composite: original audio + BGM
            if video.audio is not None:
                final_audio = CompositeAudioClip([video.audio, bgm_track])
            else:
                final_audio = bgm_track

            final = video.with_audio(final_audio)
            final.write_videofile(output_path, codec="libx264",
                                  audio_codec="aac", fps=fps, logger=None)
            try:
                final.close(); video.close(); gc.collect()
            except Exception:
                pass

        except Exception as e:
            print(f"    [BGM mix] Warning: {e} — skipping BGM mix")


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Timing Manifest Agent
# ─────────────────────────────────────────────────────────────────────────────
class TimingManifestAgent:
    """
    Writes the timing_manifest.json consumed by Phase 4 and Phase 5.
    Schema: { scene_id, audio_file, start_ms, end_ms, location, mood }
    MCP tool: commit_memory
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, timing_entries: List[Dict], output_path: str) -> str:
        print("  [TimingManifest] Writing timing_manifest.json...")

        manifest = {
            "version":  1,
            "entries":  timing_entries,
            "total_scenes": len(timing_entries),
            "total_duration_ms": timing_entries[-1]["end_ms"] if timing_entries else 0,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

        mcp.invoke("commit_memory", {
            "collection": "phase3",
            "doc_id":     "timing_manifest",
            "data":       manifest,
        })

        print(f"    ✓ timing_manifest.json → {output_path}")
        return output_path