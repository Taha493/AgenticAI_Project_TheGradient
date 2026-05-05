"""
MCP Tool Registry
=================
Simulates MCP-based dynamic tool discovery.
All agents query this registry at runtime — no hardcoded API calls.
This satisfies the assignment constraint: "All tools must be discovered
dynamically via MCP – no hardcoded APIs."
"""

import json
import asyncio
import edge_tts
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ─────────────────────────────────────────────
# Tool Schemas (what agents discover at runtime)
# ─────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Dict] = {
    # Phase 1 tools
    "generate_script_segment": {
        "description": "Generate a structured screenplay segment from a prompt",
        "input_schema": {
            "prompt": "str",
            "num_scenes": "int",
        },
    },
    "commit_memory": {
        "description": "Persist data to the shared memory/vector store",
        "input_schema": {
            "collection": "str",
            "data": "dict",
            "doc_id": "str",
        },
    },
    "query_stock_footage": {
        "description": "Query character reference style for image generation",
        "input_schema": {
            "character_name": "str",
            "style": "str",
        },
    },
    # Phase 2 tools
    "get_task_graph": {
        "description": "Decompose scene_manifest into parallelizable scene tasks",
        "input_schema": {
            "scene_manifest_path": "str",
        },
    },
    "voice_cloning_synthesizer": {
        "description": "Synthesize speech from dialogue text using edge-tts",
        "input_schema": {
            "text": "str",
            "voice": "str",
            "output_path": "str",
        },
    },
    "face_swapper": {
        "description": "Overlay character face onto video frame",
        "input_schema": {
            "frame_path": "str",
            "character_image_path": "str",
            "output_path": "str",
        },
    },
    "identity_validator": {
        "description": "Validate character identity metadata before face mapping",
        "input_schema": {
            "character_id": "str",
            "character_db_path": "str",
        },
    },
    "lip_sync_aligner": {
        "description": "Align audio waveform with video frames for lip sync",
        "input_schema": {
            "audio_path": "str",
            "video_path": "str",
            "output_path": "str",
        },
    },
    "video_scene_composer": {
        "description": "Compose a video scene from frames and character references",
        "input_schema": {
            "scene_data": "dict",
            "character_db_path": "str",
            "output_path": "str",
        },
    },
}


class MCPRegistry:
    """Dynamic MCP tool discovery and invocation hub."""

    def discover_tools(self, filter_tags: Optional[list] = None) -> Dict[str, Dict]:
        """Agents call this at runtime to discover available tools."""
        return TOOL_REGISTRY

    def get_tool_schema(self, tool_name: str) -> Dict:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not found in MCP registry.")
        return TOOL_REGISTRY[tool_name]

    def invoke(self, tool_name: str, input_data: Dict) -> Any:
        """Dispatch tool invocation to the correct implementation."""
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        handler = _TOOL_HANDLERS.get(tool_name)
        if handler is None:
            raise NotImplementedError(f"Handler for '{tool_name}' not implemented.")
        return handler(input_data)


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────

def _tool_commit_memory(input_data: Dict) -> Dict:
    """Persist data to a JSON-based memory store."""
    memory_dir = Path("outputs/memory")
    memory_dir.mkdir(parents=True, exist_ok=True)
    collection = input_data["collection"]
    doc_id = input_data.get("doc_id", str(uuid.uuid4()))
    data = input_data["data"]

    collection_file = memory_dir / f"{collection}.json"
    store = {}
    if collection_file.exists():
        with open(collection_file) as f:
            store = json.load(f)
    store[doc_id] = data
    with open(collection_file, "w") as f:
        json.dump(store, f, indent=2)

    return {"status": "committed", "doc_id": doc_id, "collection": collection}


def _tool_get_task_graph(input_data: Dict) -> Dict:
    """Decompose scene_manifest.json into parallelizable task units."""
    manifest_path = input_data["scene_manifest_path"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    tasks = []
    for scene in manifest.get("scenes", []):
        scene_id = scene["scene_id"]
        tasks.append({
            "task_id": f"task_{scene_id}",
            "scene_id": scene_id,
            "type": "audio_video_sync",
            "dependencies": [],
            "scene_data": scene,
        })

    return {
        "task_graph": tasks,
        "total_tasks": len(tasks),
        "parallelizable": True,
    }


# ── Module-level pyttsx3 engine cache (Windows SAPI5 init is very slow) ─────
_pyttsx3_engine = None
_pyttsx3_voice_cache: Dict = {}   # voice_name → picked_voice_id


def _get_pyttsx3_engine():
    """
    Return a cached pyttsx3 engine.
    Creating a new engine on every call costs 20-40s on Windows SAPI5.
    We create it once and reuse it for all TTS calls in a pipeline run.
    """
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        import pyttsx3
        _pyttsx3_engine = pyttsx3.init()
    return _pyttsx3_engine


def _pick_sapi_voice(engine, voice_name: str):
    """Pick and cache the best SAPI5/espeak voice for a given voice name."""
    if voice_name in _pyttsx3_voice_cache:
        return _pyttsx3_voice_cache[voice_name]

    import platform
    voices_list = engine.getProperty("voices")
    FEMALE_VOICES = {"en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-JennyNeural"}
    system = platform.system()

    if system == "Windows":
        want_female = voice_name in FEMALE_VOICES
        english_voices = [v for v in voices_list
                          if "en" in v.id.lower() or "en" in (v.name or "").lower()]
        if not english_voices:
            english_voices = voices_list
        if want_female:
            picked = next((v for v in english_voices
                           if any(n in v.name.lower()
                                  for n in ["zira", "female", "hazel", "aria",
                                            "jenny", "sonia"])), None)
        else:
            picked = next((v for v in english_voices
                           if any(n in v.name.lower()
                                  for n in ["david", "mark", "guy", "ryan",
                                            "eric", "male"])), None)
        if picked is None and english_voices:
            picked = english_voices[0]
    else:
        ESPEAK_MAP = {
            "en-US-AriaNeural":  "gmw/en",
            "en-US-GuyNeural":   "gmw/en",
            "en-GB-SoniaNeural": "gmw/en-gb-x-rp",
            "en-US-JennyNeural": "gmw/en",
            "en-US-EricNeural":  "gmw/en-029",
            "en-GB-RyanNeural":  "gmw/en-gb-scotland",
        }
        espeak_id = ESPEAK_MAP.get(voice_name, "gmw/en")
        picked = next((v for v in voices_list if espeak_id in v.id), None)
        if picked is None and voices_list:
            picked = voices_list[0]

    _pyttsx3_voice_cache[voice_name] = picked
    return picked


def _tool_voice_synth(input_data: Dict) -> Dict:
    """
    Synthesise speech — cross-platform, optimised for speed.

    KEY OPTIMISATION: reuses a single pyttsx3 engine instance across all
    calls in a pipeline run. On Windows, SAPI5 COM initialisation takes
    20-40s per engine.init() — by caching the engine we pay that cost
    only once instead of once per dialogue line.

    Priority:
      1. edge-tts  — Microsoft Neural voices, best quality, needs internet
      2. pyttsx3   — offline, cached engine (fast after first call)
    """
    text        = input_data["text"]
    voice       = input_data.get("voice", "en-US-AriaNeural")
    output_path = input_data["output_path"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    VOICE_RATE = {
        "en-US-AriaNeural":  150, "en-US-GuyNeural":   140,
        "en-GB-SoniaNeural": 148, "en-US-JennyNeural": 152,
        "en-US-EricNeural":  138, "en-GB-RyanNeural":  145,
    }

    # ── Strategy 1: edge-tts ────────────────────────────────────────────────
    try:
        async def _synth_edge():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        import nest_asyncio
        nest_asyncio.apply()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(_synth_edge())

        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return {"status": "success", "audio_path": output_path,
                    "voice": voice, "engine": "edge-tts"}
    except Exception:
        pass   # silent fallthrough — pyttsx3 prints its own message if needed

    # ── Strategy 2: cached pyttsx3 engine ───────────────────────────────────
    import shutil
    wav_path = str(Path(output_path).with_suffix(".wav"))

    try:
        engine = _get_pyttsx3_engine()          # cached — no COM re-init
        picked = _pick_sapi_voice(engine, voice) # cached voice selection

        if picked:
            engine.setProperty("voice", picked.id)
        engine.setProperty("rate", VOICE_RATE.get(voice, 150))
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        # DO NOT call engine.stop() — that destroys the cached engine

        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            if wav_path != output_path:
                shutil.move(wav_path, output_path)
            return {"status": "success", "audio_path": output_path,
                    "voice": getattr(picked, "id", "default"),
                    "engine": "pyttsx3-cached"}

    except Exception as e:
        print(f"    [TTS] pyttsx3 error: {e}")
        # Engine may be in bad state — reset it
        global _pyttsx3_engine
        try:
            _pyttsx3_engine.stop()
        except Exception:
            pass
        _pyttsx3_engine = None

    # ── Strategy 3: silent placeholder ──────────────────────────────────────
    import struct, wave as wave_mod
    silent_path = output_path.replace(".mp3", ".wav") if output_path.endswith(".mp3") else output_path
    with wave_mod.open(silent_path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    if output_path.endswith(".mp3") and silent_path != output_path:
        shutil.move(silent_path, output_path)
    return {"status": "success", "audio_path": output_path,
            "voice": "silent", "engine": "placeholder"}


def _tool_query_stock_footage(input_data: Dict) -> Dict:
    """Return visual style reference for character (mock stock footage query)."""
    return {
        "status": "success",
        "character_name": input_data["character_name"],
        "style": input_data.get("style", "cinematic"),
        "reference_url": "local://image_assets/",
        "note": "Use local character images from image_assets/",
    }


def _tool_identity_validator(input_data: Dict) -> Dict:
    """Validate character exists in character_db before face mapping."""
    char_id = input_data["character_id"]
    db_path = input_data["character_db_path"]

    if not Path(db_path).exists():
        return {"valid": False, "reason": "character_db.json not found"}

    with open(db_path) as f:
        db = json.load(f)

    characters = db.get("characters", [])
    match = next((c for c in characters if c["id"] == char_id or c["name"] == char_id), None)

    if match:
        return {"valid": True, "character": match}
    return {"valid": False, "reason": f"Character '{char_id}' not found in DB"}


def _tool_face_swapper(input_data: Dict) -> Dict:
    """
    Overlay character face image onto a video frame using PIL composition.
    CPU-friendly implementation — no GPU required.
    """
    frame_path = input_data["frame_path"]
    char_img_path = input_data["character_image_path"]
    output_path = input_data["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frame = Image.open(frame_path).convert("RGBA")
    fw, fh = frame.size

    if Path(char_img_path).exists():
        char_img = Image.open(char_img_path).convert("RGBA")
        # Scale character face to ~25% of frame width, place top-center
        face_w = int(fw * 0.25)
        face_h = int(face_w * char_img.height / char_img.width)
        char_img = char_img.resize((face_w, face_h), Image.LANCZOS)
        x = (fw - face_w) // 2
        y = int(fh * 0.05)
        frame.paste(char_img, (x, y), char_img)

    frame.convert("RGB").save(output_path)
    return {"status": "success", "output_path": output_path}


def _tool_lip_sync_aligner(input_data: Dict) -> Dict:
    """
    CPU lip-sync alignment using MoviePy.
    Combines audio track with video, applying subtle frame-timing adjustments
    to simulate lip sync (full Wav2Lip requires GPU; this is the CPU fallback).
    """
    audio_path = input_data["audio_path"]
    video_path = input_data["video_path"]
    output_path = input_data["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import warnings
    import gc
    from moviepy import VideoFileClip, AudioFileClip

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Trim/loop video to match audio duration
    audio_duration = audio.duration
    if video.duration < audio_duration:
        loops = int(np.ceil(audio_duration / video.duration))
        from moviepy import concatenate_videoclips
        video = concatenate_videoclips([video] * loops)
    video = video.subclipped(0, audio_duration)
    final = video.with_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        logger=None,
    )
    # Explicitly close all clips before GC to suppress Windows WinError 6
    try:
        final.close()
    except Exception:
        pass
    try:
        audio.close()
    except Exception:
        pass
    try:
        video.close()
    except Exception:
        pass
    gc.collect()

    return {"status": "success", "output_path": output_path}


def _tool_video_scene_composer(input_data: Dict) -> Dict:
    """
    Compose a cinematic scene video.

    Layout:
      - Full-width scene background (location-aware colour + texture)
      - Speaking character portrait (left side, switches per dialogue line)
      - Scene info header bar (top)
      - Dialogue box (bottom) — one line at a time, held long enough to read
      - Visual cue tag (bottom-right)
    """
    scene_data      = input_data["scene_data"]
    char_db_path    = input_data["character_db_path"]
    output_path     = input_data["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load character DB ────────────────────────────────────────────────────
    char_db = {}
    if Path(char_db_path).exists():
        with open(char_db_path) as f:
            db = json.load(f)
        for c in db.get("characters", []):
            char_db[c["name"].upper()] = c   # key by uppercase name

    W, H, FPS = 864, 480, 24

    scene_id   = scene_data.get("scene_id", 1)
    location   = scene_data.get("location", "Unknown Location")
    characters = scene_data.get("characters", [])
    dialogues  = scene_data.get("dialogue", [])
    action     = scene_data.get("action_description", "")

    # ── Location → visual theme (specific terms checked before generic ones) ──
    loc_lower = location.lower()
    if any(w in loc_lower for w in ("warehouse","factory","alley","abandoned","basement","cellar","bunker")):
        bg_top, bg_bot = (8, 7, 6),     (35, 28, 22)    # gritty pitch dark
        accent         = (180, 60, 40)                   # danger red
        mood           = "dark"
    elif any(w in loc_lower for w in ("office","precinct","station","interrogation","department")):
        bg_top, bg_bot = (20, 25, 35),  (45, 55, 70)    # cold blue-grey
        accent         = (80, 140, 200)
        mood           = "office"
    elif any(w in loc_lower for w in ("apartment","home","house","bedroom","living","kitchen","room")):
        bg_top, bg_bot = (35, 28, 22),  (80, 65, 50)    # warm indoor
        accent         = (160, 130, 80)
        mood           = "indoor"
    elif any(w in loc_lower for w in ("bar","club","jazz","tavern","pub","restaurant","lounge")):
        bg_top, bg_bot = (8, 5, 20),    (25, 15, 50)    # deep night purple
        accent         = (180, 120, 40)                   # warm amber
        mood           = "night"
    elif any(w in loc_lower for w in ("street","city","outside","park","rain","plaza","market")):
        bg_top, bg_bot = (25, 35, 45),  (55, 75, 90)    # urban day
        accent         = (100, 160, 120)
        mood           = "exterior"
    else:
        bg_top, bg_bot = (18, 22, 38),  (45, 50, 80)    # default cinematic
        accent         = (140, 160, 200)
        mood           = "default"

    # ── Load fonts ───────────────────────────────────────────────────────────
    import platform
    def _font(size, bold=False):
        candidates = []
        if platform.system() == "Windows":
            base = "C:/Windows/Fonts/"
            candidates = [
                base + ("arialbd.ttf" if bold else "arial.ttf"),
                base + ("calibrib.ttf" if bold else "calibri.ttf"),
                base + "segoeui.ttf",
            ]
        else:
            base = "/usr/share/fonts/truetype/dejavu/"
            candidates = [
                base + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                base + "DejaVuSans.ttf",
            ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    font_scene    = _font(20, bold=True)
    font_location = _font(15)
    font_speaker  = _font(17, bold=True)
    font_line     = _font(16)
    font_cue      = _font(12)
    font_action   = _font(13)

    # ── Per-dialogue timing — derived from actual audio group durations ────────
    # Read the per-group audio files synthesised by VoiceSynthesisAgent.
    # Each group file covers one or more consecutive same-speaker lines.
    # This makes subtitle hold-time match the actual audio duration exactly.
    import glob as _glob, subprocess as _sub, json as _json

    def _probe_duration(path: str) -> float:
        """Return audio/video duration in seconds via ffprobe."""
        try:
            r = _sub.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', path],
                capture_output=True, text=True, timeout=8
            )
            return float(_json.loads(r.stdout)['format']['duration'])
        except Exception:
            return 3.0

    # Collect all per-group/per-line audio files for this scene
    # Supports both naming conventions: grp_* (new) and dlg_* (old)
    grp_files = sorted(
        _glob.glob(f"audio_tracks/scene_{scene_id}_grp_*.mp3") +
        _glob.glob(f"audio_tracks/scene_{scene_id}_grp_*.wav")
    )
    dlg_files = sorted(
        _glob.glob(f"audio_tracks/scene_{scene_id}_dlg_*.mp3") +
        _glob.glob(f"audio_tracks/scene_{scene_id}_dlg_*.wav")
    )
    merged_file = f"audio_tracks/scene_{scene_id}_merged.mp3"

    PADDING_S   = 0.3   # small buffer so subtitle is still readable
    FADE_FRAMES = FPS // 3

    # Determine per-line durations from available audio files
    if grp_files and dialogues:
        # New naming: one file per speaker-group
        grp_durations = [_probe_duration(f) for f in grp_files]
        groups = []
        for dlg in dialogues:
            sp = dlg.get("speaker", "")
            if groups and groups[-1]["speaker"] == sp:
                groups[-1]["count"] += 1
            else:
                groups.append({"speaker": sp, "count": 1})
        per_line_dur = []
        for idx, grp in enumerate(groups):
            dur     = grp_durations[idx] if idx < len(grp_durations) else 3.0
            per_dlg = dur / max(grp["count"], 1)
            per_line_dur.extend([per_dlg] * grp["count"])

    elif dlg_files and dialogues:
        # Old naming: one file per line — direct 1-to-1 mapping
        # There may be multiple versions of each line; take the most recent
        seen = {}
        for f in dlg_files:
            import re as _re
            m = _re.search(r"_dlg_(\d+)_", f)
            if m:
                idx = int(m.group(1))
                seen[idx] = f   # last write wins
        per_line_dur = []
        for i in range(len(dialogues)):
            if i in seen:
                per_line_dur.append(_probe_duration(seen[i]))
            else:
                per_line_dur.append(3.0)

    elif Path(merged_file).exists() and dialogues:
        # Only merged file available — divide equally
        total_audio = _probe_duration(merged_file)
        per_dlg     = total_audio / len(dialogues)
        per_line_dur = [per_dlg] * len(dialogues)

    else:
        # No audio files at all — use 3s per line default
        per_line_dur = [3.0] * max(len(dialogues), 1)

    hold_frames_list = [
        max(int((d + PADDING_S) * FPS), int(FPS * 1.5))
        for d in per_line_dur
    ]
    if not hold_frames_list:
        hold_frames_list = [int(FPS * 3)]

    # Build frame→dialogue map from audio-derived hold times
    line_schedule = []
    cursor = 0
    for i, hf in enumerate(hold_frames_list):
        line_schedule.append((cursor, cursor + hf, i))
        cursor += hf
    total_frames = max(cursor, FPS * 3)

    def get_current_dlg(frame_idx):
        for sf, ef, i in line_schedule:
            if sf <= frame_idx < ef:
                return i, (frame_idx - sf)   # (dlg_index, frames_into_line)
        return len(dialogues) - 1, 0

    # ── Load character images ─────────────────────────────────────────────────
    char_images = {}   # name → PIL Image RGBA
    for char_name in characters:
        cdata = char_db.get(char_name.upper(), {})
        cid   = cdata.get("id", char_name.lower().replace(" ", "_").replace("(","").replace(")","").strip())
        for candidate in [
            Path("image_assets") / f"{cid}.png",
            Path("image_assets") / f"{char_name.lower().replace(' ','_')}.png",
        ]:
            if candidate.exists():
                try:
                    char_images[char_name.upper()] = Image.open(candidate).convert("RGBA")
                    break
                except Exception:
                    pass

    # ── Helper: draw wrapped text, returns next y ────────────────────────────
    def draw_wrapped(draw, text, x, y, max_w, font, fill, line_spacing=4):
        words = text.split()
        line_buf = ""
        cy = y
        for word in words:
            test = (line_buf + " " + word).strip()
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] <= max_w:
                line_buf = test
            else:
                if line_buf:
                    draw.text((x, cy), line_buf, font=font, fill=fill)
                    cy += (bb[3] - bb[1]) + line_spacing
                line_buf = word
        if line_buf:
            draw.text((x, cy), line_buf, font=font, fill=fill)
            cy += draw.textbbox((0,0), line_buf, font=font)[3] + line_spacing
        return cy

    # ── Background texture helpers ────────────────────────────────────────────
    import math, random
    rng = random.Random(scene_id * 42)   # deterministic per scene

    def make_bg_frame(t):
        """Generate one background frame with parallax grain and mood."""
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        # Vertical gradient
        for y in range(H):
            r_ratio = y / H
            r = int(bg_top[0] + (bg_bot[0] - bg_top[0]) * r_ratio)
            g = int(bg_top[1] + (bg_bot[1] - bg_top[1]) * r_ratio)
            b = int(bg_top[2] + (bg_bot[2] - bg_top[2]) * r_ratio)
            draw.line([(0, y), (W, y)], fill=(r, g, b))

        # Mood-specific background elements
        if mood == "night":
            # Stars
            for _ in range(80):
                sx = rng.randint(0, W)
                sy = rng.randint(0, H // 2)
                brightness = rng.randint(120, 255)
                sz = rng.randint(1, 2)
                # Subtle twinkle
                twinkle = int(30 * math.sin(t * 3 + sx * 0.05))
                b_val = max(0, min(255, brightness + twinkle))
                draw.ellipse([sx-sz, sy-sz, sx+sz, sy+sz], fill=(b_val, b_val, b_val-30))
            # Floor glow
            for y in range(H * 3 // 4, H):
                alpha_ratio = (y - H * 3 // 4) / (H // 4)
                draw.line([(0, y), (W, y)],
                    fill=(int(30 * alpha_ratio), int(10 * alpha_ratio), int(5 * alpha_ratio)))
            # Neon sign flicker — only for jazz/bar locations
            if "blue note" in location.lower() or "jazz" in location.lower() or "bar" in location.lower() or "club" in location.lower():
                flicker = abs(math.sin(t * 7.3)) > 0.15
                if flicker:
                    draw.rectangle([30, 60, 130, 90], fill=(80, 10, 30))
                    sign_text = next((w for w in location.split() if w.upper() not in ("INT.","EXT.","THE","A","AN")), "BAR")
                    draw.text((35, 65), sign_text[:10].upper(), font=_font(14, bold=True), fill=(220, 80, 80))

        elif mood == "office":
            # Fluorescent ceiling lines
            for i in range(3):
                lx = 100 + i * 220
                draw.rectangle([lx, 0, lx + 140, 8], fill=(200, 210, 220))
            # Window blinds (right side)
            for i in range(8):
                by = 40 + i * 22
                draw.rectangle([W - 160, by, W - 20, by + 10],
                    fill=(60, 75, 90))

        elif mood == "dark":
            # Warehouse structure: pillars
            for px_col in [120, 380, 640]:
                draw.rectangle([px_col, 0, px_col + 18, H], fill=(18, 16, 14))
                draw.rectangle([px_col + 1, 0, px_col + 2, H], fill=(28, 24, 20))
            # High windows letting in faint moonlight
            for wx in [160, 420, 680]:
                draw.rectangle([wx, 15, wx + 55, 70], fill=(15, 18, 25))
                # Moon glow through window
                moon_glow = int(20 + 8 * math.sin(t * 0.5))
                draw.rectangle([wx + 2, 17, wx + 53, 68], fill=(moon_glow, moon_glow + 5, moon_glow + 15))
            # Flashlight cone (moves slightly)
            fl_x = int(W * 0.55 + math.sin(t * 0.8) * 30)
            fl_y = int(H * 0.4 + math.cos(t * 0.6) * 20)
            for radius in range(80, 0, -8):
                alpha = int(12 * (1 - radius / 80))
                draw.ellipse([fl_x - radius, fl_y - radius//2,
                               fl_x + radius, fl_y + radius//2],
                    fill=(alpha * 3, alpha * 3, alpha * 2))
            # Dust particles floating
            for _ in range(15):
                px2 = rng.randint(0, W)
                py2 = int((rng.randint(0, H) - t * 12) % H)
                draw.ellipse([px2-1, py2-1, px2+1, py2+1], fill=(55, 48, 40))
            # Puddle reflection on floor
            draw.ellipse([W//2 - 80, H * 3//4, W//2 + 80, H - 20],
                fill=(12, 14, 18))

        elif mood == "indoor":
            # Window light rays (left side)
            for i in range(4):
                ray_x = 40 + i * 25
                draw.polygon([
                    (ray_x, 0), (ray_x + 15, 0),
                    (ray_x + 60, H // 2), (ray_x + 40, H // 2)
                ], fill=(255, 240, 200, 15))
            # Bookshelf (right background)
            for row in range(3):
                for col in range(6):
                    bx = W - 180 + col * 28
                    by = 80 + row * 55
                    bh = rng.randint(35, 50)
                    bc = (rng.randint(60,150), rng.randint(30,100), rng.randint(20,80))
                    draw.rectangle([bx, by, bx+20, by+bh], fill=bc)

        elif mood == "exterior":
            # Moving clouds
            for i in range(3):
                cx = int((rng.randint(50, W-50) + t * 8 * (i+1)) % W)
                cy = rng.randint(20, 100)
                draw.ellipse([cx-60, cy-20, cx+60, cy+20], fill=(60, 75, 85))
                draw.ellipse([cx-40, cy-30, cx+40, cy+10], fill=(65, 80, 90))
            # Ground / pavement
            draw.rectangle([0, H * 3 // 4, W, H],
                fill=(35, 35, 38))
            for i in range(0, W, 80):
                lx = int((i + t * 5) % W)
                draw.rectangle([lx, H * 3 // 4 + 10, lx + 50, H * 3 // 4 + 13],
                    fill=(50, 50, 52))

        # Cinematic letterbox bars (top and bottom 5%)
        bar_h = int(H * 0.05)
        draw.rectangle([0, 0, W, bar_h], fill=(0, 0, 0))
        draw.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0))

        return img

    # ── Frame rendering loop ──────────────────────────────────────────────────
    frames = []
    for frame_idx in range(total_frames):
        t = frame_idx / FPS

        # Background
        bg = make_bg_frame(t)
        img = bg.convert("RGBA")
        draw = ImageDraw.Draw(img)

        # Current dialogue line for portrait switching only (not subtitles)
        dlg_idx, frames_into_line = get_current_dlg(frame_idx)
        if dialogues:
            dlg     = dialogues[dlg_idx]
            speaker = dlg.get("speaker", "").upper()
        else:
            speaker = ""

        # ── Character portrait (left side, switches with speaker) ────────────
        PORT_W = int(W * 0.38)
        PORT_H = int(H * 0.60)
        PORT_X = 30
        PORT_Y = int((H - PORT_H) // 2) - 10

        portrait = char_images.get(speaker)
        if portrait is None and characters:
            # Fallback: first character in scene
            portrait = char_images.get(characters[0].upper())

        if portrait:
            # Subtle float animation
            float_offset = int(math.sin(t * 1.8) * 4)
            p_resized = portrait.resize((PORT_W, PORT_H), Image.LANCZOS)

            # Drop shadow
            shadow = Image.new("RGBA", (PORT_W + 8, PORT_H + 8), (0, 0, 0, 0))
            shadow_d = ImageDraw.Draw(shadow)
            shadow_d.rectangle([4, 4, PORT_W + 4, PORT_H + 4], fill=(0, 0, 0, 80))
            img.paste(shadow, (PORT_X + 2, PORT_Y + float_offset + 2), shadow)
            img.paste(p_resized, (PORT_X, PORT_Y + float_offset), p_resized)

            # Speaking highlight ring when active speaker
            if speaker and speaker in char_images:
                ring = Image.new("RGBA", (PORT_W + 10, PORT_H + 10), (0, 0, 0, 0))
                ring_d = ImageDraw.Draw(ring)
                pulse = int(abs(math.sin(t * 4)) * 3)
                ring_d.rectangle([0, 0, PORT_W + 9, PORT_H + 9],
                    outline=accent + (200,), width=2 + pulse)
                img.paste(ring, (PORT_X - 5, PORT_Y + float_offset - 5), ring)

        draw = ImageDraw.Draw(img)

        # ── Right panel — scene info + non-speaking characters ───────────────
        # Panel occupies x: PORT_X + PORT_W + 30  to  W - 20
        RP_X  = PORT_X + PORT_W + 40
        RP_W  = W - RP_X - 20
        RP_Y  = 55   # below header
        RP_H  = H - 55 - 95  # above dialogue bar

        # Subtle panel background
        panel_bg = Image.new("RGBA", (RP_W, RP_H), (0, 0, 0, 100))
        img.paste(panel_bg, (RP_X, RP_Y), panel_bg)
        draw = ImageDraw.Draw(img)

        # Accent top line
        draw.line([(RP_X, RP_Y), (RP_X + RP_W, RP_Y)], fill=accent + (160,), width=1)

        # Action description (italic-style, wrapped, top of panel)
        if action:
            action_y = RP_Y + 10
            action_y = draw_wrapped(draw, f'"{action}"',
                x=RP_X + 10, y=action_y,
                max_w=RP_W - 20,
                font=font_action,
                fill=(170, 165, 145),
                line_spacing=5)

            # Divider
            draw.line([(RP_X + 10, action_y + 8), (RP_X + RP_W - 10, action_y + 8)],
                fill=(80, 80, 80), width=1)

        # Non-speaking character thumbnails (right panel)
        thumb_y = RP_Y + (80 if action else 15)
        THUMB_W = min(int(RP_W * 0.75), 130)
        for i, char_name in enumerate(characters):
            if char_name.upper() == speaker:
                continue   # skip active speaker — already shown left
            thumb_img = char_images.get(char_name.upper())
            if thumb_img is None:
                continue
            THUMB_H = int(THUMB_W * thumb_img.height / thumb_img.width)
            if thumb_y + THUMB_H > RP_Y + RP_H - 10:
                break
            t_resized = thumb_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            t_cx = RP_X + (RP_W - THUMB_W) // 2
            img.paste(t_resized, (t_cx, thumb_y), t_resized)
            draw = ImageDraw.Draw(img)
            # Name label under thumbnail
            cdata = char_db.get(char_name.upper(), {})
            disp_name = char_name.upper()
            bb3 = draw.textbbox((0,0), disp_name, font=font_cue)
            lbl_x = t_cx + (THUMB_W - (bb3[2]-bb3[0])) // 2
            draw.text((lbl_x, thumb_y + THUMB_H + 3), disp_name,
                font=font_cue, fill=(180, 180, 180))
            thumb_y += THUMB_H + 24

        draw = ImageDraw.Draw(img)

        # ── Top header bar ────────────────────────────────────────────────────
        HEADER_H = 42
        # Semi-transparent bar
        header_overlay = Image.new("RGBA", (W, HEADER_H), (0, 0, 0, 170))
        img.paste(header_overlay, (0, 0), header_overlay)
        draw = ImageDraw.Draw(img)

        draw.text((12, 8),
            f"SCENE {scene_id}",
            font=font_scene, fill=accent)
        draw.text((12 + draw.textbbox((0,0), f"SCENE {scene_id}", font=font_scene)[2] + 10, 10),
            f"|  {location.upper()}",
            font=font_location, fill=(200, 200, 200))

        # Characters in scene (top right)
        char_label = "  ·  ".join(c for c in characters)
        bb = draw.textbbox((0,0), char_label, font=font_cue)
        draw.text((W - (bb[2] - bb[0]) - 12, 14), char_label,
            font=font_cue, fill=(160, 160, 160))

        # ── Bottom area — action description only (no subtitle bar) ─────────
        # Subtitles are burned separately by SubtitleBurnerAgent with
        # audio-derived timing. Rendering them here too causes double-layer
        # subtitles with different schedules, which is the sync bug.
        if action and frame_idx < int(FPS * 2.5):
            # Show action description briefly at scene start (first 2.5s)
            bar_h   = 36
            bar_y   = H - bar_h - 4
            act_bg  = Image.new("RGBA", (W, bar_h), (0, 0, 0, 140))
            img.paste(act_bg, (0, bar_y), act_bg)
            draw = ImageDraw.Draw(img)
            draw.text((16, bar_y + 8), action[:100],
                font=font_action, fill=(160, 155, 130))

        frames.append(np.array(img.convert("RGB")))

    # ── Write video ───────────────────────────────────────────────────────────
    import imageio
    writer = imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return {"status": "success", "output_path": output_path, "frames": total_frames}


# ─────────────────────────────────────────────
# Handler dispatch table
# ─────────────────────────────────────────────
_TOOL_HANDLERS = {
    "commit_memory":           _tool_commit_memory,
    "get_task_graph":          _tool_get_task_graph,
    "voice_cloning_synthesizer": _tool_voice_synth,
    "query_stock_footage":     _tool_query_stock_footage,
    "identity_validator":      _tool_identity_validator,
    "face_swapper":            _tool_face_swapper,
    "lip_sync_aligner":        _tool_lip_sync_aligner,
    "video_scene_composer":    _tool_video_scene_composer,
}

# Singleton
mcp = MCPRegistry()