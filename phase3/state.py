"""
Phase 3 LangGraph State Schema
Video Composition — takes raw per-scene .mp4s from Phase 2 and produces
a single composited final_output.mp4 with transitions, BGM, and subtitles.
"""
from typing import Dict, List, Optional
from typing_extensions import TypedDict


class Phase3State(TypedDict):
    # ── Inputs from Phase 1 + Phase 2 ──────────────────────────────────────
    scene_manifest_path: str          # outputs/scene_manifest.json
    character_db_path: str            # outputs/character_db.json
    phase2_log_path: str              # task_logs/phase2_log.json

    # ── Intermediate ────────────────────────────────────────────────────────
    scene_clips: Optional[List[Dict]]     # validated clip metadata per scene
    bgm_paths: Optional[List[str]]        # background music files per scene
    timed_clips: Optional[List[Dict]]     # clips with timing info
    subtitled_clips: Optional[List[str]]  # per-scene video paths with subtitles

    # ── Outputs ─────────────────────────────────────────────────────────────
    timing_manifest_path: Optional[str]   # outputs/timing_manifest.json
    final_video_path: Optional[str]       # outputs/final_output.mp4
    version: Optional[int]               # snapshot version number

    # ── Status ──────────────────────────────────────────────────────────────
    status: str                           # processing | complete | failed
    error: Optional[str]