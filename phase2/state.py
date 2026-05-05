"""
Phase 2 LangGraph State Schema
Shared state object passed between all Phase 2 nodes.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class Phase2State(TypedDict):
    # Inputs from Phase 1
    scene_manifest_path: str
    character_db_path: str

    # Intermediate
    task_graph: Optional[List[Dict]]      # from scene_parser
    audio_results: Optional[List[Dict]]   # from voice_synth per scene
    video_results: Optional[List[Dict]]   # from video_gen per scene
    face_swap_results: Optional[List[Dict]]
    lip_sync_results: Optional[List[Dict]]

    # Final outputs
    output_videos: Optional[List[str]]    # paths to raw_scenes/*.mp4
    task_logs: Optional[List[Dict]]

    # Status
    status: str
    error: Optional[str]
