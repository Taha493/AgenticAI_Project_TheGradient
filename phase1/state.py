"""
Phase 1 LangGraph State Schema
Shared state object passed between all Phase 1 nodes.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class Phase1State(TypedDict):
    # Input
    input_mode: str                       # "manual" | "auto"
    raw_input: str                        # prompt (auto) or raw script text (manual)

    # Intermediate
    validation_result: Optional[Dict]    # from validator node
    script: Optional[Dict]               # structured scene JSON
    hitl_approved: Optional[bool]        # human approval flag
    characters: Optional[List[Dict]]     # extracted character list
    images: Optional[List[str]]          # generated image paths

    # Outputs
    scene_manifest_path: Optional[str]
    character_db_path: Optional[str]
    status: str                           # "processing" | "complete" | "failed"
    error: Optional[str]
