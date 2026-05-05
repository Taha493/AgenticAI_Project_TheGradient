"""
Phase 5 State Schema
Edit Agent — intent classification + targeted re-execution + undo
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class EditIntent(TypedDict):
    intent:     str          # e.g. "change_voice_tone", "make_scene_darker"
    target:     str          # "audio" | "video_frame" | "video" | "script"
    scope:      str          # e.g. "character:HERO", "scene:2", "all"
    parameters: Dict         # e.g. {"tone": "whispered"}, {"filter": "darken"}
    confidence: float        # 0.0 – 1.0


class EditState(TypedDict):
    # Input
    raw_query:        str                    # free-text user edit command
    conversation_history: List[Dict]        # multi-turn edit session

    # Classified intent
    intent:           Optional[EditIntent]

    # Execution
    execution_result: Optional[Dict]        # what was changed + new asset paths
    error:            Optional[str]

    # Versioning
    version_before:   Optional[int]
    version_after:    Optional[int]

    # Status
    status:           str                   # classifying|executing|complete|failed