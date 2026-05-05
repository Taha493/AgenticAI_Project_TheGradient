"""
Phase 3 LangGraph Workflow — Video Generation & Composition
============================================================
Nodes:
  clip_validator → bgm_selector → subtitle_burner → compositor → manifest_writer → END

Input:  raw_scenes/*.mp4  +  audio_tracks/*.mp3  +  scene_manifest.json
Output: outputs/final_output.mp4  +  outputs/timing_manifest.json
"""

import json
import sys
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3.state import Phase3State
from phase3.agents import (
    ClipValidatorAgent,
    BGMSelectorAgent,
    SubtitleBurnerAgent,
    TransitionCompositorAgent,
    TimingManifestAgent,
)
from mcp_server.registry import mcp

FINAL_VIDEO_PATH   = "outputs/final_output.mp4"
TIMING_MANIFEST    = "outputs/timing_manifest.json"
PHASE2_LOG         = "task_logs/phase2_log.json"


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def clip_validator_node(state: Phase3State) -> Phase3State:
    print("\n[Node: clip_validator] Validating Phase 2 clips...")
    agent = ClipValidatorAgent()
    try:
        clips = agent.run(state["phase2_log_path"], state["scene_manifest_path"])
        if not clips:
            return {**state, "status": "failed",
                    "error": "No valid scene clips found from Phase 2."}
        return {**state, "scene_clips": clips}
    except Exception as e:
        return {**state, "status": "failed", "error": str(e)}


def route_after_validate(state: Phase3State) -> Literal["bgm_selector_node", "__end__"]:
    return "__end__" if state["status"] == "failed" else "bgm_selector_node"


def bgm_selector_node(state: Phase3State) -> Phase3State:
    print("\n[Node: bgm_selector] Generating background music...")
    agent = BGMSelectorAgent()
    try:
        bgm_paths = agent.run(state["scene_clips"])
        return {**state, "bgm_paths": bgm_paths}
    except Exception as e:
        print(f"  [BGM] Warning: {e} — continuing without BGM")
        return {**state, "bgm_paths": []}


def subtitle_burner_node(state: Phase3State) -> Phase3State:
    print("\n[Node: subtitle_burner] Burning subtitles...")
    agent = SubtitleBurnerAgent()
    try:
        subtitled = agent.run(state["scene_clips"])
        return {**state, "subtitled_clips": subtitled}
    except Exception as e:
        print(f"  [Subtitles] Warning: {e} — using raw clips")
        raw_paths = [c["video_path"] for c in state["scene_clips"]]
        return {**state, "subtitled_clips": raw_paths}


def compositor_node(state: Phase3State) -> Phase3State:
    print("\n[Node: compositor] Stitching scenes with transitions + BGM...")
    agent = TransitionCompositorAgent()
    try:
        final_path, timing_entries = agent.run(
            subtitled_clips=state["subtitled_clips"],
            bgm_paths=state.get("bgm_paths") or [],
            clips_meta=state["scene_clips"],
            output_path=FINAL_VIDEO_PATH,
        )
        return {**state, "final_video_path": final_path, "timed_clips": timing_entries}
    except Exception as e:
        return {**state, "status": "failed", "error": f"Compositor failed: {e}"}


def route_after_compositor(state: Phase3State) -> Literal["manifest_writer_node", "__end__"]:
    return "__end__" if state["status"] == "failed" else "manifest_writer_node"


def manifest_writer_node(state: Phase3State) -> Phase3State:
    print("\n[Node: manifest_writer] Writing timing manifest...")
    agent = TimingManifestAgent()
    try:
        manifest_path = agent.run(
            timing_entries=state.get("timed_clips") or [],
            output_path=TIMING_MANIFEST,
        )
        return {**state, "timing_manifest_path": manifest_path, "status": "complete"}
    except Exception as e:
        return {**state, "status": "failed", "error": f"Manifest writer failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_phase3_graph() -> StateGraph:
    g = StateGraph(Phase3State)

    g.add_node("clip_validator_node",   clip_validator_node)
    g.add_node("bgm_selector_node",     bgm_selector_node)
    g.add_node("subtitle_burner_node",  subtitle_burner_node)
    g.add_node("compositor_node",       compositor_node)
    g.add_node("manifest_writer_node",  manifest_writer_node)

    g.set_entry_point("clip_validator_node")

    g.add_conditional_edges("clip_validator_node", route_after_validate, {
        "bgm_selector_node": "bgm_selector_node",
        "__end__": END,
    })
    g.add_edge("bgm_selector_node",    "subtitle_burner_node")
    g.add_edge("subtitle_burner_node", "compositor_node")
    g.add_conditional_edges("compositor_node", route_after_compositor, {
        "manifest_writer_node": "manifest_writer_node",
        "__end__": END,
    })
    g.add_edge("manifest_writer_node", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3(
    scene_manifest_path: str = "outputs/scene_manifest.json",
    character_db_path:   str = "outputs/character_db.json",
    phase2_log_path:     str = "task_logs/phase2_log.json",
    version:             int = 1,
) -> Phase3State:
    graph = build_phase3_graph()

    initial = Phase3State(
        scene_manifest_path=scene_manifest_path,
        character_db_path=character_db_path,
        phase2_log_path=phase2_log_path,
        scene_clips=None,
        bgm_paths=None,
        timed_clips=None,
        subtitled_clips=None,
        timing_manifest_path=None,
        final_video_path=None,
        version=version,
        status="processing",
        error=None,
    )

    print("═" * 60)
    print("  PROJECT MONTAGE — PHASE 3: VIDEO COMPOSITION")
    print("═" * 60)

    final = graph.invoke(initial)

    print("\n" + "═" * 60)
    print(f"  Phase 3 Status: {final['status'].upper()}")
    if final.get("error"):
        print(f"  Error: {final['error']}")
    else:
        print(f"  Final video      → {final['final_video_path']}")
        print(f"  Timing manifest  → {final['timing_manifest_path']}")
    print("═" * 60)

    return final


if __name__ == "__main__":
    run_phase3()