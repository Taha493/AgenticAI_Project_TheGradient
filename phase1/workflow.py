"""
Phase 1 LangGraph Workflow
===========================
Nodes: mode_selector → validator/scriptwriter → hitl → character → image → memory_commit
Outputs: scene_manifest.json, character_db.json, image_assets/
"""

import json
import sys
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase1.state import Phase1State
from phase1.agents import (
    ScriptwriterAgent,
    ScriptValidatorAgent,
    HITLAgent,
    CharacterDesignerAgent,
    ImageSynthesizerAgent,
)
from mcp_server.registry import mcp

# Output paths
OUTPUTS_DIR    = Path("outputs")
MANIFEST_PATH  = OUTPUTS_DIR / "scene_manifest.json"
CHAR_DB_PATH   = OUTPUTS_DIR / "character_db.json"
IMAGE_ASSETS   = "image_assets"


# ──────────────────────────────────────────────
# NODE: Mode Selector
# ──────────────────────────────────────────────
def mode_selector_node(state: Phase1State) -> Phase1State:
    """Determines whether to route to validator (manual) or scriptwriter (auto)."""
    print("\n[Node: mode_selector] Detecting input mode...")
    mode = state.get("input_mode", "auto").lower()
    print(f"  Mode: {mode}")
    return {**state, "input_mode": mode, "status": "processing"}


def route_after_mode(state: Phase1State) -> Literal["validator_node", "scriptwriter_node"]:
    if state["input_mode"] == "manual":
        return "validator_node"
    return "scriptwriter_node"


# ──────────────────────────────────────────────
# NODE: Script Validator (manual mode)
# ──────────────────────────────────────────────
def validator_node(state: Phase1State) -> Phase1State:
    print("\n[Node: validator] Validating manual script...")
    agent = ScriptValidatorAgent()
    result = agent.run(state["raw_input"])
    if not result["valid"]:
        print(f"  ✗ Validation failed: {result['errors']}")
        return {**state, "validation_result": result, "status": "failed",
                "error": "; ".join(result["errors"])}
    print(f"  ✓ Script valid with {len(result['script'].get('scenes', []))} scenes.")
    return {**state, "validation_result": result, "script": result["script"]}


# ──────────────────────────────────────────────
# NODE: Scriptwriter (auto mode)
# ──────────────────────────────────────────────
def scriptwriter_node(state: Phase1State) -> Phase1State:
    print("\n[Node: scriptwriter] Generating script from prompt...")
    agent = ScriptwriterAgent()
    script = agent.run(state["raw_input"], num_scenes=3)
    return {**state, "script": script}


def route_after_script(state: Phase1State) -> Literal["hitl_node", END]:
    if state.get("status") == "failed":
        return END
    return "hitl_node"


# ──────────────────────────────────────────────
# NODE: HITL
# ──────────────────────────────────────────────
def hitl_node(state: Phase1State) -> Phase1State:
    print("\n[Node: hitl] Human-in-the-Loop checkpoint...")
    agent = HITLAgent()
    # auto_approve=True for non-interactive runs (set False for live demo)
    approved = agent.run(state["script"], auto_approve=True)
    if not approved:
        return {**state, "hitl_approved": False, "status": "failed",
                "error": "Script rejected by user at HITL checkpoint."}
    return {**state, "hitl_approved": True}


def route_after_hitl(state: Phase1State) -> Literal["character_node", END]:
    if not state.get("hitl_approved", False):
        return END
    return "character_node"


# ──────────────────────────────────────────────
# NODE: Character Designer
# ──────────────────────────────────────────────
def character_node(state: Phase1State) -> Phase1State:
    print("\n[Node: character] Designing character identities...")
    agent = CharacterDesignerAgent()
    characters = agent.run(state["script"])
    return {**state, "characters": characters}


# ──────────────────────────────────────────────
# NODE: Image Synthesizer
# ──────────────────────────────────────────────
def image_node(state: Phase1State) -> Phase1State:
    print("\n[Node: image] Generating character portraits...")
    agent = ImageSynthesizerAgent()
    image_paths = agent.run(state["characters"], output_dir=IMAGE_ASSETS)
    return {**state, "images": image_paths}


# ──────────────────────────────────────────────
# NODE: Memory Commit + Output Writer
# ──────────────────────────────────────────────
def memory_commit_node(state: Phase1State) -> Phase1State:
    print("\n[Node: memory_commit] Writing final outputs...")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build scene_manifest.json
    script = state["script"]
    manifest = {
        "input_mode": state["input_mode"],
        "script": script,
        "characters": [c["id"] for c in state["characters"]],
        "images": state.get("images", []),
        "status": "complete",
        "scenes": script.get("scenes", []),
        "title": script.get("title", "Untitled"),
        "genre": script.get("genre", "drama"),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ scene_manifest.json → {MANIFEST_PATH}")

    # Build character_db.json
    char_db = {"characters": state["characters"]}
    with open(CHAR_DB_PATH, "w") as f:
        json.dump(char_db, f, indent=2)
    print(f"  ✓ character_db.json   → {CHAR_DB_PATH}")

    # Commit both to memory via MCP
    mcp.invoke("commit_memory", {
        "collection": "phase1_outputs",
        "doc_id": "manifest",
        "data": {"manifest_path": str(MANIFEST_PATH), "char_db_path": str(CHAR_DB_PATH)},
    })

    return {
        **state,
        "scene_manifest_path": str(MANIFEST_PATH),
        "character_db_path": str(CHAR_DB_PATH),
        "status": "complete",
    }


# ──────────────────────────────────────────────
# Build LangGraph StateGraph
# ──────────────────────────────────────────────
def build_phase1_graph() -> StateGraph:
    g = StateGraph(Phase1State)

    g.add_node("mode_selector_node",  mode_selector_node)
    g.add_node("validator_node",      validator_node)
    g.add_node("scriptwriter_node",   scriptwriter_node)
    g.add_node("hitl_node",           hitl_node)
    g.add_node("character_node",      character_node)
    g.add_node("image_node",          image_node)
    g.add_node("memory_commit_node",  memory_commit_node)

    g.set_entry_point("mode_selector_node")

    g.add_conditional_edges("mode_selector_node", route_after_mode, {
        "validator_node":   "validator_node",
        "scriptwriter_node": "scriptwriter_node",
    })
    g.add_conditional_edges("validator_node", route_after_script, {
        "hitl_node": "hitl_node",
        END: END,
    })
    g.add_conditional_edges("scriptwriter_node", route_after_script, {
        "hitl_node": "hitl_node",
        END: END,
    })
    g.add_conditional_edges("hitl_node", route_after_hitl, {
        "character_node": "character_node",
        END: END,
    })
    g.add_edge("character_node",     "image_node")
    g.add_edge("image_node",         "memory_commit_node")
    g.add_edge("memory_commit_node", END)

    return g.compile()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def run_phase1(prompt: str = None, manual_script: str = None) -> Phase1State:
    """
    Run Phase 1 pipeline.
    Args:
        prompt:        For auto mode — story prompt
        manual_script: For manual mode — raw script text or JSON
    """
    graph = build_phase1_graph()

    if manual_script:
        initial_state = Phase1State(
            input_mode="manual",
            raw_input=manual_script,
            validation_result=None,
            script=None,
            hitl_approved=None,
            characters=None,
            images=None,
            scene_manifest_path=None,
            character_db_path=None,
            status="processing",
            error=None,
        )
    else:
        initial_state = Phase1State(
            input_mode="auto",
            raw_input=prompt or "A detective investigates a mysterious disappearance in a rain-soaked city.",
            validation_result=None,
            script=None,
            hitl_approved=None,
            characters=None,
            images=None,
            scene_manifest_path=None,
            character_db_path=None,
            status="processing",
            error=None,
        )

    print("═" * 60)
    print("  PROJECT MONTAGE — PHASE 1: THE WRITER'S ROOM")
    print("═" * 60)
    final_state = graph.invoke(initial_state)
    print("\n" + "═" * 60)
    print(f"  Phase 1 Status: {final_state['status'].upper()}")
    if final_state.get("error"):
        print(f"  Error: {final_state['error']}")
    else:
        print(f"  scene_manifest.json → {final_state['scene_manifest_path']}")
        print(f"  character_db.json   → {final_state['character_db_path']}")
        print(f"  Images generated    → {len(final_state.get('images', []))}")
    print("═" * 60)
    return final_state


if __name__ == "__main__":
    run_phase1(prompt="A space explorer discovers an ancient alien civilization on a distant planet.")
