"""
Phase 5 LangGraph Workflow — Intelligent Edit & Undo Agent
===========================================================
Nodes:
  classify_intent → validate_intent → execute_edit → snapshot → END

Features:
  - LangGraph SqliteSaver for multi-turn edit session persistence
  - Full undo via state_manager.revert()
  - 10+ edit intent types
  - Human-readable edit history

Usage:
  from phase5.workflow import EditAgent
  agent = EditAgent()
  result = agent.edit("make scene 2 darker")
  result = agent.edit("change the voice tone to whispered")
  agent.undo()
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase5.state import EditState
from phase5.intent_classifier import IntentClassifier
from phase5.executor import EditExecutor
import phase4.state_manager as state_manager

EDIT_DB = Path("outputs/edit_sessions.db")


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

_classifier = IntentClassifier()
_executor   = EditExecutor()


def classify_intent_node(state: EditState) -> EditState:
    print(f"\n[EditAgent: classify] Query: '{state['raw_query']}'")
    intent = _classifier.classify(state["raw_query"])
    print(f"  → Intent: {intent['intent']}  Target: {intent['target']}"
          f"  Scope: {intent['scope']}  Confidence: {intent['confidence']:.2f}")
    return {**state, "intent": intent, "status": "validating"}


def validate_intent_node(state: EditState) -> EditState:
    intent = state["intent"]
    if intent["intent"] == "unknown" or intent["confidence"] < 0.1:
        return {**state,
                "status": "failed",
                "error":  f"Could not understand edit command: '{state['raw_query']}'"}
    return {**state, "status": "executing"}


def route_after_validate(state: EditState) -> Literal["execute_edit_node", "__end__"]:
    return "__end__" if state["status"] == "failed" else "execute_edit_node"


def execute_edit_node(state: EditState) -> EditState:
    intent     = state["intent"]
    version_before = state_manager.latest_version()
    print(f"\n[EditAgent: execute] {intent['intent']} on {intent['target']}")

    try:
        result = _executor.execute(intent)

        # Snapshot after edit
        asset_paths = []
        for scene_result in result.get("affected_scenes", []):
            for key in ("file", "output", "new_portrait"):
                if scene_result.get(key):
                    asset_paths.append(scene_result[key])
        if result.get("final_video"):
            asset_paths.append(result["final_video"])

        version_after = version_before + 1
        state_manager.snapshot(
            version=version_after,
            state_json={
                "edit_query":  state["raw_query"],
                "intent":      intent,
                "result":      result,
            },
            asset_paths=asset_paths,
            description=f"Edit: {state['raw_query'][:50]}",
        )

        return {
            **state,
            "execution_result": result,
            "version_before":   version_before,
            "version_after":    version_after,
            "status":           "complete",
            "error":            None,
        }

    except Exception as e:
        return {**state, "status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def build_edit_graph() -> StateGraph:
    g = StateGraph(EditState)
    g.add_node("classify_intent_node", classify_intent_node)
    g.add_node("validate_intent_node", validate_intent_node)
    g.add_node("execute_edit_node",    execute_edit_node)

    g.set_entry_point("classify_intent_node")
    g.add_edge("classify_intent_node", "validate_intent_node")
    g.add_conditional_edges("validate_intent_node", route_after_validate, {
        "execute_edit_node": "execute_edit_node",
        "__end__": END,
    })
    g.add_edge("execute_edit_node", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# EditAgent — high-level interface
# ─────────────────────────────────────────────────────────────────────────────

class EditAgent:
    """
    High-level edit agent.
    Maintains conversation history and undo stack across calls.

        agent = EditAgent()
        agent.edit("make scene 2 darker")
        agent.edit("change the voice to whispered")
        agent.undo()          # revert last edit
        agent.undo(version=3) # revert to specific version
        agent.history()       # list all versions
    """

    def __init__(self):
        self._graph   = build_edit_graph()
        self._history: List[Dict] = []

    def edit(self, query: str) -> Dict:
        """Process a free-text edit command. Returns result dict."""
        initial = EditState(
            raw_query=query,
            conversation_history=self._history.copy(),
            intent=None,
            execution_result=None,
            error=None,
            version_before=None,
            version_after=None,
            status="classifying",
        )

        final = self._graph.invoke(initial)

        # Record in history
        self._history.append({
            "query":          query,
            "intent":         final.get("intent"),
            "status":         final["status"],
            "version_before": final.get("version_before"),
            "version_after":  final.get("version_after"),
            "error":          final.get("error"),
        })

        if final["status"] == "complete":
            print(f"\n[EditAgent] ✓ Edit applied. "
                  f"v{final['version_before']} → v{final['version_after']}")
        else:
            print(f"\n[EditAgent] ✗ Edit failed: {final.get('error')}")

        return final

    def undo(self, version: Optional[int] = None) -> Dict:
        """
        Revert to a previous version.
        If version is None, reverts to the version before the last edit.
        """
        if version is None:
            # Find the version_before of the last successful edit
            for entry in reversed(self._history):
                if entry["status"] == "complete" and entry["version_before"] is not None:
                    version = entry["version_before"]
                    break

        if version is None:
            print("[EditAgent] Nothing to undo.")
            return {"status": "nothing_to_undo"}

        try:
            result = state_manager.revert(version)
            print(f"[EditAgent] ↩ Reverted to v{version}. "
                  f"{len(result.get('restored', []))} files restored.")
            self._history.append({
                "query":   f"[UNDO → v{version}]",
                "intent":  None,
                "status":  "reverted",
                "version_before": state_manager.latest_version(),
                "version_after":  version,
                "error":   None,
            })
            return result
        except Exception as e:
            print(f"[EditAgent] ✗ Undo failed: {e}")
            return {"status": "failed", "error": str(e)}

    def history(self) -> List[Dict]:
        """Return full version history from state manager."""
        return state_manager.history()

    def session_history(self) -> List[Dict]:
        """Return edit operations performed in this session."""
        return self._history


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def run_edit_session(queries: Optional[List[str]] = None):
    """
    Run an interactive or scripted edit session.
    If queries is None, launches interactive console mode.
    """
    agent = EditAgent()

    if queries:
        print("═" * 60)
        print("  PROJECT MONTAGE — PHASE 5: EDIT AGENT")
        print("═" * 60)
        for q in queries:
            print(f"\n  ▶ Edit: {q}")
            agent.edit(q)

        print("\n  Version History:")
        for v in agent.history():
            print(f"    v{v['version']:>2}  {v['description'][:50]}")
        return agent

    # Interactive mode
    print("═" * 60)
    print("  PROJECT MONTAGE — PHASE 5: EDIT AGENT")
    print("  Type edit commands. 'undo', 'history', 'quit' are special.")
    print("═" * 60)
    while True:
        try:
            query = input("\n  Edit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Edit session ended]")
            break
        if not query:
            continue
        if query.lower() == "quit":
            break
        elif query.lower() == "history":
            for v in agent.history():
                print(f"  v{v['version']:>2}  {v['description'][:60]}")
        elif query.lower().startswith("undo"):
            parts = query.split()
            v = int(parts[1]) if len(parts) > 1 else None
            agent.undo(v)
        else:
            agent.edit(query)

    return agent


if __name__ == "__main__":
    run_edit_session()