#!/usr/bin/env python3
"""
PROJECT MONTAGE — Full Pipeline Runner
=======================================
Usage:
    python run.py --prompt "A detective investigates a mysterious city crime"
    python run.py --prompt "..." --scenes 4
    python run.py --phase3-only          # use existing Phase 1+2 outputs
    python run.py --serve                # start web UI only
    python run.py --prompt "..." --serve # run pipeline then open web UI
"""

import argparse
import os
import sys
from pathlib import Path

import nest_asyncio
# Only apply nest_asyncio patch for pipeline runs (not for --serve only).
# It gets applied inside run_pipeline() when needed.
_nest_applied = False

def _ensure_nest():
    global _nest_applied
    if not _nest_applied:
        nest_asyncio.apply()
        _nest_applied = True

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from phase1.workflow import run_phase1
from phase2.workflow import run_phase2
from phase3.workflow import run_phase3
import phase4.state_manager as state_manager


def run_pipeline(prompt=None, manual_script=None, scenes=3,
                 phase2_only=False, phase3_only=False):

    _ensure_nest()
    manifest_path = "outputs/scene_manifest.json"
    char_db_path  = "outputs/character_db.json"

    if not phase2_only and not phase3_only:
        if manual_script:
            with open(manual_script) as f:
                raw = f.read()
            p1 = run_phase1(manual_script=raw)
        else:
            p1 = run_phase1(prompt=prompt or
                "A brilliant scientist discovers a portal to a parallel world.")
        if p1["status"] != "complete":
            print(f"\nPhase 1 failed: {p1.get('error')}")
            sys.exit(1)
        manifest_path = p1["scene_manifest_path"]
        char_db_path  = p1["character_db_path"]

    if not phase3_only:
        print("\n")
        p2 = run_phase2(scene_manifest_path=manifest_path,
                        character_db_path=char_db_path)
        if p2["status"] != "complete":
            print(f"\nPhase 2 failed: {p2.get('error')}")
            sys.exit(1)

    print("\n")
    version = state_manager.latest_version() + 1
    p3 = run_phase3(scene_manifest_path=manifest_path,
                    character_db_path=char_db_path,
                    phase2_log_path="task_logs/phase2_log.json",
                    version=version)
    if p3["status"] != "complete":
        print(f"\nPhase 3 failed: {p3.get('error')}")
        sys.exit(1)

    state_manager.snapshot(
        version=version,
        state_json={"prompt": prompt or "", "manifest": manifest_path,
                    "final_video": p3["final_video_path"]},
        asset_paths=[p3["final_video_path"] or "",
                     p3["timing_manifest_path"] or "",
                     manifest_path, char_db_path],
        description=f"CLI: {(prompt or 'manual')[:50]}",
    )

    print("\n\n✅  PROJECT MONTAGE COMPLETE")
    print("─" * 55)
    print(f"  scene_manifest.json  → {manifest_path}")
    print(f"  character_db.json    → {char_db_path}")
    print(f"  Character images     → image_assets/")
    print(f"  Audio tracks         → audio_tracks/")
    print(f"  Scene videos         → raw_scenes/")
    print(f"  Final output         → {p3['final_video_path']}")
    print(f"  Timing manifest      → {p3['timing_manifest_path']}")
    print(f"  Version snapshot     → outputs/versions/v{version}/")
    print("─" * 55)
    return p3


def serve_web():
    try:
        import uvicorn
    except ImportError:
        print("Error: pip install uvicorn")
        sys.exit(1)

    # nest_asyncio patches asyncio.run() which breaks uvicorn on Windows.
    # We must unpatch it before handing control to uvicorn.
    try:
        import asyncio, nest_asyncio
        # Remove the nest_asyncio patch by restoring the original asyncio.run
        if hasattr(asyncio, "_original_run"):
            asyncio.run = asyncio._original_run
    except Exception:
        pass

    print("\n" + "═"*55)
    print("  PROJECT MONTAGE — PHASE 4: WEB INTERFACE")
    print("═"*55)
    print("  Open browser → http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("═"*55 + "\n")

    # Use uvicorn.Config + Server directly to avoid loop_factory issues
    config = uvicorn.Config(
        "phase4.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        loop="asyncio",        # explicit: use standard asyncio loop
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    p = argparse.ArgumentParser(description="PROJECT MONTAGE")
    p.add_argument("--prompt",      type=str, default=None)
    p.add_argument("--manual",      type=str, default=None)
    p.add_argument("--phase2-only", action="store_true")
    p.add_argument("--phase3-only", action="store_true")
    p.add_argument("--scenes",      type=int, default=3)
    p.add_argument("--serve",       action="store_true",
                   help="Start Phase 4 web UI")
    args = p.parse_args()

    if args.serve and not args.prompt and not args.manual:
        serve_web()
        return

    if not args.serve:
        run_pipeline(prompt=args.prompt, manual_script=args.manual,
                     scenes=args.scenes, phase2_only=args.phase2_only,
                     phase3_only=args.phase3_only)
        return

    run_pipeline(prompt=args.prompt, manual_script=args.manual,
                 scenes=args.scenes, phase2_only=args.phase2_only,
                 phase3_only=args.phase3_only)
    serve_web()


if __name__ == "__main__":
    main()


def run_edit(queries=None):
    """Run Phase 5 edit agent."""
    from phase5.workflow import run_edit_session
    run_edit_session(queries=queries)


# Extend main() to support --edit flag
import sys as _sys
if "--edit" in _sys.argv:
    idx = _sys.argv.index("--edit")
    query = " ".join(_sys.argv[idx+1:]) if idx + 1 < len(_sys.argv) else None
    os.chdir(Path(__file__).parent)
    _sys.path.insert(0, str(Path(__file__).parent))
    from phase5.workflow import EditAgent
    agent = EditAgent()
    if query:
        agent.edit(query)
    else:
        from phase5.workflow import run_edit_session
        run_edit_session()
    _sys.exit(0)