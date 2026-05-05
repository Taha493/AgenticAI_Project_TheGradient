"""
StateManager — Versioned Pipeline Snapshots
============================================
Every pipeline run (or partial re-run) creates a numbered snapshot.
Supports full revert to any previous version.

StateManager.snapshot(version, state_json, asset_paths)  → persisted
StateManager.revert(version)   → restores assets + state
StateManager.history()         → list of all versions with summary
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

VERSIONS_DIR = Path("outputs/versions")
STATE_LOG    = Path("outputs/versions/state_log.json")


def _load_log() -> List[Dict]:
    if STATE_LOG.exists():
        with open(STATE_LOG) as f:
            return json.load(f)
    return []


def _save_log(log: List[Dict]):
    STATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_LOG, "w") as f:
        json.dump(log, f, indent=2)


def snapshot(
    version: int,
    state_json: Dict,
    asset_paths: List[str],
    description: str = "",
) -> Dict:
    """
    Save a complete snapshot of the pipeline state at this version.
    Copies all asset files into outputs/versions/v{version}/.
    """
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    snap_dir = VERSIONS_DIR / f"v{version}"
    snap_dir.mkdir(exist_ok=True)

    # Copy assets
    saved_assets = []
    for src in asset_paths:
        src_path = Path(src)
        if src_path.exists():
            dst = snap_dir / src_path.name
            shutil.copy2(src_path, dst)
            saved_assets.append(str(dst))

    # Save state JSON
    state_file = snap_dir / "state.json"
    with open(state_file, "w") as f:
        json.dump(state_json, f, indent=2)

    entry = {
        "version":     version,
        "timestamp":   time.time(),
        "description": description or f"Version {version}",
        "state_file":  str(state_file),
        "assets":      saved_assets,
        "asset_count": len(saved_assets),
    }

    log = _load_log()
    # Replace if version already exists
    log = [e for e in log if e["version"] != version]
    log.append(entry)
    log.sort(key=lambda e: e["version"])
    _save_log(log)

    return entry


def revert(version: int) -> Dict:
    """
    Restore all assets and pipeline state from the specified version snapshot.
    Copies files back to their original locations.
    """
    log = _load_log()
    entry = next((e for e in log if e["version"] == version), None)
    if not entry:
        raise ValueError(f"Version {version} not found in state log.")

    snap_dir   = VERSIONS_DIR / f"v{version}"
    state_file = snap_dir / "state.json"

    if not state_file.exists():
        raise FileNotFoundError(f"State file missing for version {version}.")

    with open(state_file) as f:
        state = json.load(f)

    # Restore asset files
    restored = []
    for asset_path in entry.get("assets", []):
        src = Path(asset_path)
        if not src.exists():
            continue
        # Derive original location from filename
        filename = src.name
        # Common output destinations
        for dest_dir in ["outputs", "raw_scenes", "audio_tracks", "image_assets"]:
            dest = Path(dest_dir) / filename
            if dest.parent.exists():
                shutil.copy2(src, dest)
                restored.append(str(dest))
                break

    return {
        "version":  version,
        "state":    state,
        "restored": restored,
    }


def history() -> List[Dict]:
    """Return all version entries sorted newest-first."""
    log = _load_log()
    result = []
    for entry in reversed(log):
        result.append({
            "version":     entry["version"],
            "timestamp":   entry["timestamp"],
            "description": entry["description"],
            "asset_count": entry["asset_count"],
        })
    return result


def latest_version() -> int:
    log = _load_log()
    if not log:
        return 0
    return max(e["version"] for e in log)