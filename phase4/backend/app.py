"""
PROJECT MONTAGE — FastAPI Backend
===================================
Endpoints:
  POST /api/generate              — run full pipeline (phases 1-3)
  POST /api/rerun/{phase}         — re-run a single phase
  GET  /api/status/{job_id}       — job status
  WS   /ws/progress/{job_id}      — WebSocket real-time progress
  GET  /api/download/video        — stream final_output.mp4
  GET  /api/download/manifest     — timing_manifest.json
  GET  /api/history               — list version snapshots
  POST /api/revert/{version}      — revert to version
  GET  /api/outputs               — list all output files
  GET  /api/scenes                — scene_manifest.json content
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from phase1.workflow import run_phase1
from phase2.workflow import run_phase2
from phase3.workflow import run_phase3
import phase4.state_manager as state_manager

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Project Montage API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve final outputs as static files
Path("outputs").mkdir(exist_ok=True)
# Ensure output dirs exist
for _d in ["outputs", "raw_scenes", "image_assets", "audio_tracks"]:
    Path(_d).mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/raw_scenes", StaticFiles(directory="raw_scenes"), name="raw_scenes")
app.mount("/image_assets", StaticFiles(directory="image_assets"), name="image_assets")
# raw_scenes mounted lazily below
# image_assets mounted lazily below

# ── In-memory job registry ────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_ws_connections: Dict[str, List[WebSocket]] = {}


def _get_job(job_id: str) -> Dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


async def _broadcast(job_id: str, message: Dict):
    """Send a progress message to all WebSocket subscribers for this job."""
    dead = []
    for ws in _ws_connections.get(job_id, []):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections[job_id].remove(ws)


def _emit(job_id: str, phase: str, event: str, detail: str = "", pct: int = 0):
    """Synchronously emit a progress event (called from sync pipeline threads)."""
    msg = {
        "job_id": job_id,
        "phase":  phase,
        "event":  event,
        "detail": detail,
        "pct":    pct,
        "ts":     time.time(),
    }
    _jobs[job_id]["log"].append(msg)
    _jobs[job_id]["last_event"] = msg
    # Schedule async broadcast
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_broadcast(job_id, msg))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    scenes: int = 3
    mode:   str = "auto"   # "auto" | "manual"

class RerunRequest(BaseModel):
    phase: str             # "phase1" | "phase2" | "phase3"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_full_pipeline(job_id: str, prompt: str, scenes: int, mode: str):
    # Apply nest_asyncio only inside the worker thread, not at module level.
    # This lets uvicorn own the main event loop while pipeline workers use theirs.
    try:
        import nest_asyncio, asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    job = _jobs[job_id]
    job["status"] = "running"

    try:
        version = state_manager.latest_version() + 1

        # ── Phase 1 ───────────────────────────────────────────────────────
        _emit(job_id, "phase1", "start", "Generating story and script...", 5)
        job["phase1_status"] = "running"

        p1 = run_phase1(prompt=prompt)

        if p1["status"] != "complete":
            raise RuntimeError(f"Phase 1 failed: {p1.get('error')}")

        job["phase1_status"] = "complete"
        job["manifest_path"] = p1["scene_manifest_path"]
        job["chardb_path"]   = p1["character_db_path"]
        _emit(job_id, "phase1", "complete",
              f"Script '{p1['script'].get('title','')}' ready", 30)

        # ── Phase 2 ───────────────────────────────────────────────────────
        _emit(job_id, "phase2", "start", "Synthesising audio and video...", 32)
        job["phase2_status"] = "running"

        p2 = run_phase2(
            scene_manifest_path=p1["scene_manifest_path"],
            character_db_path=p1["character_db_path"],
        )

        if p2["status"] != "complete":
            raise RuntimeError(f"Phase 2 failed: {p2.get('error')}")

        job["phase2_status"] = "complete"
        job["raw_scenes"]    = p2.get("output_videos", [])
        _emit(job_id, "phase2", "complete",
              f"{len(p2.get('output_videos',[]))} scenes rendered", 65)

        # ── Phase 3 ───────────────────────────────────────────────────────
        _emit(job_id, "phase3", "start", "Compositing final video...", 67)
        job["phase3_status"] = "running"

        p3 = run_phase3(
            scene_manifest_path=p1["scene_manifest_path"],
            character_db_path=p1["character_db_path"],
            phase2_log_path="task_logs/phase2_log.json",
            version=version,
        )

        if p3["status"] != "complete":
            raise RuntimeError(f"Phase 3 failed: {p3.get('error')}")

        job["phase3_status"]   = "complete"
        job["final_video"]     = p3["final_video_path"]
        job["timing_manifest"] = p3["timing_manifest_path"]
        _emit(job_id, "phase3", "complete",
              f"Final video ready: {p3['final_video_path']}", 95)

        # ── Snapshot ──────────────────────────────────────────────────────
        asset_paths = [
            p3["final_video_path"] or "",
            p3["timing_manifest_path"] or "",
            p1["scene_manifest_path"],
            p1["character_db_path"],
        ] + (p2.get("output_videos") or [])

        state_manager.snapshot(
            version=version,
            state_json={
                "prompt": prompt,
                "phase1": {"title": p1["script"].get("title", ""), "scenes": p1["script"].get("scenes", [])},
                "phase2": {"raw_scenes": p2.get("output_videos", [])},
                "phase3": {"final_video": p3["final_video_path"]},
            },
            asset_paths=[a for a in asset_paths if a],
            description=f"Generated: {prompt[:60]}",
        )

        job["status"]  = "complete"
        job["version"] = version
        _emit(job_id, "pipeline", "complete", "All phases done!", 100)

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)
        for ph in ["phase1", "phase2", "phase3"]:
            if job.get(f"{ph}_status") == "running":
                job[f"{ph}_status"] = "failed"
        _emit(job_id, "pipeline", "error", str(e), 0)


def _rerun_phase(job_id: str, phase: str):
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    job = _jobs[job_id]
    job["status"] = "running"

    try:
        version = state_manager.latest_version() + 1

        if phase == "phase1":
            _emit(job_id, "phase1", "start", "Re-running Phase 1...", 5)
            p1 = run_phase1(prompt=job.get("prompt", ""))
            if p1["status"] != "complete":
                raise RuntimeError(f"Phase 1 failed: {p1.get('error')}")
            job["manifest_path"] = p1["scene_manifest_path"]
            job["chardb_path"]   = p1["character_db_path"]
            job["phase1_status"] = "complete"
            _emit(job_id, "phase1", "complete", "Phase 1 re-run complete", 100)

        elif phase == "phase2":
            _emit(job_id, "phase2", "start", "Re-running Phase 2...", 5)
            p2 = run_phase2(
                scene_manifest_path=job.get("manifest_path", "outputs/scene_manifest.json"),
                character_db_path=job.get("chardb_path", "outputs/character_db.json"),
            )
            if p2["status"] != "complete":
                raise RuntimeError(f"Phase 2 failed: {p2.get('error')}")
            job["raw_scenes"]    = p2.get("output_videos", [])
            job["phase2_status"] = "complete"
            _emit(job_id, "phase2", "complete", "Phase 2 re-run complete", 100)

        elif phase == "phase3":
            _emit(job_id, "phase3", "start", "Re-running Phase 3...", 5)
            p3 = run_phase3(
                scene_manifest_path=job.get("manifest_path", "outputs/scene_manifest.json"),
                character_db_path=job.get("chardb_path", "outputs/character_db.json"),
                phase2_log_path="task_logs/phase2_log.json",
                version=version,
            )
            if p3["status"] != "complete":
                raise RuntimeError(f"Phase 3 failed: {p3.get('error')}")
            job["final_video"]     = p3["final_video_path"]
            job["timing_manifest"] = p3["timing_manifest_path"]
            job["phase3_status"]   = "complete"
            _emit(job_id, "phase3", "complete", "Phase 3 re-run complete", 100)

        job["status"]  = "complete"
        job["version"] = version
        _emit(job_id, "pipeline", "complete", f"{phase} re-run done", 100)

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)
        _emit(job_id, "pipeline", "error", str(e), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/generate")
async def generate(req: GenerateRequest, bg: BackgroundTasks):
    """Start the full Phase 1→2→3 pipeline. Returns a job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id":        job_id,
        "prompt":        req.prompt,
        "scenes":        req.scenes,
        "status":        "queued",
        "phase1_status": "pending",
        "phase2_status": "pending",
        "phase3_status": "pending",
        "log":           [],
        "last_event":    None,
        "final_video":   None,
        "timing_manifest": None,
        "manifest_path": None,
        "chardb_path":   None,
        "raw_scenes":    [],
        "version":       None,
        "error":         None,
    }
    bg.add_task(_run_full_pipeline, job_id, req.prompt, req.scenes, req.mode)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/rerun/{phase}")
async def rerun_phase(phase: str, bg: BackgroundTasks):
    """Re-run a single phase using the most recent job's outputs as input."""
    if phase not in ("phase1", "phase2", "phase3"):
        raise HTTPException(400, "phase must be phase1, phase2, or phase3")

    job_id = str(uuid.uuid4())[:8]
    # Inherit paths from last completed job if available
    last = next(reversed(_jobs.values()), {}) if _jobs else {}
    _jobs[job_id] = {
        "job_id":        job_id,
        "prompt":        last.get("prompt", ""),
        "status":        "queued",
        "phase1_status": "pending",
        "phase2_status": "pending",
        "phase3_status": "pending",
        "log":           [],
        "last_event":    None,
        "final_video":   last.get("final_video"),
        "timing_manifest": last.get("timing_manifest"),
        "manifest_path": last.get("manifest_path", "outputs/scene_manifest.json"),
        "chardb_path":   last.get("chardb_path", "outputs/character_db.json"),
        "raw_scenes":    last.get("raw_scenes", []),
        "version":       None,
        "error":         None,
    }
    bg.add_task(_rerun_phase, job_id, phase)
    return {"job_id": job_id, "phase": phase, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = _get_job(job_id)
    return {
        "job_id":        job["job_id"],
        "status":        job["status"],
        "phase1_status": job.get("phase1_status", "pending"),
        "phase2_status": job.get("phase2_status", "pending"),
        "phase3_status": job.get("phase3_status", "pending"),
        "final_video":   job.get("final_video"),
        "timing_manifest": job.get("timing_manifest"),
        "raw_scenes":    job.get("raw_scenes", []),
        "version":       job.get("version"),
        "error":         job.get("error"),
        "last_event":    job.get("last_event"),
    }


@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """Real-time progress stream for a job."""
    await websocket.accept()
    _ws_connections.setdefault(job_id, []).append(websocket)

    # Send backlog of past events immediately on connect
    if job_id in _jobs:
        for msg in _jobs[job_id].get("log", []):
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    try:
        while True:
            await asyncio.sleep(1)
            if job_id in _jobs:
                job = _jobs[job_id]
                if job["status"] in ("complete", "failed"):
                    await websocket.send_json({
                        "job_id": job_id,
                        "phase":  "pipeline",
                        "event":  job["status"],
                        "detail": job.get("error", "Done"),
                        "pct":    100 if job["status"] == "complete" else 0,
                    })
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _ws_connections and websocket in _ws_connections[job_id]:
            _ws_connections[job_id].remove(websocket)


@app.get("/api/download/video")
async def download_video():
    path = Path("outputs/final_output.mp4")
    if not path.exists():
        raise HTTPException(404, "Final video not yet generated")
    return FileResponse(str(path), media_type="video/mp4",
                        filename="montage_output.mp4")


@app.get("/api/download/manifest")
async def download_manifest():
    path = Path("outputs/timing_manifest.json")
    if not path.exists():
        raise HTTPException(404, "Timing manifest not yet generated")
    return FileResponse(str(path), media_type="application/json",
                        filename="timing_manifest.json")


@app.get("/api/history")
async def get_history():
    return {"versions": state_manager.history()}


@app.post("/api/revert/{version}")
async def revert_version(version: int):
    try:
        result = state_manager.revert(version)
        return {"status": "reverted", "version": version,
                "restored_files": result["restored"]}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/outputs")
async def list_outputs():
    """List all generated output files."""
    files = {}
    for folder in ["outputs", "raw_scenes", "audio_tracks", "image_assets"]:
        p = Path(folder)
        if p.exists():
            files[folder] = [
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "url":  f"/{folder}/{f.name}",
                }
                for f in sorted(p.iterdir())
                if f.is_file() and not f.name.startswith(".")
            ]
    return files


@app.get("/api/scenes")
async def get_scenes():
    path = Path("outputs/scene_manifest.json")
    if not path.exists():
        raise HTTPException(404, "scene_manifest.json not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/characters")
async def get_characters():
    path = Path("outputs/character_db.json")
    if not path.exists():
        raise HTTPException(404, "character_db.json not found")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase4.backend.app:app", host="0.0.0.0", port=8000, reload=False)


# ── Serve frontend HTML ───────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse

FRONTEND_HTML = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_HTML.exists():
        return HTMLResponse(content=FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Edit Agent API
# ─────────────────────────────────────────────────────────────────────────────

from phase5.workflow import EditAgent
from phase5.intent_classifier import IntentClassifier

_edit_agent    = EditAgent()
_classifier_p5 = IntentClassifier()


class EditRequest(BaseModel):
    query: str


@app.post("/api/edit")
async def apply_edit(req: EditRequest, bg: BackgroundTasks):
    """
    Apply a free-text edit command.
    Classifies intent → executes edit → snapshots result.
    Returns immediately with job_id; WebSocket streams progress.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id":  job_id,
        "type":    "edit",
        "query":   req.query,
        "status":  "queued",
        "log":     [],
        "last_event": None,
        "result":  None,
        "error":   None,
    }

    def _run_edit(jid: str, query: str):
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass
        _jobs[jid]["status"] = "running"
        _emit(jid, "edit", "start", f"Processing: {query}", 10)
        try:
            # Classify
            _emit(jid, "edit", "classifying", "Classifying intent...", 30)
            intent = _classifier_p5.classify(query)
            _emit(jid, "edit", "classified",
                  f"Intent: {intent['intent']} → {intent['target']}", 50)

            # Execute
            _emit(jid, "edit", "executing",
                  f"Executing {intent['target']} edit...", 60)
            final = _edit_agent.edit(query)

            _jobs[jid]["result"] = {
                "intent":   final.get("intent"),
                "status":   final["status"],
                "version_after": final.get("version_after"),
                "error":    final.get("error"),
            }
            _jobs[jid]["status"] = final["status"]
            _emit(jid, "edit", final["status"],
                  f"v{final.get('version_after', '?')} saved" if final["status"] == "complete"
                  else final.get("error", ""), 100)
        except Exception as e:
            _jobs[jid]["status"] = "failed"
            _jobs[jid]["error"]  = str(e)
            _emit(jid, "edit", "error", str(e), 0)

    bg.add_task(_run_edit, job_id, req.query)
    return {"job_id": job_id, "query": req.query}


@app.post("/api/edit/classify")
async def classify_edit(req: EditRequest):
    """Preview what an edit command would do without executing it."""
    intent = _classifier_p5.classify(req.query)
    return {"query": req.query, "intent": intent}


@app.post("/api/undo")
async def undo_edit(version: Optional[int] = None):
    """Undo the last edit, or revert to a specific version."""
    try:
        result = _edit_agent.undo(version)
        return {"status": "reverted", "result": result}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/edit/history")
async def get_edit_history():
    """Return the edit session history for this server instance."""
    return {
        "session_history": _edit_agent.session_history(),
        "version_history": _edit_agent.history(),
    }