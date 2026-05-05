"""
Unit Tests — PROJECT MONTAGE
Tests each phase module independently with mock inputs.
Run:  python -m pytest tests/ -v
"""

import json
import os
import sys
import wave
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

# ── Setup project path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_MANIFEST = {
    "title": "Test Story",
    "genre": "thriller",
    "scenes": [
        {
            "scene_id": 1,
            "location": "INT. DARK WAREHOUSE - NIGHT",
            "characters": ["HERO", "VILLAIN"],
            "dialogue": [
                {"speaker": "HERO",    "line": "We need to talk.",   "visual_cue": "Close-up"},
                {"speaker": "VILLAIN", "line": "It's too late now.", "visual_cue": "Wide shot"},
            ],
            "action_description": "Hero confronts villain in an abandoned warehouse."
        }
    ]
}

SAMPLE_CHAR_DB = {
    "characters": [
        {
            "id": "hero",
            "name": "HERO",
            "personality_traits": ["brave", "determined"],
            "appearance": "Tall person in dark clothing.",
            "voice_profile": {"tone": "calm", "accent": "American",
                              "edge_tts_voice": "en-US-GuyNeural"},
        },
        {
            "id": "villain",
            "name": "VILLAIN",
            "personality_traits": ["cunning", "ruthless"],
            "appearance": "Shadowy figure in a long coat.",
            "voice_profile": {"tone": "serious", "accent": "British",
                              "edge_tts_voice": "en-GB-RyanNeural"},
        },
    ]
}


@pytest.fixture(scope="session")
def tmp_project(tmp_path_factory):
    """Create a temporary project directory with sample fixtures."""
    d = tmp_path_factory.mktemp("montage_test")
    (d / "outputs").mkdir()
    (d / "raw_scenes").mkdir()
    (d / "audio_tracks").mkdir()
    (d / "image_assets").mkdir()
    (d / "task_logs").mkdir()
    (d / "outputs" / "versions").mkdir()

    with open(d / "outputs" / "scene_manifest.json", "w") as f:
        json.dump(SAMPLE_MANIFEST, f)
    with open(d / "outputs" / "character_db.json", "w") as f:
        json.dump(SAMPLE_CHAR_DB, f)

    return d


@pytest.fixture(scope="session")
def sample_video(tmp_project):
    """Write a tiny 1-second MP4-like file using imageio."""
    import imageio
    out = str(tmp_project / "raw_scenes" / "scene_01.mp4")
    frames = [np.zeros((480, 864, 3), dtype=np.uint8) for _ in range(24)]
    writer = imageio.get_writer(out, fps=24, codec="libx264", quality=5)
    for f in frames:
        writer.append_data(f)
    try:
        writer.close()
    except Exception:
        pass
    return out


@pytest.fixture(scope="session")
def sample_audio(tmp_project):
    """Write a 1-second silent WAV file."""
    out = str(tmp_project / "audio_tracks" / "scene_1_merged.mp3")
    with wave.open(out.replace(".mp3", ".wav"), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    shutil.copy(out.replace(".mp3", ".wav"), out)
    return out


@pytest.fixture(scope="session")
def phase2_log(tmp_project, sample_video, sample_audio):
    """Write a mock phase2_log.json."""
    log = [{
        "scene_id":   1,
        "audio_path": str(tmp_project / "audio_tracks" / "scene_1_merged.mp3"),
        "video_path": str(tmp_project / "raw_scenes"   / "scene_01.mp4"),
        "output_path": str(tmp_project / "raw_scenes"  / "scene_01.mp4"),
        "status": "success",
    }]
    path = str(tmp_project / "task_logs" / "phase2_log.json")
    with open(path, "w") as f:
        json.dump(log, f)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1:
    def test_mcp_registry_loads(self):
        from mcp_server.registry import mcp, TOOL_REGISTRY
        assert len(TOOL_REGISTRY) >= 9
        assert "generate_script_segment" in TOOL_REGISTRY
        assert "commit_memory" in TOOL_REGISTRY

    def test_mcp_discover_tools(self):
        from mcp_server.registry import mcp
        tools = mcp.discover_tools()
        assert isinstance(tools, dict)
        assert len(tools) > 0

    def test_commit_memory_tool(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        r = mcp.invoke("commit_memory", {
            "collection": "test",
            "doc_id":     "unit_test",
            "data":       {"key": "value"},
        })
        assert r["status"] == "committed"
        assert r["doc_id"] == "unit_test"

    def test_image_synthesizer_no_llm(self, tmp_project):
        os.chdir(tmp_project)
        from phase1.agents import ImageSynthesizerAgent
        agent  = ImageSynthesizerAgent()
        paths  = agent.run(SAMPLE_CHAR_DB["characters"],
                           output_dir=str(tmp_project / "image_assets"))
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 1000   # non-empty image

    def test_phase1_state_schema(self):
        from phase1.state import Phase1State
        s = Phase1State(
            input_mode="auto", raw_input="test",
            validation_result=None, script=None, hitl_approved=None,
            characters=None, images=None, scene_manifest_path=None,
            character_db_path=None, status="processing", error=None,
        )
        assert s["input_mode"] == "auto"
        assert s["status"] == "processing"

    def test_hitl_auto_approve(self):
        from phase1.agents import HITLAgent
        agent    = HITLAgent()
        approved = agent.run(SAMPLE_MANIFEST, auto_approve=True)
        assert approved is True

    def test_script_validator_valid_json(self, tmp_project):
        os.chdir(tmp_project)
        from phase1.agents import ScriptValidatorAgent
        agent  = ScriptValidatorAgent()
        result = agent.run(json.dumps(SAMPLE_MANIFEST))
        assert result["valid"] is True
        assert len(result["script"]["scenes"]) == 1

    def test_script_validator_invalid_json(self, tmp_project):
        os.chdir(tmp_project)
        from phase1.agents import ScriptValidatorAgent
        agent  = ScriptValidatorAgent()
        result = agent.run('{"scenes": [{"missing_id": true}]}')
        assert result["valid"] is False

    def test_query_stock_footage_tool(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        r = mcp.invoke("query_stock_footage", {
            "character_name": "HERO", "style": "cinematic"
        })
        assert r["status"] == "success"
        assert r["character_name"] == "HERO"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2:
    def test_get_task_graph_tool(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        r = mcp.invoke("get_task_graph", {
            "scene_manifest_path": str(tmp_project / "outputs" / "scene_manifest.json")
        })
        assert r["total_tasks"] == 1
        assert r["parallelizable"] is True
        assert r["task_graph"][0]["scene_id"] == 1

    def test_identity_validator_found(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        r = mcp.invoke("identity_validator", {
            "character_id": "HERO",
            "character_db_path": str(tmp_project / "outputs" / "character_db.json")
        })
        assert r["valid"] is True

    def test_identity_validator_not_found(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        r = mcp.invoke("identity_validator", {
            "character_id": "NONEXISTENT",
            "character_db_path": str(tmp_project / "outputs" / "character_db.json")
        })
        assert r["valid"] is False

    def test_voice_synthesis_creates_file(self, tmp_project):
        os.chdir(tmp_project)
        from mcp_server.registry import mcp
        out = str(tmp_project / "audio_tracks" / "test_tts.mp3")
        r = mcp.invoke("voice_cloning_synthesizer", {
            "text": "Hello world.", "voice": "en-US-GuyNeural", "output_path": out
        })
        assert r["status"] == "success"
        assert Path(out).exists()
        assert Path(out).stat().st_size > 100

    def test_phase2_state_schema(self):
        from phase2.state import Phase2State
        s = Phase2State(
            scene_manifest_path="a", character_db_path="b",
            task_graph=None, audio_results=None, video_results=None,
            face_swap_results=None, lip_sync_results=None,
            output_videos=None, task_logs=None,
            status="processing", error=None,
        )
        assert s["status"] == "processing"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3:
    def test_clip_validator_finds_clips(self, tmp_project, sample_video, sample_audio, phase2_log):
        os.chdir(tmp_project)
        from phase3.agents import ClipValidatorAgent
        agent = ClipValidatorAgent()
        clips = agent.run(phase2_log, str(tmp_project / "outputs" / "scene_manifest.json"))
        assert len(clips) == 1
        assert clips[0]["scene_id"] == 1
        assert clips[0]["mood"] in ("tense", "mysterious", "dark", "serious", "dramatic", "ambient", "neutral")

    def test_infer_mood(self):
        from phase3.agents import _infer_mood
        assert _infer_mood("INT. DARK WAREHOUSE - NIGHT") == "tense"
        assert _infer_mood("INT. JAZZ CLUB - NIGHT")      == "mysterious"
        assert _infer_mood("INT. POLICE PRECINCT - DAY")  == "serious"
        assert _infer_mood("INT. APARTMENT - MORNING")    == "dramatic"

    def test_bgm_generator_creates_wav(self, tmp_project):
        os.chdir(tmp_project)
        from phase3.agents import BGMSelectorAgent
        agent = BGMSelectorAgent()
        clips = [{
            "scene_id": 1, "mood": "tense", "duration_s": 2.0,
            "video_path": "", "audio_path": "", "location": "", "dialogue": [], "action": ""
        }]
        paths = agent.run(clips)
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert Path(paths[0]).stat().st_size > 500

    def test_subtitle_schedule_correct_count(self):
        from phase3.agents import SubtitleBurnerAgent
        agent = SubtitleBurnerAgent()
        dialogues = [{"speaker": "A", "line": "Hello."}, {"speaker": "B", "line": "Hi!"}]
        schedule  = agent._build_subtitle_schedule(dialogues, 4.0, 24.0)
        assert len(schedule) == 2
        assert schedule[0]["start_frame"] == 0
        assert schedule[1]["start_frame"] > 0

    def test_timing_manifest_writer(self, tmp_project):
        os.chdir(tmp_project)
        from phase3.agents import TimingManifestAgent
        agent   = TimingManifestAgent()
        entries = [{"scene_id": 1, "location": "Test", "start_ms": 0,
                    "end_ms": 5000, "start_frame": 0, "end_frame": 120,
                    "audio_path": "", "mood": "tense"}]
        out     = str(tmp_project / "outputs" / "timing_manifest_test.json")
        path    = agent.run(entries, out)
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert data["total_scenes"] == 1
        assert data["total_duration_ms"] == 5000

    def test_phase3_state_schema(self):
        from phase3.state import Phase3State
        s = Phase3State(
            scene_manifest_path="a", character_db_path="b", phase2_log_path="c",
            scene_clips=None, bgm_paths=None, timed_clips=None, subtitled_clips=None,
            timing_manifest_path=None, final_video_path=None, version=1,
            status="processing", error=None,
        )
        assert s["version"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4:
    def test_state_manager_snapshot(self, tmp_project):
        os.chdir(tmp_project)
        import phase4.state_manager as sm
        entry = sm.snapshot(
            version=99,
            state_json={"prompt": "test"},
            asset_paths=[],
            description="Unit test snapshot",
        )
        assert entry["version"] == 99
        assert "Unit test" in entry["description"]

    def test_state_manager_history(self, tmp_project):
        os.chdir(tmp_project)
        import phase4.state_manager as sm
        history = sm.history()
        assert isinstance(history, list)
        # Should have at least the snapshot we just created
        assert any(e["version"] == 99 for e in history)

    def test_state_manager_revert(self, tmp_project):
        os.chdir(tmp_project)
        import phase4.state_manager as sm
        # Create a snapshot with a real file
        test_file = str(tmp_project / "outputs" / "test_asset.json")
        with open(test_file, "w") as f:
            json.dump({"revert_test": True}, f)
        sm.snapshot(version=98, state_json={"x": 1},
                    asset_paths=[test_file], description="Revert test")
        result = sm.revert(98)
        assert result["version"] == 98

    def test_state_manager_latest_version(self, tmp_project):
        os.chdir(tmp_project)
        import phase4.state_manager as sm
        v = sm.latest_version()
        assert isinstance(v, int)
        assert v >= 98

    @pytest.mark.asyncio
    async def test_api_health(self, tmp_project):
        os.chdir(tmp_project)
        from fastapi.testclient import TestClient
        from phase4.backend.app import app
        client = TestClient(app)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_api_history_endpoint(self, tmp_project):
        os.chdir(tmp_project)
        from fastapi.testclient import TestClient
        from phase4.backend.app import app
        client = TestClient(app)
        r = client.get("/api/history")
        assert r.status_code == 200
        assert "versions" in r.json()

    @pytest.mark.asyncio
    async def test_api_outputs_endpoint(self, tmp_project):
        os.chdir(tmp_project)
        from fastapi.testclient import TestClient
        from phase4.backend.app import app
        client = TestClient(app)
        r = client.get("/api/outputs")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_api_frontend_served(self, tmp_project):
        os.chdir(tmp_project)
        from fastapi.testclient import TestClient
        from phase4.backend.app import app
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "Project Montage" in r.text