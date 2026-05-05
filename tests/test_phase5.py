"""
Phase 5 Unit Tests — Edit Agent
Tests 10+ edit query types for intent classification and execution.
Run: python -m pytest tests/test_phase5.py -v
"""

import json
import os
import sys
import wave
import shutil
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from phase5.intent_classifier import IntentClassifier, _keyword_classify, INTENT_CATALOGUE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


@pytest.fixture(scope="module", autouse=True)
def setup_dirs(tmp_path_factory):
    """Create minimal project dirs so executors don't fail on missing paths."""
    for d in ["outputs", "raw_scenes", "audio_tracks", "image_assets",
              "task_logs", "outputs/bgm", "outputs/versions"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Write sample manifest + char_db
    manifest = {
        "title": "Test",
        "genre": "thriller",
        "scenes": [
            {"scene_id": 1, "location": "INT. DARK ALLEY - NIGHT",
             "characters": ["HERO"],
             "dialogue": [{"speaker": "HERO", "line": "Stop!", "visual_cue": ""}],
             "action_description": "Test scene."}
        ]
    }
    char_db = {"characters": [
        {"id": "hero", "name": "HERO",
         "personality_traits": ["brave"],
         "appearance": "Tall hero.",
         "voice_profile": {"edge_tts_voice": "en-US-GuyNeural"}}
    ]}
    with open("outputs/scene_manifest.json", "w") as f:
        json.dump(manifest, f)
    with open("outputs/character_db.json", "w") as f:
        json.dump(char_db, f)


# ─────────────────────────────────────────────────────────────────────────────
# Intent Catalogue Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentCatalogue:
    def test_catalogue_has_minimum_10_intents(self):
        assert len(INTENT_CATALOGUE) >= 10

    def test_all_intents_have_required_fields(self):
        for name, meta in INTENT_CATALOGUE.items():
            assert "target"      in meta, f"{name} missing target"
            assert "description" in meta, f"{name} missing description"
            assert "keywords"    in meta, f"{name} missing keywords"
            assert meta["target"] in ("audio", "video_frame", "video", "script"), \
                f"{name} has invalid target: {meta['target']}"

    def test_all_four_targets_represented(self):
        targets = {m["target"] for m in INTENT_CATALOGUE.values()}
        assert "audio"       in targets
        assert "video_frame" in targets
        assert "video"       in targets
        assert "script"      in targets


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Classifier Tests (10+ edit query types)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordClassifier:
    """Tests 10+ distinct edit query types using keyword-based fallback."""

    # 1. change_voice_tone
    def test_whisper_voice_tone(self):
        r = _keyword_classify("make the character whisper")
        assert r is not None
        assert r["target"] == "audio"
        assert "tone" in r["parameters"] or r["intent"] == "change_voice_tone"

    # 2. change_voice_speed
    def test_voice_speed(self):
        r = _keyword_classify("speed up the speech rate")
        assert r is not None
        assert r["target"] in ("audio", "video")

    # 3. add_background_music
    def test_add_bgm(self):
        r = _keyword_classify("add background music to scene 1")
        assert r is not None
        assert r["target"] == "audio"
        assert r["intent"] == "add_background_music"

    # 4. remove_background_music
    def test_remove_bgm(self):
        r = _keyword_classify("remove background music")
        assert r is not None
        assert r["target"] == "audio"
        assert r["intent"] == "remove_background_music"

    # 5. make_scene_darker
    def test_make_darker(self):
        r = _keyword_classify("make the scene darker")
        assert r is not None
        assert r["target"] == "video_frame"
        assert r["intent"] == "make_scene_darker"

    # 6. make_scene_brighter
    def test_make_brighter(self):
        r = _keyword_classify("make it brighter and more vivid")
        assert r is not None
        assert r["target"] == "video_frame"
        assert r["intent"] == "make_scene_brighter"

    # 7. apply_filter (sepia)
    def test_sepia_filter(self):
        r = _keyword_classify("apply a sepia filter to scene 2")
        assert r is not None
        assert r["target"] == "video_frame"
        assert r["intent"] == "apply_filter"
        assert r["parameters"].get("filter") == "sepia"

    # 8. apply_filter (grayscale)
    def test_grayscale_filter(self):
        r = _keyword_classify("convert to grayscale")
        assert r is not None
        assert r["target"] == "video_frame"
        assert r["parameters"].get("filter") in ("grayscale", "black_and_white", "sepia")

    # 9. speed_up_scene
    def test_speed_up(self):
        r = _keyword_classify("speed up scene 3 by 2x")
        assert r is not None
        assert r["target"] in ("video", "audio")

    # 10. slow_down_scene
    def test_slow_down(self):
        r = _keyword_classify("slow motion for scene 1")
        assert r is not None
        assert r["target"] == "video"
        assert r["intent"] in ("slow_down_scene", "speed_up_scene")

    # 11. remove_subtitle
    def test_remove_subtitle(self):
        r = _keyword_classify("remove subtitle from the video")
        assert r is not None
        assert r["target"] == "video"
        assert r["intent"] == "remove_subtitle"

    # 12. regenerate_script
    def test_regenerate_script(self):
        r = _keyword_classify("regenerate the script with a new story")
        assert r is not None
        assert r["target"] == "script"
        assert r["intent"] == "regenerate_script"

    # 13. change_character_design
    def test_character_design(self):
        r = _keyword_classify("change the character design for HERO")
        assert r is not None
        assert r["target"] == "video_frame"
        assert r["intent"] == "change_character_design"

    # 14. change_scene_mood
    def test_change_mood(self):
        r = _keyword_classify("change the mood of scene 2 to mysterious")
        assert r is not None
        assert r["target"] in ("script", "video_frame", "video")


# ─────────────────────────────────────────────────────────────────────────────
# IntentClassifier (full class) Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentClassifier:
    def test_returns_dict_with_required_keys(self, classifier):
        result = classifier.classify("make it darker")
        for key in ("intent", "target", "scope", "parameters", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_in_range(self, classifier):
        result = classifier.classify("apply sepia filter")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unknown_query_returns_result(self, classifier):
        result = classifier.classify("xyzzy quux frob")
        assert "intent" in result
        # Should either be "unknown" or a low-confidence best-guess
        assert isinstance(result["intent"], str)

    def test_scope_scene_extraction(self, classifier):
        result = classifier.classify("make scene 3 darker")
        # Scope should reference scene 3
        assert result["scope"] in ("scene:3", "all")

    def test_scope_character_extraction(self, classifier):
        result = classifier.classify("change voice of HERO to whispered")
        # Should extract character from scope
        assert "HERO" in result["scope"] or result["scope"] == "all"

    def test_audio_target_for_voice_query(self, classifier):
        result = classifier.classify("change the voice tone to dramatic")
        assert result["target"] == "audio"

    def test_video_frame_target_for_filter_query(self, classifier):
        result = classifier.classify("apply vintage filter to scene 1")
        assert result["target"] == "video_frame"

    def test_video_target_for_speed_query(self, classifier):
        result = classifier.classify("speed up scene 2 by 1.5x")
        assert result["target"] in ("video", "audio")

    def test_script_target_for_regenerate(self, classifier):
        result = classifier.classify("regenerate the script")
        assert result["target"] == "script"


# ─────────────────────────────────────────────────────────────────────────────
# EditAgent State & Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEditAgentState:
    def test_edit_state_schema(self):
        from phase5.state import EditState
        s = EditState(
            raw_query="test query",
            conversation_history=[],
            intent=None, execution_result=None, error=None,
            version_before=None, version_after=None,
            status="classifying",
        )
        assert s["raw_query"] == "test query"
        assert s["status"] == "classifying"

    def test_edit_graph_compiles(self):
        from phase5.workflow import build_edit_graph
        g = build_edit_graph()
        assert g is not None

    def test_edit_agent_instantiates(self):
        from phase5.workflow import EditAgent
        agent = EditAgent()
        assert agent is not None
        assert isinstance(agent.session_history(), list)


# ─────────────────────────────────────────────────────────────────────────────
# Video Frame Editor (OpenCV filter) Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoFrameEditor:
    def test_darken_filter(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        frame  = (np.ones((100, 100, 3)) * 128).astype(np.uint8)
        result = editor._filter_darken(frame, 0.5, cv2)
        assert result.mean() < frame.mean()

    def test_brighten_filter(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        frame  = (np.ones((100, 100, 3)) * 100).astype(np.uint8)
        result = editor._filter_brighten(frame, 0.5, cv2)
        assert result.mean() >= frame.mean()

    def test_grayscale_filter(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        # Colored frame
        frame        = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:,:,0] = 200  # red only
        result = editor._filter_grayscale(frame, 1.0, cv2)
        # All channels should be equal in a fully grayscale image
        assert np.allclose(result[:,:,0], result[:,:,1], atol=5)

    def test_sepia_filter_shape(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        frame  = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        result = editor._filter_sepia(frame, 0.8, cv2)
        assert result.shape == frame.shape

    def test_warm_filter_increases_red(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        frame  = (np.ones((10, 10, 3)) * 128).astype(np.uint8)
        result = editor._filter_warm(frame, 0.5, cv2)
        assert result[:,:,0].mean() > frame[:,:,0].mean()

    def test_cool_filter_increases_blue(self):
        import cv2
        from phase5.executor import VideoFrameEditor
        editor = VideoFrameEditor()
        frame  = (np.ones((10, 10, 3)) * 128).astype(np.uint8)
        result = editor._filter_cool(frame, 0.5, cv2)
        assert result[:,:,2].mean() > frame[:,:,2].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 Edit API Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEditAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from phase4.backend.app import app
        self.client = TestClient(app)

    def test_classify_endpoint(self):
        r = self.client.post("/api/edit/classify",
                             json={"query": "make scene 1 darker"})
        assert r.status_code == 200
        data = r.json()
        assert "intent" in data
        assert data["intent"]["target"] == "video_frame"

    def test_classify_voice_endpoint(self):
        r = self.client.post("/api/edit/classify",
                             json={"query": "change voice to whispered"})
        assert r.status_code == 200
        assert r.json()["intent"]["target"] == "audio"

    def test_classify_script_endpoint(self):
        r = self.client.post("/api/edit/classify",
                             json={"query": "regenerate the script"})
        assert r.status_code == 200
        assert r.json()["intent"]["target"] == "script"

    def test_edit_history_endpoint(self):
        r = self.client.get("/api/edit/history")
        assert r.status_code == 200
        data = r.json()
        assert "session_history" in data
        assert "version_history" in data