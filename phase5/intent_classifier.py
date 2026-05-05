"""
Phase 5 — Intent Classifier
============================
LLM-powered classification agent.
Maps free-text edit queries → structured EditIntent objects.

Covers 10+ edit types:
  change_voice_tone      → audio
  change_voice_speed     → audio
  add_background_music   → audio
  remove_background_music→ audio
  make_scene_darker      → video_frame
  make_scene_brighter    → video_frame
  apply_filter           → video_frame
  change_character_design→ video_frame
  speed_up_scene         → video
  slow_down_scene        → video
  remove_subtitle        → video
  add_subtitle           → video
  regenerate_script      → script
  change_scene_mood      → script
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Intent catalogue ──────────────────────────────────────────────────────────

INTENT_CATALOGUE = {
    # ── AUDIO intents ────────────────────────────────────────────────────────
    "change_voice_tone": {
        "target": "audio",
        "description": "Change the vocal tone/emotion of a character's voice",
        "keywords": ["voice", "tone", "speak", "whisper", "shout", "calm", "angry",
                     "emotion", "dramatic", "softer", "louder", "nervous"],
        "param_keys": ["tone", "character", "scene_id"],
    },
    "change_voice_speed": {
        "target": "audio",
        "description": "Change the speaking speed/rate of a character",
        "keywords": ["faster", "slower", "speed", "rate", "pace", "talking",
                     "quick", "slow", "speech"],
        "param_keys": ["speed", "character", "scene_id"],
    },
    "add_background_music": {
        "target": "audio",
        "description": "Add or replace background music for a scene",
        "keywords": ["music", "bgm", "background", "score", "soundtrack",
                     "ambient", "add music", "play"],
        "param_keys": ["mood", "scene_id", "volume"],
    },
    "remove_background_music": {
        "target": "audio",
        "description": "Remove background music from a scene",
        "keywords": ["remove background", "no background", "remove the music",
                     "no music please", "mute the music", "silence the music",
                     "remove bgm", "turn off the music", "without music",
                     "delete background music", "remove background music"],
        "param_keys": ["scene_id"],
    },

    # ── VIDEO_FRAME intents ──────────────────────────────────────────────────
    "make_scene_darker": {
        "target": "video_frame",
        "description": "Darken the visual appearance of a scene",
        "keywords": ["darker", "dark", "dim", "shadow", "gloomy",
                     "noir", "murky", "low light"],
        "param_keys": ["scene_id", "intensity"],
    },
    "make_scene_brighter": {
        "target": "video_frame",
        "description": "Brighten the visual appearance of a scene",
        "keywords": ["brighter", "bright", "light", "vivid", "illuminate",
                     "lighten", "exposure"],
        "param_keys": ["scene_id", "intensity"],
    },
    "apply_filter": {
        "target": "video_frame",
        "description": "Apply a visual filter (sepia, grayscale, warm, cool, etc.)",
        "keywords": ["filter", "sepia", "grayscale", "black and white",
                     "vintage", "cool", "warm", "cinematic", "color grade",
                     "tint", "effect"],
        "param_keys": ["filter", "scene_id", "intensity"],
    },
    "change_character_design": {
        "target": "video_frame",
        "description": "Regenerate character visual for a scene",
        "keywords": ["character", "design", "appearance", "look", "style",
                     "outfit", "redesign", "portrait"],
        "param_keys": ["character", "scene_id", "description"],
    },

    # ── VIDEO intents ────────────────────────────────────────────────────────
    "speed_up_scene": {
        "target": "video",
        "description": "Increase playback speed of a scene",
        "keywords": ["speed up", "faster", "quick", "accelerate", "2x",
                     "timelapse", "hurry"],
        "param_keys": ["scene_id", "factor"],
    },
    "slow_down_scene": {
        "target": "video",
        "description": "Decrease playback speed of a scene (slow motion)",
        "keywords": ["slow", "slow motion", "slowmo", "slo-mo", "decelerate",
                     "0.5x"],
        "param_keys": ["scene_id", "factor"],
    },
    "remove_subtitle": {
        "target": "video",
        "description": "Remove subtitle overlay from video",
        "keywords": ["remove subtitle", "no subtitle", "no captions",
                     "hide text", "subtitle off"],
        "param_keys": ["scene_id"],
    },
    "add_subtitle": {
        "target": "video",
        "description": "Add or re-enable subtitle overlay",
        "keywords": ["add subtitle", "caption", "subtitle", "show text",
                     "text overlay"],
        "param_keys": ["scene_id"],
    },

    # ── SCRIPT intents ───────────────────────────────────────────────────────
    "regenerate_script": {
        "target": "script",
        "description": "Regenerate the entire story/script from scratch",
        "keywords": ["regenerate the script", "regenerate script", "rewrite script",
                     "new script", "redo script", "regenerate story",
                     "rewrite the script", "new story", "redo story",
                     "remake story", "fresh script", "generate new"],
        "param_keys": ["prompt", "preserve_characters"],
    },
    "change_scene_mood": {
        "target": "script",
        "description": "Change the mood/tone of a specific scene in the script",
        "keywords": ["mood", "tone", "atmosphere", "feeling", "vibe",
                     "scene feeling", "make it scary", "make it funny"],
        "param_keys": ["scene_id", "mood"],
    },
}


# ── Keyword-based fallback classifier (no LLM needed) ────────────────────────

def _keyword_classify(query: str) -> Optional[Dict]:
    """
    Fast keyword-based classifier used when LLM is unavailable.
    Scores each intent by keyword overlap (weighted by keyword length for specificity).
    """
    q_lower = query.lower()
    best_intent = None
    best_score  = 0.0

    for intent_name, meta in INTENT_CATALOGUE.items():
        # Weight longer keywords higher (more specific phrases beat single words)
        score = sum(len(kw.split()) for kw in meta["keywords"] if kw in q_lower)
        if score > best_score:
            best_score  = score
            best_intent = intent_name

    if not best_intent or best_score == 0:
        return None

    meta    = INTENT_CATALOGUE[best_intent]
    target  = meta["target"]
    scope   = _extract_scope(q_lower, target)
    params  = _extract_params(q_lower, best_intent)

    return {
        "intent":     best_intent,
        "target":     target,
        "scope":      scope,
        "parameters": params,
        "confidence": min(0.6, best_score * 0.2),
    }


def _extract_scope(query: str, target: str) -> str:
    """Extract scope (character name or scene number) from query text."""
    # Scene number
    m = re.search(r"scene\s*(\d+)", query, re.IGNORECASE)
    if m:
        return f"scene:{m.group(1)}"

    # Character name (capitalized word after 'character' or before 'voice')
    m = re.search(r"(?:character|for|voice of|by)\s+([A-Z][A-Za-z\s]+?)(?:\s|$|'s)", query)
    if m:
        return f"character:{m.group(1).strip()}"

    return "all"


def _extract_params(query: str, intent: str) -> Dict:
    """Extract intent-specific parameters from query."""
    params: Dict = {}
    q = query.lower()

    if intent == "change_voice_tone":
        for tone in ["whispered", "whisper", "angry", "calm", "dramatic",
                     "sad", "excited", "nervous", "serious", "cheerful"]:
            if tone in q:
                params["tone"] = tone
                break

    elif intent in ("change_voice_speed", "speed_up_scene", "slow_down_scene"):
        m = re.search(r"(\d+(?:\.\d+)?)\s*x", q)
        if m:
            params["factor"] = float(m.group(1))
        elif "double" in q or "2x" in q:
            params["factor"] = 2.0
        elif "half" in q or "0.5x" in q:
            params["factor"] = 0.5
        else:
            params["factor"] = 1.5 if "fast" in q else 0.75

    elif intent in ("make_scene_darker", "make_scene_brighter"):
        m = re.search(r"(\d+)\s*%", q)
        params["intensity"] = int(m.group(1)) if m else 50

    elif intent == "apply_filter":
        for f in ["sepia", "grayscale", "black and white", "warm", "cool",
                  "vintage", "cinematic", "blur", "sharpen"]:
            if f in q:
                params["filter"] = f.replace(" ", "_")
                break
        if "filter" not in params:
            params["filter"] = "sepia"

    elif intent == "add_background_music":
        for mood in ["tense", "mysterious", "dramatic", "happy",
                     "sad", "action", "ambient", "romantic"]:
            if mood in q:
                params["mood"] = mood
                break
        if "mood" not in params:
            params["mood"] = "mysterious"
        params["volume"] = 0.2

    elif intent in ("regenerate_script", "change_scene_mood"):
        for mood in ["scary", "funny", "dark", "romantic", "action",
                     "mysterious", "dramatic", "comedic"]:
            if mood in q:
                params["mood"] = mood
                break

    return params


# ── LLM-powered classifier ────────────────────────────────────────────────────

class IntentClassifier:
    """
    LLM-powered edit intent classifier.
    Falls back to keyword matching if LLM is unavailable.
    """
    SYSTEM_PROMPT = """You are an intent classifier for a video editing AI system.
Given a user's free-text edit command, return ONLY a valid JSON object.

Available intents and their targets:
""" + "\n".join(
        f'  "{k}": target="{v["target"]}" — {v["description"]}'
        for k, v in INTENT_CATALOGUE.items()
    ) + """

Response format (JSON only, no markdown):
{
  "intent": "<intent_name>",
  "target": "<audio|video_frame|video|script>",
  "scope": "<all|scene:N|character:NAME>",
  "parameters": { <relevant key-value pairs> },
  "confidence": <0.0-1.0>
}

Rules:
- scope should be "scene:N" if a scene number is mentioned
- scope should be "character:NAME" if a character name is mentioned
- scope is "all" if no specific scene or character is mentioned
- confidence should reflect how clear the intent is (0.9 for obvious, 0.5 for ambiguous)
- If the query is truly ambiguous, pick the most likely intent
"""

    def __init__(self):
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                from google import genai
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    self._llm_client = genai.Client(api_key=api_key)
            except Exception:
                pass
        return self._llm_client

    def classify(self, query: str) -> Dict:
        """
        Classify a free-text edit query into a structured EditIntent.
        Returns EditIntent dict.
        """
        # Try LLM first
        client = self._get_llm_client()
        if client:
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{self.SYSTEM_PROMPT}\n\nUser query: {query}",
                )
                text = resp.text.strip()
                # Strip markdown fences if present
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
                intent_data = json.loads(text)
                # Validate required keys
                for k in ("intent", "target", "scope", "parameters", "confidence"):
                    if k not in intent_data:
                        raise ValueError(f"Missing key: {k}")
                return intent_data
            except Exception as e:
                pass  # Fall through to keyword classifier

        # Keyword fallback
        result = _keyword_classify(query)
        if result:
            return result

        # Unknown intent
        return {
            "intent":     "unknown",
            "target":     "unknown",
            "scope":      "all",
            "parameters": {},
            "confidence": 0.0,
        }