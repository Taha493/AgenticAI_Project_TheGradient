"""
Phase 1 Agent Definitions
==========================
Scriptwriter, Script Validator, HITL, Character Designer, Image Synthesizer.
Each agent discovers tools via MCP — no hardcoded API calls.
"""

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.registry import mcp

load_dotenv()
_client = None  # lazy-initialized

# Model fallback chain — tries each in order if previous gives 503/429
_MODEL_CHAIN = [
    "gemini-2.5-flash",
]


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Add it to .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _llm(prompt: str) -> str:
    """
    Call Gemini with automatic model fallback + retry on 503/429.
    Tries each model in _MODEL_CHAIN; on 503 or 429 waits and retries,
    then moves to next model if still failing.
    """
    import time
    client = _get_client()
    last_err = None

    for model in _MODEL_CHAIN:
        for attempt in range(3):  # 3 attempts per model
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                if attempt > 0 or model != _MODEL_CHAIN[0]:
                    print(f"    [LLM] Success with model={model} attempt={attempt+1}")
                return resp.text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e)
                # 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED → wait and retry
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    wait = 8 * (attempt + 1)
                    print(f"    [LLM] {model} unavailable (503), retrying in {wait}s... (attempt {attempt+1}/3)")
                    time.sleep(wait)
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 15 * (attempt + 1)
                    print(f"    [LLM] {model} rate-limited (429), retrying in {wait}s... (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    # Non-retriable error — break immediately
                    print(f"    [LLM] {model} error: {e}")
                    break  # try next model

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def _extract_json(text: str) -> Dict:
    """Extract first JSON block from LLM response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced block
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # Try any {...} block
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


# ──────────────────────────────────────────────
# 6.1  Scriptwriter Agent
# ──────────────────────────────────────────────
class ScriptwriterAgent:
    """
    Transforms abstract prompts into structured, production-ready scripts.
    MCP tools used: generate_script_segment, commit_memory
    """
    def __init__(self):
        tools = mcp.discover_tools()
        self.tool_generate = mcp.get_tool_schema("generate_script_segment")
        self.tool_memory   = mcp.get_tool_schema("commit_memory")

    def run(self, prompt: str, num_scenes: int = 3) -> Dict:
        print(f"  [Scriptwriter] Generating {num_scenes}-scene script from prompt...")

        # MCP tool: generate_script_segment
        llm_prompt = f"""
You are a professional screenplay writer.
Generate a structured screenplay with exactly {num_scenes} scenes based on this prompt:
"{prompt}"

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "title": "Story Title",
  "genre": "genre",
  "scenes": [
    {{
      "scene_id": 1,
      "location": "Location description",
      "characters": ["CharacterA", "CharacterB"],
      "dialogue": [
        {{
          "speaker": "CharacterA",
          "line": "Dialogue line here",
          "visual_cue": "Camera/lighting description"
        }}
      ],
      "action_description": "What happens in this scene"
    }}
  ]
}}
"""
        raw = _llm(llm_prompt)
        script = _extract_json(raw)

        # MCP tool: commit_memory
        mcp.invoke("commit_memory", {
            "collection": "scripts",
            "doc_id": "current_script",
            "data": script,
        })

        print(f"  [Scriptwriter] Generated '{script.get('title', 'Untitled')}' with {len(script.get('scenes', []))} scenes.")
        return script


# ──────────────────────────────────────────────
# 6.2  Script Validator Agent
# ──────────────────────────────────────────────
class ScriptValidatorAgent:
    """
    Validates manually provided scripts for structure correctness.
    Checks: scene headings, dialogue labels, action descriptions.
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, raw_script: str) -> Dict:
        print("  [Validator] Validating manually provided script...")

        # Try JSON first
        try:
            parsed = json.loads(raw_script)
            scenes = parsed.get("scenes", [])
            errors = []
            for s in scenes:
                if "scene_id" not in s:
                    errors.append(f"Scene missing scene_id: {s}")
                if "location" not in s:
                    errors.append(f"Scene {s.get('scene_id', '?')} missing location")
                if "dialogue" not in s or not s["dialogue"]:
                    errors.append(f"Scene {s.get('scene_id', '?')} missing dialogue")
                if "action_description" not in s:
                    errors.append(f"Scene {s.get('scene_id', '?')} missing action_description")
            if errors:
                return {"valid": False, "errors": errors, "script": None}
            return {"valid": True, "errors": [], "script": parsed}
        except json.JSONDecodeError:
            pass

        # Plain text — use LLM to parse and validate
        llm_prompt = f"""
Parse this screenplay text and convert it to structured JSON.
Check for: scene headings, dialogue labels, action descriptions.
Return ONLY valid JSON with same structure as this example:
{{
  "title": "Title",
  "genre": "drama",
  "scenes": [
    {{
      "scene_id": 1,
      "location": "...",
      "characters": ["..."],
      "dialogue": [{{"speaker": "...", "line": "...", "visual_cue": "..."}}],
      "action_description": "..."
    }}
  ]
}}

Screenplay text:
{raw_script[:3000]}
"""
        try:
            raw = _llm(llm_prompt)
            parsed = _extract_json(raw)
            return {"valid": True, "errors": [], "script": parsed}
        except Exception as e:
            return {"valid": False, "errors": [str(e)], "script": None}


# ──────────────────────────────────────────────
# 6.3  Human-in-the-Loop (HITL) Agent
# ──────────────────────────────────────────────
class HITLAgent:
    """
    Checkpoint: shows script summary to user and requests approval.
    Prevents hallucinated scripts from propagating downstream.
    """
    def run(self, script: Dict, auto_approve: bool = False) -> bool:
        if auto_approve:
            print("  [HITL] Auto-approve mode — skipping manual review.")
            return True

        print("\n" + "═" * 60)
        print("  HUMAN-IN-THE-LOOP REVIEW")
        print("═" * 60)
        print(f"  Title : {script.get('title', 'Untitled')}")
        print(f"  Genre : {script.get('genre', 'Unknown')}")
        print(f"  Scenes: {len(script.get('scenes', []))}")
        for s in script.get("scenes", []):
            print(f"\n  Scene {s['scene_id']} — {s.get('location', '')}")
            for d in s.get("dialogue", [])[:2]:
                print(f"    {d.get('speaker', '?')}: \"{d.get('line', '')}\"")
        print("\n" + "═" * 60)
        answer = input("  Approve this script? [y/n]: ").strip().lower()
        return answer in ("y", "yes", "")


# ──────────────────────────────────────────────
# 6.4  Character Designer Agent
# ──────────────────────────────────────────────
class CharacterDesignerAgent:
    """
    Extracts and formalizes character identities from the script.
    MCP tools used: commit_memory, query_stock_footage
    """
    def __init__(self):
        mcp.discover_tools()

    def run(self, script: Dict) -> List[Dict]:
        print("  [CharacterDesigner] Extracting character identities...")

        # Collect all unique character names
        all_names = set()
        for scene in script.get("scenes", []):
            for name in scene.get("characters", []):
                all_names.add(name)

        characters = []
        for name in sorted(all_names):
            # Query MCP stock footage for style reference
            style_ref = mcp.invoke("query_stock_footage", {
                "character_name": name,
                "style": "cinematic portrait",
            })

            # LLM: generate character profile
            llm_prompt = f"""
Create a character profile for "{name}" based on this script context.
Script title: {script.get('title', '')}
Genre: {script.get('genre', '')}

Return ONLY valid JSON:
{{
  "id": "{name.lower().replace(' ', '_')}",
  "name": "{name}",
  "personality_traits": ["trait1", "trait2", "trait3"],
  "appearance": "detailed physical description for image generation",
  "voice_profile": {{
    "tone": "calm/energetic/serious/etc",
    "accent": "American/British/etc",
    "edge_tts_voice": "en-US-AriaNeural"
  }},
  "reference_style": "cinematic portrait"
}}
"""
            try:
                raw = _llm(llm_prompt)
                profile = _extract_json(raw)
            except Exception:
                profile = {
                    "id": name.lower().replace(" ", "_"),
                    "name": name,
                    "personality_traits": ["determined", "intelligent"],
                    "appearance": f"A character named {name}, professional appearance",
                    "voice_profile": {
                        "tone": "calm",
                        "accent": "American",
                        "edge_tts_voice": "en-US-AriaNeural",
                    },
                    "reference_style": "cinematic portrait",
                }

            characters.append(profile)
            print(f"    → Designed character: {name}")

        # Commit to memory
        mcp.invoke("commit_memory", {
            "collection": "characters",
            "doc_id": "character_db",
            "data": {"characters": characters},
        })

        return characters


# ──────────────────────────────────────────────
# 6.5  Image Synthesizer Agent
# ──────────────────────────────────────────────
class ImageSynthesizerAgent:
    """
    Generates visual character portraits using PIL (CPU-based).
    In production: would call Stable Diffusion / ComfyUI via MCP.
    MCP tool: commit_memory
    """
    VOICE_COLORS = {
        "en-US-AriaNeural":    ((180, 100, 120), (220, 160, 180)),
        "en-US-GuyNeural":     ((80,  120, 180), (140, 180, 220)),
        "en-GB-SoniaNeural":   ((100, 160, 100), (160, 210, 160)),
        "en-US-JennyNeural":   ((160, 120, 200), (200, 170, 230)),
    }

    def run(self, characters: List[Dict], output_dir: str = "image_assets") -> List[str]:
        print("  [ImageSynthesizer] Generating character visuals...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_paths = []

        for char in characters:
            path = self._generate_portrait(char, output_dir)
            image_paths.append(path)
            print(f"    → Generated portrait: {path}")

        # Commit to memory
        mcp.invoke("commit_memory", {
            "collection": "images",
            "doc_id": "image_manifest",
            "data": {"image_paths": image_paths},
        })

        return image_paths

    def _generate_portrait(self, char: Dict, output_dir: str) -> str:
        W, H = 512, 512
        name = char["name"]
        appearance = char.get("appearance", "")
        voice = char.get("voice_profile", {}).get("edge_tts_voice", "en-US-AriaNeural")
        traits = char.get("personality_traits", [])

        c1, c2 = self.VOICE_COLORS.get(voice, ((120, 80, 160), (180, 140, 210)))

        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        # Radial gradient background
        for y in range(H):
            for x in range(W):
                dx = (x - W // 2) / (W // 2)
                dy = (y - H // 2) / (H // 2)
                dist = min(1.0, (dx**2 + dy**2) ** 0.5)
                r = int(c1[0] + (c2[0] - c1[0]) * dist)
                g = int(c1[1] + (c2[1] - c1[1]) * dist)
                b = int(c1[2] + (c2[2] - c1[2]) * dist)
                img.putpixel((x, y), (r, g, b))

        # Silhouette (simple geometric figure)
        draw.ellipse([196, 80, 316, 200], fill=(240, 210, 180))    # head
        draw.rounded_rectangle([166, 200, 346, 380], radius=30,
                                fill=(60, 60, 90))                  # body
        draw.ellipse([130, 200, 196, 320], fill=(60, 60, 90))       # left arm
        draw.ellipse([316, 200, 382, 320], fill=(60, 60, 90))       # right arm

        # Glow ring around head
        draw.ellipse([176, 60, 336, 220], outline=(255, 220, 100), width=3)

        # Name plate
        draw.rectangle([0, H - 120, W, H], fill=(0, 0, 0, 200))

        try:
            font_name  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            font_trait = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font_name = font_trait = font_small = ImageFont.load_default()

        # Center name
        bbox = draw.textbbox((0, 0), name, font=font_name)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, H - 110), name, font=font_name, fill=(255, 220, 100))

        # Traits
        trait_str = "  ·  ".join(traits[:3])
        bbox2 = draw.textbbox((0, 0), trait_str, font=font_trait)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((W - tw2) // 2, H - 68), trait_str, font=font_trait, fill=(200, 200, 200))

        # Voice label
        draw.text((10, H - 30), f"Voice: {voice}", font=font_small, fill=(140, 140, 180))

        # Appearance snippet
        snip = appearance[:60] + ("…" if len(appearance) > 60 else "")
        draw.text((10, H - 110 + 5), "", font=font_small, fill=(180, 180, 180))

        # Soft blur for cinematic feel
        img = img.filter(ImageFilter.GaussianBlur(0.6))

        out_path = str(Path(output_dir) / f"{char['id']}.png")
        img.save(out_path)
        return out_path
