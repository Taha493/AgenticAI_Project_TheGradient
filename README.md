# 🎬 Project Montage
### AI-Powered Animated Video Generation System
*From Prompt to Polished Short Film — End-to-End with LLM Agents*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B6B?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-66%2F66_Passing-34D399?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

**Group:** Inference Lab &nbsp;|&nbsp; **Course:** Agentic AI CS-4015 &nbsp;|&nbsp; **FAST-NUCES Islamabad, Spring 2026**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Phases](#-phases)
- [Tech Stack](#-tech-stack)
- [Models Used](#-models-used)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Web Interface](#-web-interface)
- [Edit Agent](#-edit-agent)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Team](#-team)

---

## 🌟 Overview

Project Montage is an **end-to-end, agent-orchestrated video generation system** that accepts a single natural-language prompt and autonomously produces a complete short animated video — including story, dialogue, character voices, visual scenes, background music, and synchronized subtitles — with **zero manual creative intervention**.

The system is built as a **five-phase agentic pipeline**, each phase implemented as a LangGraph `StateGraph` with clearly defined inputs, outputs, and inter-phase JSON contracts.

> **Key Constraint:** All tools are discovered dynamically via an MCP (Model Context Protocol) registry — zero hardcoded API calls anywhere in the codebase.

```
"A detective investigates a mysterious disappearance in a rain-soaked city"
                                    ↓
        ┌─────────────────────────────────────────────┐
        │          PROJECT MONTAGE PIPELINE            │
        │  Phase 1 → Phase 2 → Phase 3 → Phase 4/5   │
        └─────────────────────────────────────────────┘
                                    ↓
                    final_output.mp4  (complete short film)
```

---

## 🎥 Demo

```powershell
# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
cp .env.example .env
# Edit .env → GEMINI_API_KEY=your_key_here

# Run the full pipeline
python run.py --prompt "A scientist discovers a portal to a parallel world"

# OR start the web interface
python run.py --serve
# Open → http://localhost:8000
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Agent Pipeline** | 5 LangGraph StateGraphs, 27+ agent nodes |
| 🔌 **MCP Tool Discovery** | 9 tools discovered at runtime — zero hardcoded APIs |
| 🎙️ **Voice Synthesis** | Microsoft Neural voices (online) + pyttsx3/SAPI5 (offline) |
| 🎬 **Animated Scenes** | PIL-rendered scenes with location-aware backgrounds |
| 🎵 **Background Music** | NumPy harmonic synthesis per scene mood |
| 📝 **Synced Subtitles** | Audio-derived timing via ffprobe — exact sync |
| 🎞️ **Video Composition** | MoviePy FadeIn/FadeOut transitions, BGM mixing |
| 🌐 **Web Interface** | FastAPI + React, WebSocket real-time progress |
| ✏️ **Edit Agent** | 14 intent types, natural language commands |
| ↩️ **Undo / Revert** | Full versioned snapshots at every operation |
| 🧪 **66 Unit Tests** | All passing across all 5 phases |
| 💻 **CPU Only** | No GPU required — runs on Windows, Linux, macOS |

---

## 🏗️ System Architecture

```
User Prompt
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 — Story, Script & Character Design (LangGraph)         │
│  mode_selector → scriptwriter/validator → hitl →                │
│  character_node → image_node → memory_commit                    │
│  OUTPUT: scene_manifest.json, character_db.json, portraits      │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2 — Audio Generation & Scene Video (LangGraph)           │
│  scene_parser → voice_synth → video_gen →                       │
│  face_swap → lip_sync                                           │
│  OUTPUT: audio_tracks/*.mp3, raw_scenes/*.mp4                   │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3 — Video Composition (LangGraph)                        │
│  clip_validator → bgm_selector → subtitle_burner →              │
│  compositor → manifest_writer                                   │
│  OUTPUT: outputs/final_output.mp4, timing_manifest.json         │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4 — Web Interface (FastAPI + React)                      │
│  Orchestrates Phases 1–3 · WebSocket progress ·                 │
│  Version history · Per-phase rerun · Video preview              │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5 — Intelligent Edit & Undo Agent (LangGraph)            │
│  classify_intent → validate_intent → execute_edit               │
│  14 intent types · OpenCV filters · Full undo stack             │
└─────────────────────────────────────────────────────────────────┘
```

### MCP Tool Registry

All agents call `mcp.discover_tools()` at init and `mcp.invoke("tool_name", {...})` at runtime:

| Tool | Phase | Purpose |
|---|---|---|
| `generate_script_segment` | 1 | Gemini screenplay generation |
| `commit_memory` | 1+2 | Persist state to JSON store |
| `query_stock_footage` | 1 | Character visual style reference |
| `get_task_graph` | 2 | Decompose manifest into tasks |
| `voice_cloning_synthesizer` | 2 | TTS synthesis |
| `video_scene_composer` | 2 | Animated scene video generation |
| `identity_validator` | 2 | Validate character before face swap |
| `face_swapper` | 2 | Portrait overlay on video frame |
| `lip_sync_aligner` | 2+3 | Fuse audio and video streams |

---

## 📐 Phases

### Phase 1 — Story, Script & Character Design

Transforms a free-text prompt into a structured screenplay using Gemini 2.5 Flash.

- **Auto mode:** LLM generates multi-scene script with dialogue, visual cues, character descriptions
- **Manual mode:** Validates and standardizes uploaded scripts
- **HITL checkpoint:** Human approval gate before proceeding
- **Character Designer:** Generates personality profiles and voice assignments
- **Image Synthesizer:** Creates PIL-based character portraits

**Outputs:** `outputs/scene_manifest.json`, `outputs/character_db.json`, `image_assets/*.png`

### Phase 2 — Audio Generation & Scene Video

Produces per-scene MP4 files with synchronized audio.

- **Voice Synthesis:** Groups same-speaker lines → one TTS call per group. Cached pyttsx3 engine (1 SAPI5 init per run, not per line). Reduces Phase 2 from 50+ min → 8–12 min on Windows.
- **Video Generation:** PIL frame-by-frame animated scenes. Hold time per line = actual audio duration via ffprobe.
- **Face Swap:** Identity validated via MCP before portrait compositing
- **Lip Sync:** MoviePy fuses dialogue audio + video (audio preserved)

**Outputs:** `audio_tracks/scene_N_grp_*.mp3`, `raw_scenes/scene_NN.mp4`

### Phase 3 — Video Composition

Composes the final video from per-scene clips.

- **BGM Selector:** NumPy harmonic synthesis — tense, mysterious, dramatic, ambient moods
- **Subtitle Burner:** MoviePy `clip.transform()` — audio track preserved, timing from ffprobe
- **Compositor:** MoviePy FadeIn/FadeOut, `concatenate_videoclips()`, BGM at 18% volume
- **Timing Manifest:** `{scene_id, start_ms, end_ms, audio_path, mood}` per scene

**Outputs:** `outputs/final_output.mp4`, `outputs/timing_manifest.json`

### Phase 4 — Web Interface

Full-stack application orchestrating all phases.

- **Backend:** FastAPI + Uvicorn, 9 REST endpoints, WebSocket progress streaming
- **Frontend:** React 18 (CDN, no build step), served directly from FastAPI
- **State Manager:** Versioned snapshots at `outputs/versions/vN/` for every run/edit

### Phase 5 — Intelligent Edit & Undo Agent

Natural language editing with 14 intent types across 4 targets:

| Target | Intents |
|---|---|
| `audio` | change_voice_tone, change_voice_speed, add_background_music, remove_background_music |
| `video_frame` | make_scene_darker, make_scene_brighter, apply_filter, change_character_design |
| `video` | speed_up_scene, slow_down_scene, remove_subtitle, add_subtitle |
| `script` | regenerate_script, change_scene_mood |

---

## 🛠️ Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Gemini 2.5 Flash | Free tier, structured JSON output |
| Orchestration | LangGraph 0.2+ | StateGraph, conditional routing |
| TTS (primary) | edge-tts (Microsoft Neural) | Needs internet |
| TTS (fallback) | pyttsx3 + SAPI5/espeak | Fully offline, cached engine |
| Image Rendering | PIL + imageio | No GPU required |
| Video Composition | MoviePy 2.x + FFmpeg | Audio preservation |
| CV Filters | OpenCV (cv2) | Phase 5 frame-level edits |
| BGM | NumPy harmonic synthesis | No external API |
| Backend | FastAPI + Uvicorn | Async, WebSocket native |
| Frontend | React 18 (CDN) | No build step needed |
| State Store | JSON + shutil | Versioned snapshots |

---

## 🤖 Models Used

### LLM
- **Gemini 2.5 Flash** — script generation (Phase 1) and intent classification (Phase 5)

### Text-to-Speech
- **Microsoft Neural voices** via edge-tts: `en-US-AriaNeural`, `en-US-GuyNeural`, `en-GB-SoniaNeural`, `en-US-JennyNeural`, `en-US-EricNeural`, `en-GB-RyanNeural`
- **Windows SAPI5** via pyttsx3: `Microsoft David`, `Microsoft Zira` (offline fallback)
- **espeak-ng** via pyttsx3: `gmw/en`, `gmw/en-gb-x-rp` (Linux/macOS offline fallback)

### Computer Vision
- **PIL (Pillow)** — character portraits, frame rendering, subtitle overlay
- **OpenCV (cv2)** — sepia, grayscale, darken, brighten, warm, cool, vintage, blur, sharpen filters

### Audio / Video
- **NumPy harmonic synthesis** — custom background music model
- **MoviePy 2.x** — composition, transitions, audio mixing
- **imageio + FFmpeg** — raw frame writing

---

## 📁 Project Structure

```
montage_full/
├── run.py                          ← Master entry point (all phases)
├── requirements.txt
├── .env.example                    ← Copy to .env, add Gemini key
│
├── mcp_server/
│   └── registry.py                 ← 9 MCP tools, dynamic discovery
│
├── phase1/                         ← Story, Script & Character Design
│   ├── agents.py                   ← 5 agents
│   ├── state.py                    ← Phase1State TypedDict
│   └── workflow.py                 ← LangGraph StateGraph (7 nodes)
│
├── phase2/                         ← Audio & Scene Video
│   ├── agents.py                   ← 5 agents
│   ├── state.py
│   └── workflow.py                 ← LangGraph StateGraph (5 nodes)
│
├── phase3/                         ← Video Composition
│   ├── agents.py                   ← 5 agents
│   ├── state.py
│   └── workflow.py                 ← LangGraph StateGraph (5 nodes)
│
├── phase4/                         ← Web Interface
│   ├── backend/
│   │   └── app.py                  ← FastAPI app (9 endpoints + WebSocket)
│   ├── frontend/
│   │   └── index.html              ← React 18 single-file frontend
│   └── state_manager.py            ← Versioned snapshots + undo
│
├── phase5/                         ← Edit Agent
│   ├── intent_classifier.py        ← Gemini + keyword fallback classifier
│   ├── executor.py                 ← AudioEditor, VideoFrameEditor, VideoEditor, ScriptEditor
│   ├── state.py
│   └── workflow.py                 ← LangGraph edit graph + EditAgent class
│
├── tests/
│   ├── test_all_phases.py          ← Phase 1–4 unit tests (27 tests)
│   └── test_phase5.py              ← Phase 5 unit tests (39 tests)
│
├── outputs/                        ← Generated JSON outputs
├── image_assets/                   ← Character portraits
├── audio_tracks/                   ← TTS audio files
├── raw_scenes/                     ← Scene MP4 files
└── task_logs/                      ← Execution logs
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.12+
- FFmpeg (for video processing)
- Git

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/AgenticAI_Project_InferenceLab.git
cd AgenticAI_Project_InferenceLab
```

### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

**Linux only — for offline TTS:**
```bash
sudo apt install espeak-ng libespeak-ng1
```

**Optional — for better audio quality:**
```powershell
pip install pydub
# Install FFmpeg: https://ffmpeg.org/download.html
```

### 4. Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create a free API key (no billing required for Gemini 2.5 Flash)

### 5. Configure environment

```powershell
copy .env.example .env       # Windows
cp .env.example .env         # Linux / macOS
```

Edit `.env`:
```env
GEMINI_API_KEY=AIza...your_key_here
```

### 6. Create output directories

```powershell
mkdir outputs\versions
mkdir outputs\bgm
```

---

## 🚀 Running the Project

### Full Pipeline (Phases 1 → 2 → 3)

```powershell
python run.py --prompt "A detective investigates a mysterious disappearance"
```

### Custom scene count

```powershell
python run.py --prompt "A heist gone wrong in a neon-lit city" --scenes 4
```

### Manual script input

```powershell
python run.py --manual my_script.json
```

### Skip Phase 1+2 (reuse existing outputs)

```powershell
python run.py --phase3-only
```

### Start the web interface

```powershell
python run.py --serve
# Open http://localhost:8000
```

### Run pipeline then open web UI

```powershell
python run.py --prompt "A space explorer finds alien ruins" --serve
```

### Apply a single edit from CLI

```powershell
python run.py --edit "make scene 2 darker"
```

### Interactive edit session

```powershell
python run.py --edit
# Type commands at the Edit> prompt
# Type 'undo', 'history', or 'quit'
```

---

## 🌐 Web Interface

After running `python run.py --serve`, open **http://localhost:8000**

| Tab | What You Can Do |
|---|---|
| **Progress** | Live phase status, WebSocket log feed, elapsed time ticker |
| **Video** | Preview final MP4, download button, per-scene video players |
| **Scenes** | Full script — scenes, dialogue, visual cues |
| **Characters** | Character roster with portraits and voice profiles |
| **Timing** | `start_ms` / `end_ms` table for every scene |
| **Edit** | Natural language edit commands + version history + undo |

### Re-running individual phases

Use the "Re-run Phase" buttons in the sidebar to regenerate just Phase 1, 2, or 3 without restarting the full pipeline.

---

## ✏️ Edit Agent

After generating a video, describe edits in plain English:

```python
from phase5.workflow import EditAgent

agent = EditAgent()

# Apply edits
agent.edit("make scene 2 darker")
agent.edit("change the voice tone to dramatic")
agent.edit("apply sepia filter to all scenes")
agent.edit("speed up scene 3 by 1.5x")
agent.edit("add background music")

# Undo
agent.undo()              # revert last edit
agent.undo(version=3)     # revert to specific version

# History
agent.history()           # list all versions
agent.session_history()   # edits in this session
```

### Example Edit Commands

```
"Make scene 1 darker"
"Change the voice to whispered"
"Apply sepia filter to scene 2"
"Speed up scene 3"
"Add background music"
"Remove subtitles"
"Make it brighter"
"Change voice to dramatic"
"Apply grayscale filter"
"Slow down scene 1"
"Regenerate the script"
"Change scene mood to mysterious"
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | Start full pipeline. Body: `{"prompt": "...", "scenes": 3}` |
| `WS` | `/ws/progress/{job_id}` | Real-time WebSocket progress stream |
| `GET` | `/api/status/{job_id}` | Poll job + per-phase status |
| `POST` | `/api/rerun/{phase}` | Re-run `phase1`, `phase2`, or `phase3` |
| `GET` | `/api/download/video` | Stream `final_output.mp4` |
| `GET` | `/api/download/manifest` | Download `timing_manifest.json` |
| `GET` | `/api/history` | List all version snapshots |
| `POST` | `/api/revert/{version}` | Revert to version `v` |
| `GET` | `/api/outputs` | List all generated files |
| `GET` | `/api/scenes` | Get `scene_manifest.json` content |
| `GET` | `/api/characters` | Get `character_db.json` content |
| `POST` | `/api/edit` | Apply natural language edit command |
| `POST` | `/api/edit/classify` | Preview edit intent (no execution) |
| `POST` | `/api/undo` | Undo last edit or revert to version |
| `GET` | `/api/edit/history` | Session edit log + version history |

---

## 🧪 Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Exclude network-dependent TTS test
python -m pytest tests/ -v -k "not voice_synthesis"

# Run specific phase tests
python -m pytest tests/test_all_phases.py -v
python -m pytest tests/test_phase5.py -v
```

**Test coverage:**

| Phase | Tests | Status |
|---|---|---|
| Phase 1 | 8 | ✅ PASS |
| Phase 2 | 5 | ✅ PASS |
| Phase 3 | 7 | ✅ PASS |
| Phase 4 | 7 | ✅ PASS |
| Phase 5 | 39 | ✅ PASS |
| **Total** | **66** | **66/66 ✅** |

---

## 👥 Team

**Group Name:** Inference Lab  
**Course:** Agentic AI CS-4015 — FAST-NUCES Islamabad, Spring 2026

| Member | Roll No. | Phases |
|---|---|---|
| Member 1 | [Roll No.] | Phase 1 (Story & Script) + Phase 2 (Audio & Video) |
| Member 2 | [Roll No.] | Phase 3 (Video Composition) |
| Member 3 | [Roll No.] | Phase 4 (Web Interface) + Phase 5 (Edit Agent) |

---

## 📊 Performance

| Phase | Task | Time (CPU, Windows 11) |
|---|---|---|
| Phase 1 | Script + character generation | 45–90 sec |
| Phase 2 | TTS audio synthesis | 8–12 min |
| Phase 2 | Scene video generation | 3–5 min |
| Phase 3 | Subtitle burn + composition | 3–6 min |
| **Total** | **Full pipeline (3 scenes)** | **15–25 min** |

> **Note:** Phase 2 TTS was originally 50+ minutes. Reduced to 8–12 min by caching the pyttsx3/SAPI5 engine (1 initialization per run instead of 1 per dialogue line) and grouping consecutive same-speaker lines into single TTS calls.

---

## 📄 License

This project is submitted as coursework for CS-4015 Agentic AI at FAST-NUCES Islamabad.

---

<div align="center">
  <strong>Inference Lab</strong> · FAST-NUCES Islamabad · Spring 2026<br>
  <em>Built with LangGraph · Gemini · FastAPI · React · MoviePy · OpenCV</em>
</div>
