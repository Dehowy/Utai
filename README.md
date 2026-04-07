<p align="center">
  <img src="icon.png" width="96" height="96" alt="Utai">
</p>

<h1 align="center">「Utai」</h1>

<p align="center">
  <b>AI Voice Conversion Workstation</b><br>
  <sub>Node-based audio workflow for audio separation, conversion, and RVC training</sub>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-7b68ee?style=flat-square" alt="version">
  <img src="https://img.shields.io/badge/platform-Windows-0078d4?style=flat-square" alt="platform">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%2030%2F40%2F50-76b900?style=flat-square" alt="gpu">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license">
  <img src="https://img.shields.io/badge/i18n-zh%20%7C%20en%20%7C%20ja-e8b84b?style=flat-square" alt="languages">
</p>

---

## What is Utai?

**Utai** (Uta[歌] + AI) is a desktop application for AI-powered audio processing and singing synthesis. It integrates multiple projects and provides a visual node-based workflow for audio separation, voice conversion, and media export — plus a built-in DAW for AI singing voice synthesis using DiffSinger and RVC.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Audio Separation** | Split audio into vocals, instrumentals, drums, bass, etc. using 113 MSST/UVR models |
| **Voice Conversion** | Convert voice timbre using RVC (Retrieval-based Voice Conversion) |
| **RVC Training** | Train custom voice models from audio datasets with full pipeline automation |
| **Media Download** | Download audio/video from URLs via yt-dlp integration |
| **Audio Merge** | Mix multiple audio tracks (vocals + instrumental + effects) into one |
| **Media Export** | Export with full ffmpeg encoding control (codec, bitrate, filters, metadata) |
| **Singing Synthesis** | AI singing voice synthesis using DiffSinger voice banks with RVC timbre conversion |
| **DAW / Arrangement** | Multi-track piano roll, arrangement view, transport bar, track mixer, and audio effects |
| **VST Plugin Host** | Load and play VST2/VST3 instrument plugins via JUCE integration |
| **Project Files** | Save/share workflows as `.utai` files and training projects as `.utaiTr` files |

## Quick Start (End Users)

### Requirements

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA RTX 30/40/50 series (driver >= 525) recommended; CPU fallback available
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~10GB for the application, additional space for models

### Launch

1. Download the release package (.zip) from website
2. Extract the package
3. Double-click `Utai.exe`
4. First launch takes 10-20 seconds (loading ML frameworks)

### File Association

Right-click any `.utai` or `.utaiTr` file > **Open with** > Browse to `Utai.exe` > Check **"Always use this app"**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Electron Shell (Utai.exe)                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  BrowserWindow                                    │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  React Frontend (React Flow + Canvas DAW)    │  │  │
│  │  │  Served from http://127.0.0.1:{port}        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                         ▲ HTTP/WS                        │
│                         ▼                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Python Backend (FastAPI + Uvicorn)               │  │
│  │  ├─ Node Executor (topological sort, async)       │  │
│  │  ├─ MSST Wrapper (113 separation models)          │  │
│  │  ├─ RVC Wrapper (inference + 5-step training)     │  │
│  │  ├─ yt-dlp Wrapper (media download)               │  │
│  │  ├─ Audio Buffer Manager (disk-backed)            │  │
│  │  ├─ Synth Wrapper (DiffSinger synthesis)          │  │
│  │  │  └─ UtaiRenderer.exe (.NET 8 / OpenUtau Core)  │  │
│  │  └─ WebSocket (real-time progress)                │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Desktop Shell | Electron 33 |
| Frontend | React 19, TypeScript, Vite 6 |
| Node Editor | @xyflow/react 12 (React Flow) |
| State Management | Zustand 5 |
| Audio Visualization | WaveSurfer.js 7 |
| Audio Engine | Web Audio API (multi-track mixer, effects) |
| Singing Synthesis | DiffSinger (via OpenUtau Core, .NET 8) |
| Plugin Host | JUCE (VST2/VST3 instruments) |
| Backend | FastAPI 0.115, Uvicorn, Python 3.10 |
| ML Framework | PyTorch 2.5.1 (CUDA 12.1) |
| Audio Processing | librosa, soundfile, pedalboard, soxr |
| Separation | MSST (Music Source Separation Training), UVR |
| Voice Conversion | RVC (Retrieval-based Voice Conversion) |
| Media Tools | ffmpeg, yt-dlp |
| i18n | i18next (Chinese, English, Japanese) |

---

## Structure

```
utai/
├── app/
│   ├── frontend/                # React application
│   │   ├── src/
│   │   │   ├── nodes/           # Node UI components (workflow)
│   │   │   │   ├── AudioInputNode.tsx
│   │   │   │   ├── SeparationNode.tsx
│   │   │   │   ├── RVCNode.tsx
│   │   │   │   ├── AudioMergeNode.tsx
│   │   │   │   ├── MediaOutputNode.tsx
│   │   │   │   ├── YtdlpNode.tsx
│   │   │   │   └── training/    # Training page nodes
│   │   │   ├── store/           # Zustand stores
│   │   │   ├── components/      # Shared UI (Toolbar, Palette, etc.)
│   │   │   ├── api/             # Backend API client
│   │   │   ├── i18n/            # Translations (zh/en/ja)
│   │   │   ├── styles/          # utai.css (retro theme)
│   │   │   ├── components/synth/ # DAW: piano roll, arrangement, mixer, effects
│   │   │   ├── lib/synth/       # Canvas engines, audio engine, effects
│   │   │   └── pages/           # WorkflowPage, TrainingPage, SynthPage
│   │   └── package.json
│   │
│   └── backend/                 # FastAPI application
│       ├── main.py              # App entry, CORS, static serving
│       ├── config.py            # Path resolution, device detection
│       ├── core/
│       │   ├── executor.py      # Workflow execution engine
│       │   ├── graph.py         # DAG validation, topological sort
│       │   ├── audio_buffer.py  # Disk-backed audio storage
│       │   ├── node_registry.py # Node type discovery
│       │   └── model_categories.py
│       ├── nodes/               # Node implementations (workflow)
│       │   ├── base.py          # BaseNode ABC, ExecutionContext
│       │   ├── audio_input.py
│       │   ├── separation.py    # Unified MSST/VR dispatcher
│       │   ├── rvc_conversion.py
│       │   ├── audio_merge.py
│       │   └── ...
│       ├── wrappers/            # ML tool wrappers
│       │   ├── msst_wrapper.py  # MSST model loading & inference
│       │   ├── rvc_wrapper.py   # RVC inference
│       │   ├── rvc_train_wrapper.py  # 5-step training pipeline
│       │   └── ytdlp_wrapper.py
│       ├── api/                 # FastAPI route handlers
│       │   ├── routes_execute.py
│       │   ├── routes_workflow.py  # .utai/.utaiTr import/export
│       │   ├── routes_models.py
│       │   ├── routes_train.py
│       │   └── routes_synth.py    # Synth render, singer management
│       └── schemas/             # Pydantic models
│
├── electron/                    # Electron desktop wrapper
│   ├── main.js                  # Main process, backend spawning
│   ├── preload.js               # IPC bridge for native dialogs
│   ├── loading.html             # Splash screen
│   └── package.json
│
├── MSST/                        # MSST source & model configs
│   ├── configs/                 # Per-model YAML configs
│   ├── data/models_info.json    # 113 model definitions & URLs
│   └── modules/                 # MSST inference modules
│
├── synth/                       # Singing synthesis
│   ├── utai-renderer/           # UtaiRenderer.exe (.NET 8 CLI)
│   │   └── Singers/             # DiffSinger voice banks
│   └── utai-renderer-src/       # C# source for renderer
│
├── RVC/                         # RVC source
│   └── RVC20240604Nvidia/
│       ├── infer/               # Inference & training scripts
│       ├── assets/              # Pretrained models, HuBERT, RMVPE
│       └── configs/             # Training configs (32k/40k/48k)
│
├── bin/                         # ffmpeg binary
├── yt-dlp/                      # yt-dlp source
├── venv/                        # Python virtual environment
├── Utai/                        # Built portable application
│
├── requirements.txt             # Python dependencies
├── requirements-torch-cu121.txt # PyTorch CUDA 12.1
├── requirements-torch-cu124.txt # PyTorch CUDA 12.4
├── requirements-torch-cpu.txt   # PyTorch CPU
├── setup_env.bat                # Environment setup script
├── run.bat                      # Development server launcher
├── rebuild_utai.bat             # Rebuild & sync to Utai/
└── icon.png                     # Application icon
```

---

## Development Setup

### Prerequisites

- **Python 3.10** (via `py -3.10` launcher on Windows)
- **Node.js 20+** (for frontend build & Electron)
- **NVIDIA GPU** with CUDA support (optional, CPU fallback available)
- **ffmpeg** (place in `bin/` or install system-wide)

### 1. Python Environment

```bash
# Run the setup script (creates venv, installs PyTorch + dependencies)
setup_env.bat

# Or manually:
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements-torch-cu121.txt  # or cu124/cpu
pip install -r requirements.txt
```

> **Note**: After installing librosa 0.9.2, you may need to patch `site-packages/librosa/util/utils.py` line ~2185:
> Change `np.dtype(complex): np.dtype(np.float).type` to `np.dtype(complex): np.dtype(float).type`

### 2. External Dependencies

Utai integrates several external projects. These are **not included in the git repository** — you need to set them up separately:

#### MSST (Music Source Separation Training)

```bash
# Clone MSST into the project root
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git MSST

# MSST needs its own venv for some optional tools, but Utai uses the main venv
# for inference. No separate MSST setup is needed — Utai imports MSST modules
# directly through the main venv.

# Separation models are downloaded on-demand through the in-app Model Manager.
# They are stored in MSST/pretrain/{category}/ and gitignored.
```

#### RVC (Retrieval-based Voice Conversion)

```bash
# Clone or extract RVC into the project root
# The expected structure is RVC/{version_dir}/ (e.g. RVC/RVC20240604Nvidia/)
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git RVC/RVC20240604Nvidia

# Download pretrained base models (required for training):
# Place in RVC/RVC20240604Nvidia/assets/pretrained_v2/
#   - f0G32k.pth, f0G40k.pth, f0G48k.pth (Generator with F0)
#   - f0D32k.pth, f0D40k.pth, f0D48k.pth (Discriminator with F0)
#   - G32k.pth, G40k.pth, G48k.pth (Generator without F0)
#   - D32k.pth, D40k.pth, D48k.pth (Discriminator without F0)
#
# Download HuBERT model:
# Place in RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt
#
# Download RMVPE model:
# Place in RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt
#
# These can be found at:
# https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main
```

> **Note**: Utai applies several patches to RVC scripts for compatibility:
> - `train.py` — Single-GPU direct execution (no mp.Process), TensorBoard stub, exit code 2333333 handling
> - `utils.py` — StreamHandler added for real-time log output
> - `i18n/` — Minimal stub module (RVC GUI i18n not needed)
> - `configs/v2/40k.json` — Missing config file recreated

#### yt-dlp

```bash
# Clone yt-dlp source
git clone https://github.com/yt-dlp/yt-dlp.git yt-dlp
```

#### UtaiRenderer (Singing Synthesis)

```bash
# Requires .NET 8 SDK
# Clone the OpenUtau fork with DiffSinger support
cd synth
git clone --depth 1 https://github.com/keirokeer/OpenUtau-DiffSinger-Lunai.git

# Apply access modifier patches (internal → public) to OpenUtau.Core:
#   RenderEngine.cs, ExportAdapter.cs, MasterAdapter.cs, PhonemizerRunner.cs,
#   DocManager.cs, UPart.cs, UNote.cs — see synth/utai-renderer-src/ for details

# Build the self-contained renderer
cd utai-renderer-src
dotnet publish -c Release -r win-x64 --self-contained true -o "../utai-renderer"

# Download DiffSinger voice banks into synth/utai-renderer/Singers/
# Available at: https://github.com/lunaiproject/lunai_singers/releases
```

#### ffmpeg

```bash
# Download ffmpeg and place executables in bin/
# https://www.gyan.dev/ffmpeg/builds/ (Windows builds)
# Or: https://ffmpeg.org/download.html
mkdir bin
# Place ffmpeg.exe and ffprobe.exe in bin/
# Also place a copy of ffmpeg.exe in RVC/RVC20240604Nvidia/ (RVC needs it locally)
```

> **Note**: All `ffmpeg.exe` and `ffprobe.exe` files are gitignored (via `**/ffmpeg.exe` pattern).
> You must download and place them manually after cloning.

### 3. Frontend

```bash
cd app/frontend
npm install
npm run build          # Production build -> dist/
# Copy to backend static:
cp -r dist/* ../backend/static/
```

### 3. Development Server

```bash
# Option A: Use run.bat (starts backend with hot-reload)
run.bat
# Then open http://localhost:8765

# Option B: Docker for frontend dev + local backend
docker compose -f docker-compose.dev.yml up frontend-dev
# In another terminal:
venv\Scripts\activate
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8765 --reload
```

### 4. Electron Development

```bash
cd electron
npm install

# Dev mode (connects to running backend):
npm start

# Build portable app:
# (First build frontend and copy to backend/static/)
rebuild_utai.bat
```

---

## Building the Portable App

The `Utai/` directory contains the fully portable application — no Python or Node.js installation required on the target machine.

### Build Process

```bash
# 1. Build frontend
cd app/frontend && npm run build

# 2. Copy to backend/static
xcopy /e /y dist\* ..\backend\static\

# 3. Run the rebuild script (syncs everything to Utai/)
cd ../..
rebuild_utai.bat
```

### Portable Architecture

The portable build replaces Python's venv mechanism with `python310._pth` (embedded Python mode):

- `venv/Scripts/python.exe` — Real Python executable (not venv launcher)
- `venv/Scripts/python310._pth` — Controls module search paths
- `venv/Lib/` — Python standard library
- `venv/Lib/site-packages/` — All pip packages including PyTorch
- `venv/DLLs/` — Python C extension modules
- `venv/Lib/site-packages/sitecustomize.py` — Auto-patches for:
  - `subprocess.Popen` → `CREATE_NO_WINDOW` (suppress console windows)
  - `multiprocessing.spawn.is_forking` → fix for `_pth` isolated mode
  - `_winapi.CreateProcess` → `CREATE_NO_WINDOW` for worker processes

### Package Size

| Component | Size |
|-----------|------|
| Python venv (PyTorch + all deps) | ~5.5 GB |
| RVC infrastructure (pretrained models) | ~3.1 GB |
| Electron runtime | ~200 MB |
| MSST code + configs | ~2 MB |
| Frontend + Backend | ~2 MB |
| ffmpeg + yt-dlp | ~63 MB |
| **Total** | **~9 GB** |

---

## Node Types

### Workflow Page (Voice Conversion)

| Node | Category | Inputs | Outputs | Description |
|------|----------|--------|---------|-------------|
| Audio Input | Input | — | audio | Upload audio files (drag & drop) |
| Video Input | Input | — | video_path | Video file reference |
| yt-dlp Download | Input | — | audio, video_path | Download from URL |
| Separation | Separation | audio | dynamic stems | MSST/VR model separation (113 models) |
| RVC Conversion | Conversion | audio | audio | Voice timbre conversion |
| Audio Merge | Processing | audio_1..8 | audio | Mix multiple tracks |
| Media Output | Output | audio, video_path | — | Export with ffmpeg |

### Training Page (RVC Model Training)

| Node | Description |
|------|-------------|
| Audio Import | Upload training dataset (drag & drop) |
| Audio Separation | Batch MSST separation for dataset preparation |
| RVC Train | Full 5-step training pipeline |

### RVC Training Pipeline

```
Step 1: Preprocessing     — Audio slicing & resampling
Step 2: F0 Extraction     — Pitch detection (RMVPE/Harvest/Crepe/PM)
Step 3: Feature Extraction — HuBERT feature extraction
Step 4: Model Training    — GAN training with pretrained G/D
Step 5: Index Building    — FAISS index for voice retrieval
```

---

## Synth Page (DAW)

The `/synth` page is a built-in DAW for AI singing voice synthesis. It combines a DiffSinger-based vocal renderer with RVC timbre conversion, wrapped in a multi-track arrangement environment.

### Render Pipeline

```
DiffSinger Voice Bank  →  UtaiRenderer.exe  →  Raw Vocal WAV
                                                      ↓
                              RVC Voice Conversion  ←──┘
                                      ↓
                              Final Vocal WAV  →  Web Audio Playback
```

The backend orchestrates: **UtaiRenderer** (C#/.NET 8, wraps OpenUtau Core) synthesizes DiffSinger vocals from the USTX-format project, then the RVC wrapper applies timbre conversion using the selected voice model.

### Piano Roll Editor

- Canvas 2D rendering engine with retro dark theme
- Note creation, selection, drag, resize, and deletion
- Lyric editing per note (click to edit inline)
- Tool modes: Select, Draw, Erase (keyboard: `1`/`2`/`3`)
- Snap grid with 11 options (1/1 to 1/32 triplet), Alt+wheel to cycle
- Ctrl+wheel horizontal zoom, Shift+wheel horizontal pan
- Undo/Redo (Ctrl+Z/Y, max 50 history)

### Arrangement View

- Multi-track overview with canvas-based rendering
- Track types: Vocal, Audio (imported WAV), Instrument (VST plugin)
- Vocal parts shown as colored blocks with mini note visualization
- Wave parts with waveform thumbnails from peak data
- Instrument parts with note density preview
- Double-click a part to open in piano roll
- Right-click context menu: create, split (at playhead), delete parts
- Ctrl+wheel horizontal zoom, Alt+wheel vertical track height zoom (40–128px)
- Track header: mute/solo buttons, volume bar

### Track Mixer

- Per-track volume (–24 to +6 dB) and pan (L100–R100) with visible drag thumbs
- Mute (M) and Solo (S) buttons per track
- Mixer panel adapts to track height changes
- Pan row auto-hides when tracks are zoomed small

### Transport Bar

- Play / Pause / Stop controls (Space = toggle, Home = stop)
- Position display: Bar:Beat:Tick + MM:SS.ms
- BPM control and time signature
- Loop toggle with range
- WaveSurfer.js waveform strip with click-to-seek

### Audio Effects

Built-in Web Audio effects applied per-track:

| Effect | Parameters |
|--------|-----------|
| Gain | Volume (dB) |
| 3-Band EQ | Low/Mid/High gain, crossover frequencies |
| Reverb | Wet/dry mix, decay, pre-delay (convolution-based) |
| Compressor | Threshold, knee, ratio, attack, release |

### VST Plugin Support

- JUCE-based VST2/VST3 host for instrument tracks
- Plugin editor windows for parameter control
- MIDI note playback from arrangement

### Keyboard Shortcuts (Synth Page)

| Shortcut | Action |
|----------|--------|
| Space | Play / Pause |
| Home | Stop, go to start |
| Ctrl+S | Render (compile) |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| 1 / 2 / 3 | Select / Draw / Erase tool |
| Delete / Backspace | Delete selected notes |
| Ctrl+A | Select all notes |
| Ctrl+Wheel | Horizontal zoom |
| Alt+Wheel | Track height zoom (arrangement) / Snap grid cycle (piano roll) |
| Shift+Wheel | Horizontal scroll |

---

## Execution Engine

The workflow executor implements:

- **Topological sort** (Kahn's algorithm) for execution order
- **Partial execution** — re-run from any node, reusing cached upstream results
- **Single-node execution** — run one node without triggering downstream
- **Independent failure** — failed nodes don't block other branches
- **Real-time progress** — WebSocket streaming of per-node status
- **Force stop** — `asyncio.Task.cancel()` + subprocess tree kill
- **Cycle detection** — frontend prevents circular connections

---

## Model Sources

| Source | Models | Type |
|--------|--------|------|
| [Sucial/MSST-WebUI](https://huggingface.co/Sucial/MSST-WebUI) | 78 | Official MSST models |
| [Eddycrack864/audio-separator-models](https://huggingface.co/Eddycrack864/audio-separator-models) | 35 | Community models (marked with *) |

Models are downloaded on-demand through the built-in Model Manager.

---

## Project File Formats

### `.utai` — Workflow Project

```
project.utai (ZIP)
├── workflow.json              # Node graph + parameters
├── buffers/
│   ├── registry.json          # Audio buffer metadata
│   └── <buffer_id>.wav        # Embedded audio files
```

### `.utaiTr` — Training Project

```
project.utaiTr (ZIP)
├── workflow.json              # Training node graph
├── training_audio/
│   └── <node_id>/
│       └── <filename>.wav     # Training dataset audio
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8765` | Backend server port |
| `HOST` | `127.0.0.1` | Backend bind address |
| `DEVICE` | `auto` | Compute device (`auto`, `cuda`, `cpu`) |
| `MAX_CACHED_MODELS` | `2` | Max ML models kept in memory |
| `UTAI_DEV` | — | Set to enable dev mode (Vite proxy) |

### Portable Mode Patches (`sitecustomize.py`)

The portable build auto-patches Python to:
1. Suppress all console windows (`CREATE_NO_WINDOW` on all subprocess calls)
2. Fix `multiprocessing.spawn` in isolated mode (`_pth` file compatibility)
3. Suppress console windows for multiprocessing workers

---

## Known Limitations

- Windows only (macOS/Linux not yet supported)
- TensorBoard is stubbed out (training charts are built into the UI instead)
- Models marked with `*` use community-hosted download sources
- Unsigned executable triggers Windows SmartScreen warning
- UtaiRenderer requires .NET 8 runtime (bundled in self-contained publish)
- DiffSinger voice banks must be manually downloaded and placed in `synth/utai-renderer/Singers/`
- VST plugin support is Windows-only and requires plugin files installed separately

---

## Credits & Acknowledgments

- [MSST (Music Source Separation Training)](https://github.com/ZFTurbo/Music-Source-Separation-Training) — Audio separation framework
- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — Voice conversion & training
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Media downloader
- [ffmpeg](https://ffmpeg.org/) — Audio/video processing
- [OpenUtau](https://github.com/stakira/OpenUtau) — Singing synthesis engine (Core library)
- [DiffSinger](https://github.com/openvpi/DiffSinger) — Neural singing voice synthesis
- [JUCE](https://juce.com/) — VST2/VST3 plugin host framework
- [React Flow](https://reactflow.dev/) — Node-based UI framework
- [WaveSurfer.js](https://wavesurfer.xyz/) — Audio waveform visualization
- Model authors: aufr33, anvuew, Gabox, kimberley, SYHFT, pcunwa, and others

---

<p align="center">
  <sub>Built with FastAPI, React Flow, PyTorch, and a love for retro UI design</sub><br>
  <sub>「Utai」v0.1.0</sub>
</p>
