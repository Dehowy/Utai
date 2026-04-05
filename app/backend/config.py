"""Application configuration."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Tool paths
MSST_ROOT = PROJECT_ROOT / "MSST"

# Auto-detect RVC subdirectory (e.g. RVC20240604Nvidia)
_rvc_base = PROJECT_ROOT / "RVC"
_rvc_candidates = [d for d in _rvc_base.iterdir() if d.is_dir() and not d.name.startswith(".")] if _rvc_base.exists() else []
RVC_ROOT = _rvc_candidates[0] if _rvc_candidates else _rvc_base / "RVC20240604Nvidia"

# App paths
APP_ROOT = PROJECT_ROOT / "app"
TEMP_DIR = APP_ROOT / "temp"
OUTPUT_DIR = APP_ROOT / "output"
WORKFLOWS_DIR = APP_ROOT / "workflows"
STATIC_DIR = APP_ROOT / "backend" / "static"

# Synth paths
SYNTH_ROOT = PROJECT_ROOT / "synth"
SYNTH_RENDERER_DIR = SYNTH_ROOT / "utai-renderer"
SYNTH_SINGERS_DIR = SYNTH_ROOT / "Singers"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8767"))

# Device
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"

# Model manager
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "2"))


def get_device() -> str:
    if DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return DEVICE
