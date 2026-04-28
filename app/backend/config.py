"""Application configuration."""

import json
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

# VST paths
VST_ROOT = PROJECT_ROOT / "VST"
VST_INSTRUMENTS_DIR = VST_ROOT / "instruments"  # managed/imported plugins
VST_INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)

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
PORT = int(os.getenv("PORT", "8765"))

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


# ── Mirror / proxy config ──

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MIRRORS_FILE = CONFIG_DIR / "mirrors.json"

_DEFAULT_MIRRORS = {
    "pypi": "",
    "github": "",
    "huggingface": "",
    "pytorch": "",
}


def get_mirrors() -> dict:
    if MIRRORS_FILE.exists():
        try:
            return {**_DEFAULT_MIRRORS, **json.loads(MIRRORS_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(_DEFAULT_MIRRORS)


def set_mirrors(mirrors: dict) -> None:
    MIRRORS_FILE.write_text(json.dumps(mirrors, ensure_ascii=False, indent=2), encoding="utf-8")


def rewrite_url(url: str) -> str:
    """Rewrite a download URL using configured mirrors."""
    mirrors = get_mirrors()
    hf = mirrors.get("huggingface", "")
    gh = mirrors.get("github", "")
    if hf and "huggingface.co" in url:
        url = url.replace("https://huggingface.co", hf.rstrip("/"))
    if gh and "github.com" in url:
        url = url.replace("https://github.com", gh.rstrip("/"))
    return url
