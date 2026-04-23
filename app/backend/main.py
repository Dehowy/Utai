"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Windows requires ProactorEventLoop for asyncio.create_subprocess_exec
# uvicorn --reload may default to SelectorEventLoop which doesn't support subprocesses
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend.api.routes_audio import router as audio_router
from app.backend.api.routes_system import router as system_router
from app.backend.api.routes_workflow import router as workflow_router

# Optional routers — may fail if dependencies not installed
import logging as _logging
_opt_routers = []
for _name in ['routes_execute', 'routes_models', 'routes_cleanup', 'routes_train', 'routes_synth', 'routes_vst', 'routes_effects']:
    try:
        _mod = __import__(f'app.backend.api.{_name}', fromlist=['router'])
        _opt_routers.append(_mod.router)
    except Exception as _e:
        _logging.getLogger('utai.startup').warning(f'Skipped {_name}: {_e}')
from app.backend.config import TEMP_DIR
from app.backend.core.node_registry import register_all_nodes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Register all node types
register_all_nodes()

# Create app
app = FastAPI(
    title="utai",
    description="AI Voice Workstation — integrated audio workflow with MSST and RVC",
    version="0.1.0",
)

# CORS — allow any localhost port (Vite dev server, Electron backend on dynamic port)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_router)
app.include_router(system_router)
app.include_router(workflow_router)
for _r in _opt_routers:
    app.include_router(_r)

@app.on_event("startup")
async def startup_cleanup():
    """Clean leftover temp files from previous runs and check dependencies."""
    import shutil
    logger = logging.getLogger("utai.startup")

    # Check ffmpeg availability
    import subprocess, sys as _sys
    _cflags = 0x08000000 if _sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False, creationflags=_cflags)
    except FileNotFoundError:
        # Check project-local bin
        ffmpeg_local = Path(__file__).resolve().parent.parent.parent / "bin" / "ffmpeg.exe"
        if not ffmpeg_local.exists():
            logger.warning(
                "ffmpeg not found on PATH or in bin/. "
                "Video features (yt-dlp video output, audio extraction) may not work. "
                "Place ffmpeg.exe in the project bin/ folder."
            )
    if not TEMP_DIR.exists():
        return
    cleaned = []
    # Clean known temp subdirs
    for name in ("ytdlp_downloads", "video_output", "media_output"):
        d = TEMP_DIR / name
        if d.exists():
            shutil.rmtree(str(d), ignore_errors=True)
            cleaned.append(name)
    # Clean stale dataset temp dirs
    datasets_dir = TEMP_DIR / "datasets"
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir() and any(
                d.name.startswith(p) for p in ("separated_", "rvc_input_", "import_train_")
            ):
                shutil.rmtree(str(d), ignore_errors=True)
                cleaned.append(f"datasets/{d.name}")
    # Clean audio buffer temp files from previous sessions
    buffers_dir = TEMP_DIR / "audio_buffers"
    if buffers_dir.exists():
        shutil.rmtree(str(buffers_dir), ignore_errors=True)
        cleaned.append("audio_buffers")
    if cleaned:
        logger.info(f"Startup cleanup: removed {cleaned}")


# Serve static frontend (built React app) — only in Electron / production
# Requires UTAI_SERVE_STATIC=1 (set by Electron's main.js).
# In dev mode, Vite on a separate port handles the frontend.
# Direct backend access (http://localhost:8765) should NOT serve any UI.
if os.getenv("UTAI_SERVE_STATIC") == "1":
    _static_logger = logging.getLogger("utai.startup")
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and any(static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
        _static_logger.info("Static frontend mounted from %s", static_dir)
    else:
        _static_logger.error(
            "UTAI_SERVE_STATIC=1 but no static files at %s — frontend will not load!", static_dir
        )
