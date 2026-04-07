"""VST instrument plugin management and rendering API."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.backend.wrappers import vst_wrapper
from app.backend.config import VST_INSTRUMENTS_DIR, TEMP_DIR

logger = logging.getLogger("utai.api.vst")
router = APIRouter(prefix="/api/vst", tags=["vst"])


# ── Models ──


class ImportRequest(BaseModel):
    path: str
    name: str | None = None


class RenderRequest(BaseModel):
    plugin_name: str
    notes: list[dict]  # [{tone, velocity, start, duration}, ...]
    bpm: float = 120
    duration: float | None = None
    sample_rate: int = 44100
    params: dict[int, float] | None = None


class ParamSetRequest(BaseModel):
    plugin_name: str
    index: int
    value: float


# ── Endpoints ──


@router.get("/list")
async def list_plugins():
    """List all imported VST instrument plugins."""
    return vst_wrapper.list_plugins()


@router.post("/import")
async def import_plugin(req: ImportRequest):
    """Import a VST plugin from a file path (validates first)."""
    try:
        meta = vst_wrapper.validate_and_import(req.path, req.name)
        return meta
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(422, f"Plugin validation failed: {e}")
    except Exception as e:
        logger.exception("Plugin import error")
        raise HTTPException(500, str(e))


@router.delete("/{name}")
async def delete_plugin(name: str):
    """Delete an imported plugin."""
    if vst_wrapper.delete_plugin(name):
        return {"ok": True}
    raise HTTPException(404, f"Plugin '{name}' not found")


@router.get("/params/{name}")
async def get_params(name: str):
    """Get parameter descriptions for a plugin."""
    try:
        return vst_wrapper.get_plugin_params(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Plugin '{name}' not found")
    except Exception as e:
        logger.exception("Error getting plugin params")
        raise HTTPException(500, str(e))


@router.post("/param")
async def set_param(req: ParamSetRequest):
    """Set a parameter value on a loaded plugin."""
    try:
        vst_wrapper.set_plugin_param(req.plugin_name, req.index, req.value)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/render")
async def render_vst(req: RenderRequest):
    """Render MIDI notes through a VST plugin, return WAV audio."""
    logger.info(f"VST render: plugin={req.plugin_name}, notes={len(req.notes)}, bpm={req.bpm}")
    try:
        wav_path = vst_wrapper.render_midi(
            name=req.plugin_name,
            notes=req.notes,
            bpm=req.bpm,
            duration_sec=req.duration,
            sample_rate=req.sample_rate,
            params=req.params,
        )
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=f"{req.plugin_name}_render.wav",
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("VST render error")
        raise HTTPException(500, str(e))


@router.post("/editor/open/{name}")
async def open_editor(name: str):
    """Open the VST plugin's GUI editor window (spawns subprocess)."""
    try:
        return vst_wrapper.open_editor(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("Editor open error")
        raise HTTPException(500, str(e))


@router.post("/editor/close/{name}")
async def close_editor(name: str):
    """Close a running editor window."""
    return vst_wrapper.close_editor(name)


@router.get("/editor/status/{name}")
async def editor_status(name: str):
    """Check if an editor is open."""
    return {"open": vst_wrapper.is_editor_open(name)}


@router.get("/scan")
async def scan_directory(path: str):
    """Scan a directory for VST plugin files (.dll, .vst3)."""
    d = Path(path)
    if not d.exists() or not d.is_dir():
        raise HTTPException(404, "Directory not found")
    plugins = []
    for f in sorted(d.rglob("*")):
        if f.suffix.lower() in (".dll", ".vst3") and f.is_file():
            plugins.append({
                "path": str(f),
                "name": f.stem,
                "size": f.stat().st_size,
            })
    return plugins
