"""System info endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.backend.config import PROJECT_ROOT, MSST_ROOT, get_mirrors, set_mirrors
from app.backend.core.node_registry import NodeRegistry

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system/info")
async def system_info():
    """Get system information including GPU availability."""
    info = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_total": None,
        "gpu_memory_free": None,
        "device": "cpu",
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_free"] = torch.cuda.mem_get_info()[0]
            info["device"] = "cuda"
    except (ImportError, Exception):
        pass

    return info


@router.get("/nodes/definitions")
async def get_node_definitions():
    """Get all available node type definitions."""
    definitions = NodeRegistry.get_all_definitions()
    return [d.model_dump() for d in definitions]


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/system/modules")
async def installed_modules():
    """Detect which optional modules are installed."""
    modules = {"synth": False, "audio_tools": False}

    s2h_dist = PROJECT_ROOT / "synth" / "models" / "Jinguji_Suzumi.pt"
    s2h_dev = PROJECT_ROOT / "score2hubert_v2" / "archive" / "run9_final" / "best.pt"
    modules["synth"] = s2h_dist.exists() or s2h_dev.exists()

    msst_info = MSST_ROOT / "data" / "models_info.json"
    modules["audio_tools"] = msst_info.exists()

    try:
        import app.backend.wrappers.msst_wrapper  # noqa: F401
        modules["audio_tools_runtime"] = True
    except Exception:
        modules["audio_tools_runtime"] = False

    if modules["synth"]:
        modules["default_page"] = "/synth"
    elif modules["audio_tools"]:
        modules["default_page"] = "/"
    else:
        modules["default_page"] = "/"

    return modules


class MirrorsConfig(BaseModel):
    pypi: str = ""
    github: str = ""
    huggingface: str = ""
    pytorch: str = ""


@router.get("/system/mirrors")
async def get_mirrors_config():
    return get_mirrors()


@router.post("/system/mirrors")
async def set_mirrors_config(cfg: MirrorsConfig):
    set_mirrors(cfg.model_dump())
    return {"success": True}


@router.get("/system/mirrors/test")
async def test_mirror(url: str = ""):
    """Test if a URL is reachable. Returns latency in ms."""
    if not url:
        return {"reachable": False, "error": "Empty URL"}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.head(url, follow_redirects=True)
            return {"reachable": r.status_code < 400, "status": r.status_code, "latency_ms": r.elapsed.total_seconds() * 1000}
    except Exception as e:
        return {"reachable": False, "error": str(e)}
