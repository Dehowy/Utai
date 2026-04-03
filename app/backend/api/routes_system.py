"""System info endpoints."""

from __future__ import annotations

from fastapi import APIRouter

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
