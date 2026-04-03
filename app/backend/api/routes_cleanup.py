"""Temp file and training data cleanup endpoints."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from fastapi import APIRouter

from app.backend.config import TEMP_DIR, RVC_ROOT
from app.backend.core.audio_buffer import buffer_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])


def _dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@router.get("/status")
async def cleanup_status():
    """Get current temp/training data sizes."""
    temp_size = _dir_size(TEMP_DIR)

    # Training intermediate data (RVC logs minus the .pth weights)
    rvc_logs = Path(RVC_ROOT) / "logs"
    training_size = 0
    training_experiments = []
    if rvc_logs.exists():
        for exp_dir in rvc_logs.iterdir():
            if exp_dir.is_dir():
                size = _dir_size(exp_dir)
                training_experiments.append({
                    "name": exp_dir.name,
                    "size": size,
                    "size_formatted": _format_size(size),
                })
                training_size += size

    # Audio buffers in memory
    buffer_count = len(buffer_manager.buffer_ids)

    return {
        "temp_dir": str(TEMP_DIR),
        "temp_size": temp_size,
        "temp_size_formatted": _format_size(temp_size),
        "training_size": training_size,
        "training_size_formatted": _format_size(training_size),
        "training_experiments": sorted(
            training_experiments, key=lambda x: -x["size"]
        ),
        "buffer_count": buffer_count,
    }


@router.post("/temp")
async def cleanup_temp():
    """Clean up temp files (downloads, separated outputs, uploaded datasets)."""
    freed = 0
    removed_dirs = []

    if not TEMP_DIR.exists():
        return {"freed": 0, "freed_formatted": "0 B", "removed": []}

    # Clean known temp subdirectories
    for subdir in TEMP_DIR.iterdir():
        if subdir.is_dir() and subdir.name in ("ytdlp_downloads", "video_output"):
            size = _dir_size(subdir)
            try:
                shutil.rmtree(str(subdir))
                freed += size
                removed_dirs.append(subdir.name)
            except OSError as e:
                logger.warning(f"Could not remove {subdir}: {e}")

    # Clean old dataset dirs (separated_, rvc_input_, import_train_)
    datasets_dir = TEMP_DIR / "datasets"
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir() and any(
                d.name.startswith(p)
                for p in ("separated_", "rvc_input_", "import_train_")
            ):
                size = _dir_size(d)
                try:
                    shutil.rmtree(str(d))
                    freed += size
                    removed_dirs.append(f"datasets/{d.name}")
                except OSError as e:
                    logger.warning(f"Could not remove {d}: {e}")

    # Clean stale temp files in TEMP_DIR root (older than 1 hour)
    one_hour_ago = time.time() - 3600
    for f in TEMP_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < one_hour_ago:
            size = f.stat().st_size
            try:
                f.unlink()
                freed += size
            except OSError:
                pass

    logger.info(f"Temp cleanup: freed {_format_size(freed)}, removed {removed_dirs}")
    return {
        "freed": freed,
        "freed_formatted": _format_size(freed),
        "removed": removed_dirs,
    }


@router.post("/buffers")
async def cleanup_buffers():
    """Clear all in-memory audio buffers."""
    count = len(buffer_manager.buffer_ids)
    buffer_manager.clear()
    logger.info(f"Cleared {count} audio buffers")
    return {"cleared": count}


@router.post("/buffers/remove")
async def remove_buffers(buffer_ids: list[str]):
    """Remove specific audio buffers by ID."""
    removed = 0
    for bid in buffer_ids:
        if bid in buffer_manager.buffer_ids:
            buffer_manager.remove(bid)
            removed += 1
    logger.info(f"Removed {removed}/{len(buffer_ids)} audio buffers")
    return {"removed": removed}


@router.post("/training/{exp_name}")
async def cleanup_training_data(exp_name: str):
    """Clean up intermediate training data for a specific experiment.

    Removes preprocessed audio, F0 data, and feature data.
    Keeps model weights (.pth) and index files (.index).
    """
    from fastapi import HTTPException

    exp_dir = Path(RVC_ROOT) / "logs" / exp_name
    if not exp_dir.exists():
        raise HTTPException(404, f"Experiment not found: {exp_name}")

    freed = 0
    # Intermediate dirs that can be safely removed
    intermediate_dirs = [
        "0_gt_wavs",
        "1_gt_wavs",
        "2a_f0",
        "2b-f0nsf",
        "3_feature256",
        "3_feature768",
    ]
    for dirname in intermediate_dirs:
        d = exp_dir / dirname
        if d.exists():
            size = _dir_size(d)
            shutil.rmtree(str(d))
            freed += size

    # Also clean .npy files in root of exp_dir
    for f in exp_dir.glob("*.npy"):
        freed += f.stat().st_size
        f.unlink()

    logger.info(f"Training cleanup [{exp_name}]: freed {_format_size(freed)}")
    return {
        "exp_name": exp_name,
        "freed": freed,
        "freed_formatted": _format_size(freed),
    }


@router.post("/training-all")
async def cleanup_all_training_data():
    """Clean up intermediate training data for ALL experiments."""
    rvc_logs = Path(RVC_ROOT) / "logs"
    if not rvc_logs.exists():
        return {"freed": 0, "freed_formatted": "0 B", "experiments": []}

    total_freed = 0
    experiments = []
    for exp_dir in rvc_logs.iterdir():
        if not exp_dir.is_dir():
            continue
        freed = 0
        intermediate_dirs = [
            "0_gt_wavs",
            "1_gt_wavs",
            "2a_f0",
            "2b-f0nsf",
            "3_feature256",
            "3_feature768",
        ]
        for dirname in intermediate_dirs:
            d = exp_dir / dirname
            if d.exists():
                size = _dir_size(d)
                shutil.rmtree(str(d))
                freed += size
        for f in exp_dir.glob("*.npy"):
            freed += f.stat().st_size
            f.unlink()
        if freed > 0:
            experiments.append({
                "name": exp_dir.name,
                "freed": freed,
                "freed_formatted": _format_size(freed),
            })
            total_freed += freed

    logger.info(f"Training cleanup all: freed {_format_size(total_freed)}")
    return {
        "freed": total_freed,
        "freed_formatted": _format_size(total_freed),
        "experiments": experiments,
    }
