"""Model listing, categories, and download endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.backend.config import MSST_ROOT
from app.backend.core.model_categories import (
    CATEGORIES,
    ARCHITECTURES,
    categorize_model,
    get_model_arch,
    get_model_arch_detail,
    get_model_description,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

_msst_root = Path(MSST_ROOT)


def _load_models_info() -> dict[str, Any]:
    info_path = _msst_root / "data" / "models_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _is_installed(info: dict[str, Any]) -> bool:
    target = info.get("target_position", "")
    if target:
        full_path = _msst_root / target.lstrip("./")
        return full_path.exists()
    return False


def _model_entry(filename: str, info: dict[str, Any]) -> dict[str, Any]:
    """Build a unified model entry with category."""
    model_class = info.get("model_class", "")
    target = info.get("target_position", "")
    full_path = _msst_root / target.lstrip("./") if target else None
    installed = full_path and full_path.exists()

    is_official = info.get("link", "").startswith("https://huggingface.co/Sucial/")

    return {
        "name": filename,
        "model_class": model_class,
        "category": categorize_model(filename, info),
        "arch": get_model_arch(filename, info),
        "arch_detail": get_model_arch_detail(filename, info),
        "installed": bool(installed),
        "path": str(full_path) if full_path else "",
        "size": info.get("model_size", 0),
        "primary_stem": info.get("primary_stem", ""),
        "secondary_stem": info.get("secondary_stem", ""),
        "link": info.get("link", ""),
        "is_official": is_official,
        "desc_zh": get_model_description(filename, "zh"),
        "desc_ja": get_model_description(filename, "ja"),
        "desc_en": get_model_description(filename, "en"),
    }


# ──── Existing endpoints (kept for backward compat) ────


@router.get("/msst")
async def list_msst_models():
    from app.backend.wrappers.msst_wrapper import msst_wrapper
    return msst_wrapper.list_models()


@router.get("/msst/stems")
async def get_msst_model_stems(model_name: str):
    from app.backend.wrappers.msst_wrapper import msst_wrapper
    stems = msst_wrapper.get_model_stems(model_name)
    return {"model_name": model_name, "stems": stems}


@router.get("/vr")
async def list_vr_models():
    from app.backend.wrappers.vr_wrapper import vr_wrapper
    return vr_wrapper.list_models()


@router.get("/rvc")
async def list_rvc_models():
    from app.backend.wrappers.rvc_wrapper import rvc_wrapper
    return rvc_wrapper.list_models()


# ──── New category-based endpoints ────


@router.get("/categories")
async def list_categories():
    """Return all categories with installed/total model counts."""
    models_info = _load_models_info()
    result = []
    for cat_id, cat_def in sorted(CATEGORIES.items(), key=lambda x: x[1]["order"]):
        installed = 0
        total = 0
        for filename, info in models_info.items():
            if categorize_model(filename, info) == cat_id:
                total += 1
                if _is_installed(info):
                    installed += 1
        result.append({
            "id": cat_id,
            "label_zh": cat_def["label_zh"],
            "label_ja": cat_def["label_ja"],
            "label_en": cat_def["label_en"],
            "color": cat_def["color"],
            "order": cat_def["order"],
            "installed_count": installed,
            "total_count": total,
        })
    return result


@router.get("/by-category/{category}")
async def list_models_by_category(category: str):
    """List all models (installed and not) belonging to a category."""
    if category not in CATEGORIES:
        raise HTTPException(400, f"Unknown category: {category}")

    models_info = _load_models_info()
    models = []
    for filename, info in models_info.items():
        if categorize_model(filename, info) == category:
            models.append(_model_entry(filename, info))

    # Sort: installed first, then by name
    models.sort(key=lambda m: (not m["installed"], m["name"]))
    return models


@router.get("/all")
async def list_all_models():
    """List all models with category info."""
    models_info = _load_models_info()
    return [_model_entry(fn, info) for fn, info in models_info.items()]


# ──── Download endpoint ────

_active_downloads: dict[str, dict] = {}
_downloads_lock = asyncio.Lock()


class DownloadRequest(BaseModel):
    model_name: str


@router.post("/download")
async def start_download(req: DownloadRequest):
    """Start downloading a model. Connect to /ws/download/{model_name} for progress."""
    models_info = _load_models_info()
    info = models_info.get(req.model_name)
    if not info:
        raise HTTPException(404, f"Model not found: {req.model_name}")

    link = info.get("link", "")
    if not link:
        raise HTTPException(400, f"No download link for model: {req.model_name}")

    target = info.get("target_position", "")
    if not target:
        raise HTTPException(400, f"No target path for model: {req.model_name}")

    dest = _msst_root / target.lstrip("./")
    if dest.exists():
        return {"status": "already_installed", "model_name": req.model_name}

    if req.model_name in _active_downloads:
        return {"status": "already_downloading", "model_name": req.model_name}

    # Start download in background
    _active_downloads[req.model_name] = {
        "progress": 0,
        "total_size": info.get("model_size", 0),
        "downloaded": 0,
        "status": "downloading",
        "error": None,
    }

    from app.backend.config import rewrite_url
    link = rewrite_url(link)
    asyncio.create_task(_download_model(req.model_name, link, dest, info))
    return {"status": "started", "model_name": req.model_name}


async def _download_model(model_name: str, url: str, dest: Path, info: dict):
    """Download a model file with progress tracking."""
    tmp_path = dest.parent / (dest.name + ".tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        total_size = info.get("model_size", 0)
        expected_sha = info.get("sha256", "")

        async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    _active_downloads[model_name]["status"] = "error"
                    _active_downloads[model_name]["error"] = f"HTTP {resp.status_code}"
                    return

                # Get total size from headers if not in info
                content_length = resp.headers.get("content-length")
                if content_length and not total_size:
                    total_size = int(content_length)

                downloaded = 0
                hasher = hashlib.sha256() if expected_sha else None

                with open(str(tmp_path), "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        downloaded += len(chunk)

                        progress = (downloaded / total_size * 100) if total_size else 0
                        _active_downloads[model_name].update({
                            "downloaded": downloaded,
                            "total_size": total_size,
                            "progress": min(progress, 99.9),
                        })

                # Verify SHA256 if available
                if hasher and expected_sha:
                    actual_sha = hasher.hexdigest()
                    if actual_sha != expected_sha:
                        tmp_path.unlink(missing_ok=True)
                        _active_downloads[model_name]["status"] = "error"
                        _active_downloads[model_name]["error"] = "SHA256 mismatch"
                        return

                # Move to final location (shutil.move handles cross-drive on Windows)
                shutil.move(str(tmp_path), str(dest))

        _active_downloads[model_name].update({
            "status": "completed",
            "progress": 100,
        })
        logger.info(f"Model downloaded: {model_name} -> {dest}")

    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        _active_downloads[model_name]["status"] = "error"
        _active_downloads[model_name]["error"] = str(e)
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


@router.get("/download-source-check")
async def check_download_sources():
    """Check if model download sources are reachable (with current mirror settings)."""
    from app.backend.config import rewrite_url
    import httpx
    test_urls = {
        "huggingface": rewrite_url("https://huggingface.co"),
        "github": rewrite_url("https://github.com"),
    }
    results = {}
    async with httpx.AsyncClient(timeout=8) as client:
        for name, url in test_urls.items():
            try:
                r = await client.head(url, follow_redirects=True)
                results[name] = {"reachable": r.status_code < 400, "url": url}
            except Exception as e:
                results[name] = {"reachable": False, "url": url, "error": str(e)}
    return results


@router.get("/download-status/{model_name}")
async def get_download_status(model_name: str):
    """Get the download status for a model."""
    async with _downloads_lock:
        if model_name not in _active_downloads:
            return {"status": "not_found"}
        status = _active_downloads[model_name].copy()
        if status["status"] in ("completed", "error"):
            del _active_downloads[model_name]
    return status


# ──── RVC model management ────

@router.post("/rvc/import")
async def import_rvc_model(
    pth_file: UploadFile = File(...),
    index_file: UploadFile | None = File(None),
):
    """Import a user's RVC .pth model (and optional .index file)."""
    from app.backend.config import RVC_ROOT

    if not pth_file.filename or not pth_file.filename.endswith(".pth"):
        raise HTTPException(400, "File must be a .pth file")

    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    dest = weights_dir / pth_file.filename
    if dest.exists():
        raise HTTPException(409, f"Model already exists: {pth_file.filename}")

    content = await pth_file.read()
    dest.write_bytes(content)
    logger.info(f"RVC model imported: {pth_file.filename}")

    index_path = None
    if index_file and index_file.filename and index_file.filename.endswith(".index"):
        indices_dir = Path(RVC_ROOT) / "assets" / "indices"
        indices_dir.mkdir(parents=True, exist_ok=True)
        idx_dest = indices_dir / index_file.filename
        idx_content = await index_file.read()
        idx_dest.write_bytes(idx_content)
        index_path = str(idx_dest)
        logger.info(f"RVC index imported: {index_file.filename}")

    model_stem = Path(pth_file.filename).stem

    async def _auto_detect():
        try:
            from app.backend.wrappers.s2h_wrapper import get_s2h_wrapper
            _range_tasks[model_stem] = {"status": "running", "progress": 0, "total": 61}

            def progress_cb(i, total, midi):
                _range_tasks[model_stem]["progress"] = i
                _range_tasks[model_stem]["total"] = total

            wrapper = get_s2h_wrapper()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: wrapper.detect_range(pth_file.filename, 60, progress_cb),
            )
            range_json = weights_dir / f"{model_stem}_range.json"
            range_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            _range_tasks[model_stem] = {"status": "done", "result": result}
            logger.info(f"Auto range detection done: {model_stem}")
        except Exception as e:
            logger.warning(f"Auto range detection failed: {e}")
            _range_tasks[model_stem] = {"status": "error", "error": str(e)}

    asyncio.create_task(_auto_detect())

    return {
        "status": "imported",
        "model_name": model_stem,
        "index_path": index_path,
        "range_detection": "started",
    }


@router.post("/rvc/archive/{model_name}")
async def archive_rvc_model(model_name: str):
    """Move an RVC model to the archive directory with a timestamp suffix."""
    from app.backend.config import RVC_ROOT

    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    model_path = weights_dir / f"{model_name}.pth"
    if not model_path.exists():
        raise HTTPException(404, f"RVC model not found: {model_name}")

    archive_dir = weights_dir / "_archived"
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_name = f"{model_name}_{ts}"
    archived_path = archive_dir / f"{archived_name}.pth"
    shutil.move(str(model_path), str(archived_path))
    logger.info(f"RVC model archived: {model_name} -> {archived_name}")

    # Also archive matching index files
    index_dirs = [
        Path(RVC_ROOT) / "assets" / "indices",
        Path(RVC_ROOT) / "logs",
    ]
    for idx_dir in index_dirs:
        if not idx_dir.exists():
            continue
        for idx_file in idx_dir.rglob("*.index"):
            if model_name.lower() in idx_file.stem.lower():
                idx_archive = archive_dir / f"{idx_file.stem}_{ts}{idx_file.suffix}"
                shutil.move(str(idx_file), str(idx_archive))
                logger.info(f"RVC index archived: {idx_file.name}")

    return {"status": "archived", "model_name": model_name, "archived_name": archived_name}


@router.get("/rvc/archived")
async def list_archived_rvc_models():
    """List archived RVC models."""
    from app.backend.config import RVC_ROOT

    archive_dir = Path(RVC_ROOT) / "assets" / "weights" / "_archived"
    if not archive_dir.exists():
        return []

    models = []
    for f in sorted(archive_dir.glob("*.pth")):
        index_count = len(list(archive_dir.glob(f"{f.stem}*.index")))
        models.append({
            "name": f.stem,
            "filename": f.name,
            "size": f.stat().st_size,
            "index_count": index_count,
        })
    return models


@router.post("/rvc/restore/{archived_name}")
async def restore_rvc_model(archived_name: str):
    """Restore an archived RVC model back to the weights directory."""
    from app.backend.config import RVC_ROOT
    import re

    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    archive_dir = weights_dir / "_archived"
    archived_path = archive_dir / f"{archived_name}.pth"
    if not archived_path.exists():
        raise HTTPException(404, f"Archived model not found: {archived_name}")

    # Strip timestamp suffix to get original name: model_name_YYYYMMDD_HHMMSS -> model_name
    original_name = re.sub(r'_\d{8}_\d{6}$', '', archived_name)
    dest = weights_dir / f"{original_name}.pth"

    # If original name already exists, keep the timestamped name
    if dest.exists():
        dest = weights_dir / f"{archived_name}.pth"

    shutil.move(str(archived_path), str(dest))
    restored_name = dest.stem
    logger.info(f"RVC model restored: {archived_name} -> {restored_name}")

    # Restore matching index files — find indices archived with the same timestamp
    # Archive format: {original_stem}_{YYYYMMDD}_{HHMMSS}.index
    ts_match = re.search(r'_(\d{8}_\d{6})$', archived_name)
    indices_dir = Path(RVC_ROOT) / "assets" / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    if ts_match:
        ts_suffix = ts_match.group(1)
        for idx_file in archive_dir.glob(f"*_{ts_suffix}.index"):
            idx_original = re.sub(r'_\d{8}_\d{6}$', '', idx_file.stem)
            idx_dest = indices_dir / f"{idx_original}{idx_file.suffix}"
            if idx_dest.exists():
                idx_dest = indices_dir / f"{idx_file.stem}{idx_file.suffix}"
            shutil.move(str(idx_file), str(idx_dest))
            logger.info(f"RVC index restored: {idx_file.name}")

    return {"status": "restored", "model_name": restored_name}


@router.delete("/rvc/archived/{archived_name}")
async def delete_archived_rvc_model(archived_name: str):
    """Permanently delete an archived RVC model."""
    from app.backend.config import RVC_ROOT

    archive_dir = Path(RVC_ROOT) / "assets" / "weights" / "_archived"
    archived_path = archive_dir / f"{archived_name}.pth"
    if not archived_path.exists():
        raise HTTPException(404, f"Archived model not found: {archived_name}")

    archived_path.unlink()
    # Also delete matching index files
    for idx_file in archive_dir.glob(f"{archived_name}*.index"):
        idx_file.unlink()
    logger.info(f"Archived model permanently deleted: {archived_name}")
    return {"status": "deleted", "archived_name": archived_name}


@router.delete("/rvc/{model_name}")
async def delete_rvc_model(model_name: str):
    """Delete an RVC voice model (.pth) and its associated index files."""
    from app.backend.config import RVC_ROOT

    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    model_path = weights_dir / f"{model_name}.pth"
    if not model_path.exists():
        raise HTTPException(404, f"RVC model not found: {model_name}")

    model_path.unlink()
    logger.info(f"RVC model deleted: {model_name}")
    return {"status": "deleted", "model_name": model_name}


@router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Delete an installed MSST/VR model."""
    models_info = _load_models_info()
    info = models_info.get(model_name)
    if not info:
        raise HTTPException(404, f"Model not found: {model_name}")

    target = info.get("target_position", "")
    if not target:
        raise HTTPException(400, "No target path")

    dest = _msst_root / target.lstrip("./")
    if not dest.exists():
        raise HTTPException(404, "Model file not found on disk")

    dest.unlink()
    logger.info(f"Model deleted: {model_name}")
    return {"status": "deleted", "model_name": model_name}


# ── RVC range detection ──────────────────────────────────────────────────

_range_tasks: dict[str, dict] = {}  # model_name → {status, progress, total, result}


@router.post("/rvc/detect-range")
async def detect_rvc_range(
    model_name: str = Form(...),
    reference_pitch: int = Form(60),
):
    """Start background range detection for an RVC model."""
    from app.backend.config import RVC_ROOT

    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    pth = model_name if model_name.endswith(".pth") else f"{model_name}.pth"
    if not (weights_dir / pth).exists():
        raise HTTPException(404, f"RVC model not found: {model_name}")

    clean_name = model_name.replace(".pth", "")
    if clean_name in _range_tasks and _range_tasks[clean_name].get("status") == "running":
        return {"status": "already_running", "model_name": clean_name}

    _range_tasks[clean_name] = {"status": "running", "progress": 0, "total": 61}

    async def _run():
        try:
            from app.backend.wrappers.s2h_wrapper import get_s2h_wrapper
            wrapper = get_s2h_wrapper()

            def progress_cb(i, total, midi):
                _range_tasks[clean_name]["progress"] = i
                _range_tasks[clean_name]["total"] = total
                _range_tasks[clean_name]["current_midi"] = midi

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: wrapper.detect_range(pth, reference_pitch, progress_cb),
            )

            range_json = weights_dir / f"{clean_name}_range.json"
            range_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

            _range_tasks[clean_name] = {"status": "done", "result": result}
            logger.info(f"Range detection saved: {range_json}")
        except Exception as e:
            logger.error(f"Range detection failed for {clean_name}: {e}", exc_info=True)
            _range_tasks[clean_name] = {"status": "error", "error": str(e)}

    asyncio.create_task(_run())
    return {"status": "started", "model_name": clean_name}


@router.get("/rvc/range-status/{model_name}")
async def get_rvc_range_status(model_name: str):
    """Get range detection progress or results."""
    clean_name = model_name.replace(".pth", "")

    if clean_name in _range_tasks:
        task = _range_tasks[clean_name]
        if task["status"] == "done":
            return {"status": "done", **task["result"]}
        return task

    from app.backend.config import RVC_ROOT
    range_json = Path(RVC_ROOT) / "assets" / "weights" / f"{clean_name}_range.json"
    if range_json.exists():
        data = json.loads(range_json.read_text(encoding="utf-8"))
        return {"status": "done", **data}

    return {"status": "not_started"}


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_name(m: int) -> str:
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


class RangeUpdate(BaseModel):
    comfort_low: int
    comfort_high: int


@router.post("/rvc/reset-range/{model_name}")
async def reset_rvc_range(model_name: str):
    """Reset comfort range to auto-detected values from per-semitone results."""
    from app.backend.config import RVC_ROOT

    clean_name = model_name.replace(".pth", "")
    range_json = Path(RVC_ROOT) / "assets" / "weights" / f"{clean_name}_range.json"
    if not range_json.exists():
        raise HTTPException(404, f"No range data for model: {clean_name}")

    data = json.loads(range_json.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise HTTPException(400, "No per-semitone results to recalculate from")

    comfort_midis = [r["midi"] for r in results if r.get("comfort")]
    if comfort_midis:
        data["comfort_low"] = min(comfort_midis)
        data["comfort_high"] = max(comfort_midis)
    else:
        usable_midis = [r["midi"] for r in results if r.get("usable")]
        if usable_midis:
            data["comfort_low"] = min(usable_midis)
            data["comfort_high"] = max(usable_midis)

    data["comfort_label"] = f"{_midi_name(data['comfort_low'])}-{_midi_name(data['comfort_high'])}"
    range_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Range reset for {clean_name}: comfort {data['comfort_label']}")
    return data


@router.get("/rvc/range/{model_name}")
async def get_rvc_range(model_name: str):
    """Get cached range data for an RVC model (if available)."""
    from app.backend.config import RVC_ROOT

    clean_name = model_name.replace(".pth", "")
    range_json = Path(RVC_ROOT) / "assets" / "weights" / f"{clean_name}_range.json"
    if range_json.exists():
        return json.loads(range_json.read_text(encoding="utf-8"))
    return None


@router.put("/rvc/range/{model_name}")
async def update_rvc_range(model_name: str, body: RangeUpdate):
    """Manually update comfort range for an RVC model."""
    from app.backend.config import RVC_ROOT

    clean_name = model_name.replace(".pth", "")
    range_json = Path(RVC_ROOT) / "assets" / "weights" / f"{clean_name}_range.json"
    if not range_json.exists():
        raise HTTPException(404, f"No range data for model: {clean_name}")

    data = json.loads(range_json.read_text(encoding="utf-8"))
    data["comfort_low"] = body.comfort_low
    data["comfort_high"] = body.comfort_high
    data["comfort_label"] = f"{_midi_name(body.comfort_low)}-{_midi_name(body.comfort_high)}"

    range_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Range updated for {clean_name}: comfort {data['comfort_label']}")
    return data
