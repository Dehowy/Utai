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
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
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


@router.get("/download-status/{model_name}")
async def get_download_status(model_name: str):
    """Get the download status for a model."""
    if model_name not in _active_downloads:
        return {"status": "not_found"}
    status = _active_downloads[model_name].copy()
    # Clean up completed/errored downloads
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

    return {
        "status": "imported",
        "model_name": Path(pth_file.filename).stem,
        "index_path": index_path,
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


# ──────────────────────────────────────────────
# DiffSinger voice bank management
# ──────────────────────────────────────────────

from app.backend.config import SYNTH_SINGERS_DIR, SYNTH_ROOT

_ds_catalog_path = Path(__file__).resolve().parent.parent / "data" / "diffsinger_models.json"
_ds_active_downloads: dict[str, dict] = {}


def _load_ds_catalog() -> dict:
    if not _ds_catalog_path.exists():
        return {"vocoders": [], "singers": []}
    with open(_ds_catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scan_installed_singers() -> list[dict]:
    """Scan synth/Singers/ for installed DiffSinger voice banks."""
    singers_dir = SYNTH_SINGERS_DIR
    if not singers_dir.exists():
        return []
    results = []
    for subdir in sorted(singers_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        char_yaml = subdir / "character.yaml"
        char_txt = subdir / "character.txt"
        if not (char_yaml.exists() or char_txt.exists()):
            # Also check configs/ subdirectory (flat layout)
            char_yaml = subdir / "configs" / "character.yaml"
            char_txt = subdir / "configs" / "character.txt"
        if not (char_yaml.exists() or char_txt.exists()):
            continue

        info: dict[str, Any] = {"id": subdir.name, "path": str(subdir)}

        # Parse character.yaml for metadata
        for yf in [subdir / "character.yaml", subdir / "configs" / "character.yaml"]:
            if yf.exists():
                try:
                    import yaml
                    with open(yf, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    info["name"] = data.get("name", subdir.name)
                    info["singer_type"] = data.get("singer_type", "diffsinger")
                    info["phonemizer"] = data.get("default_phonemizer", "")
                    # Check for subbanks (vocal modes)
                    if "subbanks" in data:
                        info["subbanks"] = [sb.get("suffix", "") for sb in data["subbanks"] if isinstance(sb, dict)]
                except Exception:
                    info["name"] = subdir.name
                break

        if "name" not in info:
            info["name"] = subdir.name

        # Check for essential files
        has_files = (subdir / "files").exists() or (subdir / "configs" / "dsconfig.yaml").exists()
        info["has_model_files"] = has_files

        # Calculate total size
        total_size = 0
        for f in subdir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        info["size_bytes"] = total_size

        results.append(info)
    return results


@router.get("/diffsinger")
async def list_diffsinger_models():
    """List installed DiffSinger voice banks."""
    return _scan_installed_singers()


@router.get("/diffsinger/catalog")
async def get_diffsinger_catalog():
    """Return the downloadable DiffSinger model catalog with install status."""
    catalog = _load_ds_catalog()
    installed_ids = {s["id"] for s in _scan_installed_singers()}

    # Add install status to each entry
    for item in catalog.get("vocoders", []) + catalog.get("singers", []):
        item["installed"] = item["id"] in installed_ids

    return catalog


class DSDownloadRequest(BaseModel):
    model_id: str


@router.post("/diffsinger/download")
async def start_diffsinger_download(req: DSDownloadRequest):
    """Start downloading a DiffSinger voice bank (zip → extract to Singers/)."""
    catalog = _load_ds_catalog()
    all_models = catalog.get("vocoders", []) + catalog.get("singers", [])
    entry = next((m for m in all_models if m["id"] == req.model_id), None)

    if not entry:
        raise HTTPException(404, f"Model not found in catalog: {req.model_id}")

    url = entry.get("url", "")
    if not url:
        raise HTTPException(400, f"No download URL for: {req.model_id}")

    if req.model_id in _ds_active_downloads:
        return {"status": "already_downloading", "model_id": req.model_id}

    _ds_active_downloads[req.model_id] = {
        "progress": 0,
        "total_size": entry.get("size_bytes", 0),
        "downloaded": 0,
        "status": "downloading",
        "error": None,
    }

    asyncio.create_task(_download_diffsinger(req.model_id, url, entry))
    return {"status": "started", "model_id": req.model_id}


async def _download_diffsinger(model_id: str, url: str, entry: dict):
    """Download and extract a DiffSinger voice bank zip."""
    import tempfile
    import zipfile

    singers_dir = SYNTH_SINGERS_DIR
    singers_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = None

    try:
        total_size = entry.get("size_bytes", 0)

        # Download zip to temp file
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".zip", prefix=f"ds_{model_id}_")
        tmp_path = Path(tmp_path_str)
        import os
        os.close(tmp_fd)

        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(600, connect=30)) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    _ds_active_downloads[model_id]["status"] = "error"
                    _ds_active_downloads[model_id]["error"] = f"HTTP {resp.status_code}"
                    return

                content_length = resp.headers.get("content-length")
                if content_length and not total_size:
                    total_size = int(content_length)

                downloaded = 0
                with open(str(tmp_path), "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size * 100) if total_size else 0
                        _ds_active_downloads[model_id].update({
                            "downloaded": downloaded,
                            "total_size": total_size,
                            "progress": min(progress, 95),
                        })

        # Extract zip
        _ds_active_downloads[model_id].update({"progress": 96, "status": "extracting"})

        dest_dir = singers_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(tmp_path), "r") as zf:
            # Check if zip has a single root folder
            names = zf.namelist()
            roots = set()
            for n in names:
                root = n.split("/")[0] if "/" in n else n
                if root:
                    roots.add(root)

            if len(roots) == 1 and all("/" in n for n in names if n):
                # Single root folder — extract contents directly
                single_root = roots.pop()
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    # Strip the root folder from the path
                    rel = member.filename[len(single_root) + 1:]
                    if not rel:
                        continue
                    target = dest_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(str(target), "wb") as dst:
                        shutil.copyfileobj(src, dst)
            else:
                # Multiple roots or flat — extract as-is
                zf.extractall(str(dest_dir))

        # Also copy to utai-renderer/Singers/ if renderer exists
        renderer_singers = SYNTH_ROOT / "utai-renderer" / "Singers" / model_id
        if (SYNTH_ROOT / "utai-renderer").exists():
            if renderer_singers.exists():
                shutil.rmtree(str(renderer_singers), ignore_errors=True)
            shutil.copytree(str(dest_dir), str(renderer_singers))

        _ds_active_downloads[model_id].update({"status": "completed", "progress": 100})
        logger.info(f"DiffSinger model downloaded and extracted: {model_id} -> {dest_dir}")

    except Exception as e:
        logger.error(f"DiffSinger download failed for {model_id}: {e}")
        _ds_active_downloads[model_id].update({"status": "error", "error": str(e)})
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.get("/diffsinger/download-status/{model_id}")
async def get_diffsinger_download_status(model_id: str):
    """Poll download progress for a DiffSinger voice bank."""
    status = _ds_active_downloads.get(model_id)
    if not status:
        return {"status": "not_found", "model_id": model_id}

    result = {**status, "model_id": model_id}
    if status["status"] in ("completed", "error"):
        _ds_active_downloads.pop(model_id, None)
    return result


@router.post("/diffsinger/import")
async def import_diffsinger_model(file: UploadFile = File(...)):
    """Import a DiffSinger voice bank zip file."""
    import tempfile
    import zipfile

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    singers_dir = SYNTH_SINGERS_DIR
    singers_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file to temp
    contents = await file.read()
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".zip", prefix="ds_import_")
    import os
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    tmp_path.write_bytes(contents)

    try:
        with zipfile.ZipFile(str(tmp_path), "r") as zf:
            names = zf.namelist()
            roots = set()
            for n in names:
                root = n.split("/")[0] if "/" in n else n
                if root:
                    roots.add(root)

            # Determine singer name from zip
            singer_name = file.filename.rsplit(".", 1)[0]
            if len(roots) == 1:
                singer_name = roots.pop()

            dest_dir = singers_dir / singer_name
            if dest_dir.exists():
                raise HTTPException(409, f"Singer '{singer_name}' already exists")

            dest_dir.mkdir(parents=True, exist_ok=True)

            if len(roots) == 1:
                # Single root — extract flattened
                root_prefix = singer_name + "/"
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    rel = member.filename
                    if rel.startswith(root_prefix):
                        rel = rel[len(root_prefix):]
                    if not rel:
                        continue
                    target = dest_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(str(target), "wb") as dst:
                        shutil.copyfileobj(src, dst)
            else:
                zf.extractall(str(dest_dir))

        # Copy to renderer
        renderer_singers = SYNTH_ROOT / "utai-renderer" / "Singers" / singer_name
        if (SYNTH_ROOT / "utai-renderer").exists():
            if renderer_singers.exists():
                shutil.rmtree(str(renderer_singers), ignore_errors=True)
            shutil.copytree(str(dest_dir), str(renderer_singers))

        logger.info(f"DiffSinger model imported: {singer_name}")
        return {"status": "imported", "singer_name": singer_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Import failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/diffsinger/archive/{singer_id}")
async def archive_diffsinger(singer_id: str):
    """Archive a DiffSinger voice bank."""
    singers_dir = SYNTH_SINGERS_DIR
    src = singers_dir / singer_id
    if not src.exists():
        raise HTTPException(404, f"Singer not found: {singer_id}")

    archive_dir = singers_dir / "_archived"
    archive_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"{singer_id}_{ts}"
    shutil.move(str(src), str(dest))

    # Also remove from renderer
    renderer_path = SYNTH_ROOT / "utai-renderer" / "Singers" / singer_id
    if renderer_path.exists():
        shutil.rmtree(str(renderer_path), ignore_errors=True)

    logger.info(f"DiffSinger archived: {singer_id} -> {dest.name}")
    return {"status": "archived", "archived_name": dest.name}


@router.get("/diffsinger/archived")
async def list_archived_diffsinger():
    """List archived DiffSinger voice banks."""
    archive_dir = SYNTH_SINGERS_DIR / "_archived"
    if not archive_dir.exists():
        return []
    results = []
    for d in sorted(archive_dir.iterdir()):
        if d.is_dir():
            total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            results.append({
                "name": d.name,
                "size_bytes": total_size,
            })
    return results


@router.post("/diffsinger/restore/{archived_name}")
async def restore_diffsinger(archived_name: str):
    """Restore an archived DiffSinger voice bank."""
    archive_dir = SYNTH_SINGERS_DIR / "_archived"
    src = archive_dir / archived_name
    if not src.exists():
        raise HTTPException(404, f"Archived singer not found: {archived_name}")

    # Strip timestamp to get original name
    parts = archived_name.rsplit("_", 2)
    original_name = parts[0] if len(parts) >= 3 else archived_name

    dest = SYNTH_SINGERS_DIR / original_name
    if dest.exists():
        raise HTTPException(409, f"Singer '{original_name}' already exists")

    shutil.move(str(src), str(dest))

    # Copy to renderer
    renderer_path = SYNTH_ROOT / "utai-renderer" / "Singers" / original_name
    if (SYNTH_ROOT / "utai-renderer").exists():
        if renderer_path.exists():
            shutil.rmtree(str(renderer_path), ignore_errors=True)
        shutil.copytree(str(dest), str(renderer_path))

    logger.info(f"DiffSinger restored: {archived_name} -> {original_name}")
    return {"status": "restored", "singer_name": original_name}


@router.delete("/diffsinger/{singer_id}")
async def delete_diffsinger(singer_id: str):
    """Permanently delete a DiffSinger voice bank."""
    src = SYNTH_SINGERS_DIR / singer_id
    if not src.exists():
        raise HTTPException(404, f"Singer not found: {singer_id}")
    shutil.rmtree(str(src))
    # Also remove from renderer
    renderer_path = SYNTH_ROOT / "utai-renderer" / "Singers" / singer_id
    if renderer_path.exists():
        shutil.rmtree(str(renderer_path), ignore_errors=True)
    logger.info(f"DiffSinger deleted: {singer_id}")
    return {"status": "deleted", "singer_id": singer_id}


@router.delete("/diffsinger/archived/{archived_name}")
async def delete_archived_diffsinger(archived_name: str):
    """Permanently delete an archived DiffSinger voice bank."""
    src = SYNTH_SINGERS_DIR / "_archived" / archived_name
    if not src.exists():
        raise HTTPException(404, f"Archived singer not found: {archived_name}")
    shutil.rmtree(str(src))
    logger.info(f"DiffSinger archived deleted: {archived_name}")
    return {"status": "deleted", "archived_name": archived_name}
