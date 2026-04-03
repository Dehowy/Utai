"""Training API routes — RVC training pipeline with WebSocket log streaming."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from pydantic import BaseModel

from app.backend.config import TEMP_DIR, RVC_ROOT
from app.backend.wrappers.rvc_train_wrapper import rvc_train_wrapper, TrainingTask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/train", tags=["training"])

# Store active training tasks
_active_tasks: dict[str, TrainingTask] = {}
_task_queues: dict[str, list[asyncio.Queue]] = {}


# --- Request / Response models ---

from fastapi.responses import FileResponse


# --- Dataset audio streaming ---

@router.get("/dataset-audio")
async def stream_dataset_audio(path: str):
    """Stream an audio file from the dataset directory for preview playback."""
    file_path = Path(path)
    # Security: only allow files within TEMP_DIR/datasets
    datasets_root = Path(str(TEMP_DIR)) / "datasets"
    try:
        file_path.resolve().relative_to(datasets_root.resolve())
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(403, "Access denied")

    if not file_path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, "File not found")

    return FileResponse(str(file_path), media_type="audio/wav")


@router.get("/training-metrics/{exp_name}")
async def get_training_metrics(exp_name: str):
    """Parse training logs to extract loss metrics for visualization."""
    from fastapi import HTTPException
    import re

    exp_dir = Path(RVC_ROOT) / "logs" / exp_name
    if not exp_dir.exists():
        raise HTTPException(404, f"Experiment not found: {exp_name}")

    # Collect available weight files
    weights_dir = Path(RVC_ROOT) / "assets" / "weights"
    weight_files = sorted(weights_dir.glob(f"{exp_name}_e*_s*.pth"))
    checkpoints = []
    for wf in weight_files:
        m = re.match(rf"{re.escape(exp_name)}_e(\d+)_s(\d+)\.pth", wf.name)
        if m:
            checkpoints.append({"epoch": int(m.group(1)), "step": int(m.group(2)), "file": wf.name})

    return {"exp_name": exp_name, "checkpoints": checkpoints}


class BrowseFolderRequest(BaseModel):
    initial_dir: str = ""


# --- Folder picker (native dialog) ---

@router.post("/browse-folder")
async def browse_folder(req: BrowseFolderRequest):
    """Open a native folder picker dialog and return the selected path."""
    import asyncio

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initial = req.initial_dir if req.initial_dir and os.path.isdir(req.initial_dir) else None
            folder = filedialog.askdirectory(
                title="Select folder",
                initialdir=initial,
            )
            root.destroy()
            return folder or ""
        except Exception:
            return ""

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _pick)
    return {"path": path}


class BatchSeparateRequest(BaseModel):
    model_name: str
    input_dir: str
    files: list[str]  # file paths
    output_dir: str = ""  # if empty, auto-create under temp
    use_tta: bool = False
    # VR-specific
    window_size: int = 512
    aggression: int = 5
    post_process: bool = False


@router.post("/batch-separate")
async def start_batch_separation(req: BatchSeparateRequest):
    """Start batch MSST/VR separation on disk files. Returns task_id for WS progress."""
    if not req.files:
        from fastapi import HTTPException
        raise HTTPException(400, "No files provided")

    task_id = uuid.uuid4().hex[:16]
    task = TrainingTask(task_id, f"batch_sep_{req.model_name}")
    _active_tasks[task_id] = task
    _task_queues[task_id] = []

    asyncio.create_task(_run_batch_separation(task_id, req))
    return {"task_id": task_id, "status": "started"}


async def _run_batch_separation(task_id: str, req: BatchSeparateRequest):
    """Run batch separation in background."""
    import json as _json
    import soundfile as sf
    import numpy as np

    task = _active_tasks[task_id]
    log_cb = await _log_callback_factory(task_id)

    logger.info(f"[batch-sep] Task {task_id} started for model {req.model_name}")

    try:
        # Log device info
        from app.backend.config import MSST_ROOT, get_device
        device = get_device()
        await log_cb(f"[batch-sep] Device: {device.upper()}")
        if device == "cuda":
            import torch
            await log_cb(f"[batch-sep] GPU: {torch.cuda.get_device_name(0)}")
        elif device == "cpu":
            await log_cb("[batch-sep] WARNING: Running on CPU — this will be slow!")

        # Determine model class
        info_path = Path(MSST_ROOT) / "data" / "models_info.json"
        models_info = {}
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                models_info = _json.load(f)

        model_info = models_info.get(req.model_name, {})
        model_class = model_info.get("model_class", "")
        is_vr = model_class == "VR_Models"

        # Output directory
        out_dir = Path(req.output_dir) if req.output_dir else Path(str(TEMP_DIR)) / "datasets" / f"separated_{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(req.files)
        stem_files: dict[str, list[str]] = {}

        for i, file_path in enumerate(req.files):
            # Check cancellation
            if task._cancelled:
                await log_cb("[batch-sep] Cancelled by user")
                task.status = "cancelled"
                await _broadcast(task_id, {
                    "type": "complete",
                    "success": False,
                    "status": "cancelled",
                })
                return

            fp = Path(file_path)
            if not fp.exists():
                await log_cb(f"[batch-sep] Skipping missing file: {fp.name}")
                continue

            await log_cb(f"[batch-sep] ({i+1}/{total}) Separating {fp.name}...")
            await _broadcast(task_id, {
                "type": "status",
                "step": f"Separating {fp.name}",
                "progress": i / total * 100,
            })

            # Load audio
            audio, sr = sf.read(str(fp), dtype="float32")

            # Run separation
            if is_vr:
                from app.backend.wrappers.vr_wrapper import vr_wrapper
                results = await vr_wrapper.separate(
                    audio, sr, req.model_name,
                    params={
                        "use_tta": req.use_tta,
                        "window_size": req.window_size,
                        "aggression": req.aggression,
                        "post_process": req.post_process,
                    },
                )
            else:
                from app.backend.wrappers.msst_wrapper import msst_wrapper
                results = await msst_wrapper.separate(
                    audio, sr, req.model_name,
                    use_tta=req.use_tta,
                )

            # Save each stem
            stem_name_base = fp.stem
            for stem_name, (stem_audio, stem_sr) in results.items():
                stem_dir = out_dir / stem_name
                stem_dir.mkdir(parents=True, exist_ok=True)
                out_path = stem_dir / f"{stem_name_base}_{stem_name}.wav"
                sf.write(str(out_path), stem_audio, stem_sr)

                if stem_name not in stem_files:
                    stem_files[stem_name] = []
                stem_files[stem_name].append(str(out_path))

            await log_cb(f"[batch-sep] ({i+1}/{total}) Done: {list(results.keys())}")

        task.status = "completed"
        await _broadcast(task_id, {
            "type": "status",
            "step": "Complete",
            "progress": 100,
        })
        await _broadcast(task_id, {
            "type": "complete",
            "success": True,
            "output_dir": str(out_dir),
            "stem_files": stem_files,
            "stems": list(stem_files.keys()),
        })

    except Exception as e:
        logger.exception("Batch separation failed")
        task.status = "error"
        await log_cb(f"[batch-sep] Error: {e}")
        await _broadcast(task_id, {
            "type": "complete",
            "success": False,
            "status": "error",
        })


class TrainStartRequest(BaseModel):
    exp_name: str
    input_dir: str  # Path to dataset directory
    files: list[str] = []  # Explicit file paths (when set, overrides input_dir scan)
    sr: str = "40k"  # 32k, 40k, 48k
    version: str = "v2"  # v1, v2
    f0: bool = True  # Use pitch guidance
    f0_method: str = "rmvpe"
    batch_size: int = 4
    total_epoch: int = 100
    save_epoch: int = 10
    gpu_ids: str = "0"
    pretrained_g: str = ""
    pretrained_d: str = ""
    save_latest: bool = True
    cache_gpu: bool = False
    save_weights: bool = True
    export_dir: str = ""


class TrainStepRequest(BaseModel):
    exp_name: str
    step: str  # preprocess, f0, features, train, index
    # Step-specific params
    input_dir: str = ""
    sr: str = "40k"
    version: str = "v2"
    f0_method: str = "rmvpe"
    f0: bool = True
    batch_size: int = 4
    total_epoch: int = 100
    save_epoch: int = 10
    gpu_ids: str = "0"
    pretrained_g: str = ""
    pretrained_d: str = ""
    save_latest: bool = True
    cache_gpu: bool = False
    save_weights: bool = True
    export_dir: str = ""


class TrainResponse(BaseModel):
    task_id: str
    status: str


# --- Dataset management ---

@router.post("/upload-dataset")
async def upload_dataset(
    files: list[UploadFile] = File(...),
    exp_name: str = Form(...),
):
    """Upload audio files for a training dataset."""
    dataset_dir = Path(str(TEMP_DIR)) / "datasets" / exp_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = dataset_dir / f.filename
        with open(str(dest), "wb") as fp:
            content = await f.read()
            fp.write(content)
        saved.append(str(dest))

    return {
        "dataset_dir": str(dataset_dir),
        "files": saved,
        "count": len(saved),
    }


@router.get("/dataset-dir")
async def get_dataset_dir(exp_name: str):
    """Get the dataset directory path for an experiment."""
    dataset_dir = Path(str(TEMP_DIR)) / "datasets" / exp_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [f.name for f in dataset_dir.iterdir() if f.suffix.lower() in audio_exts]

    return {
        "dataset_dir": str(dataset_dir),
        "files": files,
        "count": len(files),
    }


# --- Training pipeline ---

@router.post("/start", response_model=TrainResponse)
async def start_full_training(req: TrainStartRequest):
    """Start the full RVC training pipeline (all 5 steps)."""
    # Validate input_dir exists — but only if no explicit files are provided
    # (when files are given, _run_training creates its own flat dir from them)
    has_explicit_files = bool(getattr(req, 'files', None))
    if req.input_dir and not has_explicit_files and not os.path.isdir(req.input_dir):
        from fastapi import HTTPException
        raise HTTPException(400, f"input_dir does not exist: {req.input_dir}")

    task_id = uuid.uuid4().hex[:16]
    task = TrainingTask(task_id, req.exp_name)
    _active_tasks[task_id] = task
    _task_queues[task_id] = []

    # Start training in background (will wait for WS to connect)
    asyncio.create_task(_run_training(task_id, req.model_dump()))

    return TrainResponse(task_id=task_id, status="started")


@router.post("/step", response_model=TrainResponse)
async def start_training_step(req: TrainStepRequest):
    """Run a single training step."""
    task_id = uuid.uuid4().hex[:16]
    task = TrainingTask(task_id, req.exp_name)
    _active_tasks[task_id] = task
    _task_queues[task_id] = []

    asyncio.create_task(_run_single_step(task_id, req.model_dump()))

    return TrainResponse(task_id=task_id, status="started")


@router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """Cancel a running training task."""
    task = _active_tasks.get(task_id)
    if not task:
        return {"status": "not_found"}

    task.cancel()
    return {"status": "cancelling"}


@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """Get the status of a training task."""
    task = _active_tasks.get(task_id)
    if not task:
        return {"status": "not_found"}

    return {
        "task_id": task_id,
        "status": task.status,
        "current_step": task.current_step,
        "progress": task.progress,
        "log_count": len(task.logs),
    }


@router.get("/experiments")
async def list_experiments():
    """List all training experiments."""
    return rvc_train_wrapper.list_experiments()


@router.get("/pretrained")
async def list_pretrained():
    """List available pretrained models."""
    return rvc_train_wrapper.list_pretrained_models()


# --- WebSocket for log streaming ---

@router.websocket("/ws/{task_id}")
async def training_ws(ws: WebSocket, task_id: str):
    """WebSocket endpoint for streaming training logs."""
    await ws.accept()

    task = _active_tasks.get(task_id)
    if not task:
        await ws.send_json({"type": "error", "message": "Task not found"})
        await ws.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    if task_id not in _task_queues:
        _task_queues[task_id] = []
    _task_queues[task_id].append(queue)

    # Replay existing logs (sent before WS connected)
    for log in task.logs:
        await ws.send_json({"type": "log", "message": log})

    # Replay pending status/complete messages (sent before WS connected)
    pending = getattr(task, '_pending_msgs', [])
    already_complete = False
    for pmsg in pending:
        await ws.send_json(pmsg)
        if pmsg.get("type") == "complete":
            already_complete = True

    if not already_complete:
        # Send current status
        await ws.send_json({
            "type": "status",
            "status": task.status,
            "step": task.current_step,
            "progress": task.progress,
        })

    try:
        if already_complete:
            # Task already finished before WS connected, nothing more to wait for
            pass
        else:
            while True:
                msg = await queue.get()
                await ws.send_json(msg)
                if msg.get("type") == "complete":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in _task_queues:
            try:
                _task_queues[task_id].remove(queue)
            except ValueError:
                pass
            # Clean up task if no more listeners and task is done
            if not _task_queues[task_id]:
                task = _active_tasks.get(task_id)
                if task and task.status in ("completed", "error", "cancelled"):
                    _active_tasks.pop(task_id, None)
                    _task_queues.pop(task_id, None)


# --- Internal helpers ---

async def _broadcast(task_id: str, message: dict):
    """Broadcast a message to all connected WebSocket clients for a task."""
    # Persist log messages into task.logs so late-connecting WS clients
    # can replay them (fixes race between create_task and WS connect).
    task = _active_tasks.get(task_id)
    if task and message.get("type") == "log":
        task.logs.append(message["message"])
    # Also store last status + complete message for replay
    if task and message.get("type") in ("status", "complete"):
        if not hasattr(task, '_pending_msgs'):
            task._pending_msgs = []
        task._pending_msgs.append(message)
    queues = _task_queues.get(task_id, [])
    for q in queues:
        await q.put(message)


async def _log_callback_factory(task_id: str):
    """Create a log callback function for a task."""
    async def callback(message: str):
        await _broadcast(task_id, {"type": "log", "message": message})
    return callback


async def _run_training(task_id: str, params: dict):
    """Run the full training pipeline in background."""
    task = _active_tasks[task_id]
    log_cb = await _log_callback_factory(task_id)

    try:
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        # If explicit file paths are given, prepare a flat directory for preprocess.py
        explicit_files = params.get("files", [])
        if explicit_files:
            flat_dir = Path(str(TEMP_DIR)) / "datasets" / f"rvc_input_{task_id}"
            flat_dir.mkdir(parents=True, exist_ok=True)
            await log_cb(f"[prepare] Collecting {len(explicit_files)} files into flat dir")
            copied = 0
            for fp_str in explicit_files:
                fp = Path(fp_str)
                if not fp.exists():
                    await log_cb(f"[prepare] Skipping missing: {fp.name}")
                    continue
                dest = flat_dir / fp.name
                # Handle duplicate names by adding suffix
                if dest.exists():
                    dest = flat_dir / f"{fp.stem}_{copied}{fp.suffix}"
                shutil.copy2(str(fp), str(dest))
                copied += 1
            await log_cb(f"[prepare] Copied {copied} files")
            params["input_dir"] = str(flat_dir)
        else:
            # Fallback: if input_dir has subdirs but no direct audio files, flatten them
            input_path = Path(params.get("input_dir", ""))
            if input_path.is_dir():
                direct_audio = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in audio_exts]
                subdirs = [d for d in input_path.iterdir() if d.is_dir()]
                if not direct_audio and subdirs:
                    flat_dir = Path(str(TEMP_DIR)) / "datasets" / f"rvc_input_{task_id}"
                    flat_dir.mkdir(parents=True, exist_ok=True)
                    await log_cb(f"[prepare] Input dir has subdirectories only, flattening...")
                    copied = 0
                    for audio_file in input_path.rglob("*"):
                        if audio_file.is_file() and audio_file.suffix.lower() in audio_exts:
                            dest = flat_dir / audio_file.name
                            if dest.exists():
                                dest = flat_dir / f"{audio_file.stem}_{copied}{audio_file.suffix}"
                            shutil.copy2(str(audio_file), str(dest))
                            copied += 1
                    await log_cb(f"[prepare] Flattened {copied} audio files from subdirectories")
                    params["input_dir"] = str(flat_dir)

        success = await rvc_train_wrapper.run_full_pipeline(task, params, log_callback=log_cb)
        await _broadcast(task_id, {
            "type": "complete",
            "success": success,
            "status": task.status,
        })
    except Exception as e:
        logger.exception("Training pipeline failed")
        task.status = "error"
        await _broadcast(task_id, {
            "type": "log",
            "message": f"[fatal] {e}",
        })
        await _broadcast(task_id, {
            "type": "complete",
            "success": False,
            "status": "error",
        })
    finally:
        # Auto-cleanup temp dirs (rvc_input_, import_train_, separated_)
        datasets_dir = Path(str(TEMP_DIR)) / "datasets"
        if datasets_dir.exists():
            for d in datasets_dir.iterdir():
                if d.is_dir() and any(
                    d.name.startswith(p) for p in (f"rvc_input_{task_id}", "import_train_", "separated_")
                ):
                    try:
                        shutil.rmtree(str(d))
                        logger.info(f"Auto-cleaned temp dir: {d}")
                    except OSError:
                        pass


async def _run_single_step(task_id: str, params: dict):
    """Run a single training step."""
    task = _active_tasks[task_id]
    log_cb = await _log_callback_factory(task_id)
    step = params["step"]
    exp_name = params["exp_name"]

    sr_label = params.get("sr", "40k")
    sr_int = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sr_label, 40000)
    version = params.get("version", "v2")

    try:
        success = False

        if step == "preprocess":
            input_dir = params.get("input_dir", "")
            if not input_dir:
                await log_cb("[error] No input directory specified")
            else:
                success = await rvc_train_wrapper.preprocess(
                    task, exp_name, input_dir, sr_int, log_callback=log_cb,
                )

        elif step == "f0":
            success = await rvc_train_wrapper.extract_f0(
                task, exp_name, params.get("f0_method", "rmvpe"), log_callback=log_cb,
            )

        elif step == "features":
            success = await rvc_train_wrapper.extract_features(
                task, exp_name, version, log_callback=log_cb,
            )
            if success:
                success = await rvc_train_wrapper.write_filelist(
                    exp_name, sr_int, version, log_callback=log_cb,
                )

        elif step == "train":
            success = await rvc_train_wrapper.train(
                task, exp_name,
                sr=sr_label,
                f0=params.get("f0", True),
                batch_size=params.get("batch_size", 4),
                total_epoch=params.get("total_epoch", 100),
                save_epoch=params.get("save_epoch", 10),
                gpu_ids=params.get("gpu_ids", "0"),
                pretrained_g=params.get("pretrained_g", ""),
                pretrained_d=params.get("pretrained_d", ""),
                save_latest=params.get("save_latest", True),
                cache_gpu=params.get("cache_gpu", False),
                save_weights=params.get("save_weights", True),
                version=version,
                log_callback=log_cb,
            )

        elif step == "index":
            success = await rvc_train_wrapper.build_index(
                task, exp_name, version, log_callback=log_cb,
            )

        else:
            await log_cb(f"[error] Unknown step: {step}")

        task.status = "completed" if success else "error"
        await _broadcast(task_id, {
            "type": "complete",
            "success": success,
            "status": task.status,
        })

    except Exception as e:
        logger.exception(f"Training step {step} failed")
        task.status = "error"
        await _broadcast(task_id, {
            "type": "log",
            "message": f"[fatal] {e}",
        })
        await _broadcast(task_id, {
            "type": "complete",
            "success": False,
            "status": "error",
        })
