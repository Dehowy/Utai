"""Workflow export/import endpoints — .utai project files."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response

from app.backend.schemas.workflow import Workflow
from app.backend.core.audio_buffer import AudioData, buffer_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflows"])


def _collect_buffer_entries(workflow: Workflow) -> dict[str, dict]:
    """Collect buffer_ids and their node metadata from workflow node params.

    Returns {buffer_id: {name, sample_rate, duration, ...}} for embedding in .utai.
    """
    entries: dict[str, dict] = {}
    for node in workflow.nodes:
        bid = node.params.get("buffer_id")
        if bid and isinstance(bid, str) and bid not in entries:
            entries[bid] = {
                "name": node.params.get("filename", bid),
                "sample_rate": node.params.get("sample_rate", 0),
                "duration": node.params.get("duration", 0),
            }
    return entries


def _convert_ytdlp_nodes_for_export(workflow: Workflow) -> Workflow:
    """Convert yt-dlp nodes to audio_input + video_input nodes for export.

    For each ytdlp_download node, creates:
    - An audio_input node (if the node had audio output)
    - A video_input node (if the node had video_path connections)
    Connections are remapped accordingly.
    """
    from app.backend.core.audio_buffer import AudioData, buffer_manager
    from app.backend.config import TEMP_DIR
    from app.backend.schemas.workflow import WorkflowNode, WorkflowConnection, Position

    ytdlp_temp = TEMP_DIR / "ytdlp_downloads"
    converted_nodes = []
    extra_nodes = []  # video_input nodes spawned alongside audio_input
    connection_remap: dict[tuple[str, str], tuple[str, str]] = {}  # (old_node, old_port) -> (new_node, new_port)

    for node in workflow.nodes:
        if node.type != "ytdlp_download":
            converted_nodes.append(node)
            continue

        # Check which ports have connections
        has_audio_conn = any(
            c.from_node == node.id and c.from_port == "audio"
            for c in workflow.connections
        )
        has_video_conn = any(
            c.from_node == node.id and c.from_port == "video_path"
            for c in workflow.connections
        )

        # --- Audio part ---
        bid = node.params.get("buffer_id")
        audio_converted = False
        if bid and buffer_manager.get(bid):
            new_params = {
                "buffer_id": bid,
                "filename": node.params.get("filename", node.params.get("url", "downloaded")),
                "sample_rate": node.params.get("sample_rate", 0),
                "duration": node.params.get("duration", 0),
            }
            converted_nodes.append(node.model_copy(update={
                "type": "audio_input",
                "params": new_params,
            }))
            audio_converted = True
        else:
            # Try to find downloaded audio on disk
            audio_file = None
            if ytdlp_temp.exists():
                for f in ytdlp_temp.iterdir():
                    if f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"):
                        audio_file = f
                        break
            if audio_file:
                try:
                    import soundfile as sf
                    samples, sr = sf.read(str(audio_file), dtype="float32")
                    audio = AudioData(samples=samples, sample_rate=sr, name=audio_file.name)
                    new_bid = buffer_manager.store(audio)
                    new_params = {
                        "buffer_id": new_bid,
                        "filename": audio_file.name,
                        "sample_rate": sr,
                        "duration": float(len(samples)) / sr if sr else 0,
                    }
                    converted_nodes.append(node.model_copy(update={
                        "type": "audio_input",
                        "params": new_params,
                    }))
                    audio_converted = True
                except Exception as e:
                    logger.warning(f"Failed to convert ytdlp audio for {node.id}: {e}")
                    converted_nodes.append(node)
            else:
                converted_nodes.append(node)

        # --- Video part: spawn a separate video_input node ---
        if audio_converted and has_video_conn:
            video_path = node.params.get("video_path", "")
            # Also check temp dir for video files
            if not video_path and ytdlp_temp.exists():
                for f in ytdlp_temp.iterdir():
                    if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
                        video_path = str(f)
                        break

            if video_path:
                video_node_id = f"{node.id}_video"
                video_node = WorkflowNode(
                    id=video_node_id,
                    type="video_input",
                    position=Position(x=node.position.x, y=node.position.y + 120),
                    params={
                        "video_path": video_path,
                        "video_name": Path(video_path).name if video_path else "",
                    },
                )
                extra_nodes.append(video_node)
                # Remap video_path connections from old node to new video_input node
                connection_remap[(node.id, "video_path")] = (video_node_id, "video_path")

    all_nodes = converted_nodes + extra_nodes

    # Remap connections
    new_connections = []
    converted_audio_ids = {n.id for n in converted_nodes if n.type == "audio_input"}
    ytdlp_ids = {n.id for n in workflow.nodes if n.type == "ytdlp_download"}
    converted_ytdlp_ids = converted_audio_ids & ytdlp_ids

    for c in workflow.connections:
        key = (c.from_node, c.from_port)
        if key in connection_remap:
            new_node, new_port = connection_remap[key]
            new_connections.append(c.model_copy(update={
                "from_node": new_node,
                "from_port": new_port,
            }))
        elif c.from_node in converted_ytdlp_ids and c.from_port == "video_path":
            # Converted to audio_input but no video_input spawned (no video file found)
            # Drop this connection
            continue
        else:
            new_connections.append(c)

    return workflow.model_copy(update={"nodes": all_nodes, "connections": new_connections})


def _workflow_to_utai_bytes(workflow: Workflow) -> bytes:
    """Pack a workflow into a .utai/.utaiTr (zip) file in memory.

    Structure for workflow (.utai):
        workflow.json          — node graph + params
        buffers/registry.json  — buffer metadata (name, sample_rate, ...)
        buffers/<id>.wav       — raw audio data

    Structure for training (.utaiTr) — adds:
        training_audio/<node_id>/<filename>  — training dataset audio files
    """
    buf = io.BytesIO()
    buffer_entries = _collect_buffer_entries(workflow)
    is_training = workflow.project_type == "training"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", workflow.model_dump_json(indent=2))

        # Pack audio buffers + registry (workflow projects)
        registry: dict[str, dict] = {}
        for bid, meta in buffer_entries.items():
            wav_bytes = buffer_manager.get_wav_bytes(bid)
            if wav_bytes:
                zf.writestr(f"buffers/{bid}.wav", wav_bytes)
                registry[bid] = meta
                logger.debug(f"Packed buffer {bid} ({len(wav_bytes)} bytes)")

        if registry:
            zf.writestr("buffers/registry.json", json.dumps(registry, ensure_ascii=False, indent=2))
            logger.info(f"Exported {len(registry)} audio buffer(s)")

        # Pack training dataset audio files (training projects)
        if is_training:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            packed_count = 0
            for node in workflow.nodes:
                if node.type != "train_audio_import":
                    continue
                dataset_dir = node.params.get("dataset_dir", "")
                files = node.params.get("files", [])
                if not dataset_dir or not files:
                    continue
                dataset_path = Path(dataset_dir)
                for fname in files:
                    fpath = dataset_path / fname
                    if fpath.exists() and fpath.suffix.lower() in audio_exts:
                        try:
                            zf.writestr(
                                f"training_audio/{node.id}/{fname}",
                                fpath.read_bytes(),
                            )
                            packed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to pack training audio {fname}: {e}")
            if packed_count:
                logger.info(f"Exported {packed_count} training audio file(s)")

    buf.seek(0)
    return buf.read()


def _utai_bytes_to_workflow(data: bytes) -> dict:
    """Extract workflow data from a .utai/.utaiTr (zip) file.

    Restores audio buffers (workflow) and/or training dataset files (training).
    """
    from app.backend.config import TEMP_DIR

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            if "workflow.json" not in zf.namelist():
                raise ValueError("No workflow.json found in project file")

            wf_data = json.loads(zf.read("workflow.json"))

            # ── Restore audio buffers (workflow projects) ──
            registry: dict[str, dict] = {}
            if "buffers/registry.json" in zf.namelist():
                registry = json.loads(zf.read("buffers/registry.json"))

            id_map: dict[str, str] = {}  # old_id -> new_id
            for name in zf.namelist():
                if name.startswith("buffers/") and name.endswith(".wav"):
                    old_id = name[len("buffers/"):-len(".wav")]
                    wav_bytes = zf.read(name)
                    try:
                        import soundfile as sf
                        samples, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                        meta = registry.get(old_id, {})
                        audio = AudioData(
                            samples=samples,
                            sample_rate=sr,
                            name=meta.get("name", old_id),
                        )
                        new_id = buffer_manager.store(audio)
                        id_map[old_id] = new_id
                        logger.debug(f"Restored buffer {old_id} -> {new_id}")
                    except Exception as e:
                        logger.warning(f"Failed to restore buffer {old_id}: {e}")

            if id_map:
                for node in wf_data.get("nodes", []):
                    params = node.get("params", {})
                    old_bid = params.get("buffer_id")
                    if old_bid and old_bid in id_map:
                        params["buffer_id"] = id_map[old_bid]
                logger.info(f"Imported {len(id_map)} audio buffer(s)")

            # ── Restore training dataset audio (training projects) ──
            training_audio_names = [
                n for n in zf.namelist() if n.startswith("training_audio/")
            ]
            if training_audio_names:
                # Group by node_id: training_audio/<node_id>/<filename>
                node_files: dict[str, list[str]] = {}
                for name in training_audio_names:
                    parts = name.split("/", 2)  # ["training_audio", node_id, filename]
                    if len(parts) == 3 and parts[2]:
                        node_id = parts[1]
                        node_files.setdefault(node_id, []).append(name)

                restored_count = 0
                for node_id, zip_paths in node_files.items():
                    # Create a fresh temp dir for this node's dataset
                    import uuid as _uuid
                    storage_key = f"import_{node_id.replace(chr(45), '_')}_{_uuid.uuid4().hex[:8]}"
                    dest_dir = TEMP_DIR / "datasets" / storage_key
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    filenames = []
                    for zp in zip_paths:
                        fname = zp.split("/", 2)[2]
                        file_bytes = zf.read(zp)
                        (dest_dir / fname).write_bytes(file_bytes)
                        filenames.append(fname)
                        restored_count += 1

                    # Update the node's params to point to the new directory
                    for node in wf_data.get("nodes", []):
                        if node.get("id") == node_id:
                            params = node.get("params", {})
                            params["dataset_dir"] = str(dest_dir)
                            params["files"] = filenames
                            node["params"] = params
                            break

                logger.info(f"Restored {restored_count} training audio file(s)")

            return wf_data
    except zipfile.BadZipFile:
        raise ValueError("Invalid project file (not a valid zip)")


@router.post("/export")
async def export_workflow(workflow: Workflow):
    """Export workflow as .utai file via native save dialog."""
    # Convert yt-dlp nodes to audio_input for portability
    if workflow.project_type != "training":
        workflow = _convert_ytdlp_nodes_for_export(workflow)
    utai_bytes = _workflow_to_utai_bytes(workflow)
    ext = ".utaiTr" if workflow.project_type == "training" else ".utai"
    filename = f"{workflow.name or 'untitled'}{ext}"

    def _save_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            ft = [("utai training project", "*.utaiTr")] if ext == ".utaiTr" else [("utai project", "*.utai")]
            ft.append(("All files", "*.*"))
            path = filedialog.asksaveasfilename(
                title="Export Project",
                defaultextension=ext,
                filetypes=ft,
                initialfile=filename,
            )
            root.destroy()
            return path or ""
        except Exception:
            return ""

    loop = asyncio.get_running_loop()
    save_path = await loop.run_in_executor(None, _save_dialog)

    if not save_path:
        return {"status": "cancelled"}

    Path(save_path).write_bytes(utai_bytes)
    logger.info(f"Workflow exported: {save_path}")
    return {"status": "exported", "path": save_path}


@router.post("/import")
async def import_workflow(file: UploadFile = File(...)):
    """Import a .utai project file (upload)."""
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    try:
        data = _utai_bytes_to_workflow(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return data


@router.post("/import-path")
async def import_workflow_from_path(body: dict):
    """Import a .utai file given its filesystem path (used by Electron)."""
    file_path = body.get("path", "")
    if not file_path:
        raise HTTPException(400, "Missing 'path'")

    p = Path(file_path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    if p.suffix.lower() not in (".utai", ".utaitr", ".utaisynth"):
        raise HTTPException(400, "Unsupported file extension")

    if p.suffix.lower() == ".utaisynth":
        from app.backend.api.routes_synth import _unpack_synth_project
        result = _unpack_synth_project(p.read_bytes())
        result["project_type"] = "synth"
        result["path"] = file_path
        return result

    try:
        data = _utai_bytes_to_workflow(p.read_bytes())
    except ValueError as e:
        raise HTTPException(400, str(e))

    data["_source_path"] = file_path
    return data


@router.post("/export-to-path")
async def export_workflow_to_path(body: dict):
    """Export workflow as .utai to a given filesystem path (used by Electron)."""
    save_path = body.get("path", "")
    workflow_data = body.get("workflow")
    if not save_path or not workflow_data:
        raise HTTPException(400, "Missing 'path' or 'workflow'")

    try:
        workflow = Workflow(**workflow_data)
    except Exception as e:
        raise HTTPException(400, f"Invalid workflow data: {e}")

    if workflow.project_type != "training":
        workflow = _convert_ytdlp_nodes_for_export(workflow)
    utai_bytes = _workflow_to_utai_bytes(workflow)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_bytes(utai_bytes)
    logger.info(f"Workflow exported: {save_path}")
    return {"status": "exported", "path": save_path}


@router.post("/import-dialog")
async def import_workflow_dialog():
    """Open native file dialog to select a .utai file, then import it."""
    def _open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import Project",
                filetypes=[("utai project", "*.utai;*.utaiTr;*.utaiSynth"), ("All files", "*.*")],
            )
            root.destroy()
            return path or ""
        except Exception:
            return ""

    loop = asyncio.get_running_loop()
    file_path = await loop.run_in_executor(None, _open_dialog)

    if not file_path:
        return {"status": "cancelled"}

    p = Path(file_path)
    if not p.exists():
        raise HTTPException(404, "File not found")

    # Route .utaiSynth to synth project handler
    if p.suffix.lower() == ".utaisynth":
        from app.backend.api.routes_synth import _unpack_synth_project
        result = _unpack_synth_project(p.read_bytes())
        result["project_type"] = "synth"
        result["path"] = file_path
        return result

    try:
        data = _utai_bytes_to_workflow(p.read_bytes())
    except ValueError as e:
        raise HTTPException(400, str(e))

    data["_source_path"] = file_path
    return data
