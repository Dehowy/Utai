"""Synth API routes — Score2HuBERT rendering, DiffSinger fallback, RVC conversion, G2P.

All endpoints are prefixed with /api/synth.
Score2HuBERT pipeline: notes JSON → G2P → model → RVC → post mute → WAV.
DiffSinger pipeline (legacy): USTX YAML → UtaiRenderer.exe → WAV → RVC → WAV.
"""

import io
import json
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, UploadFile, File, Body, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.backend.config import TEMP_DIR, OUTPUT_DIR
from app.backend.core.audio_buffer import AudioData, buffer_manager
from app.backend.wrappers.synth_wrapper import get_synth_wrapper, SYNTH_LOG_BUFFER


def _safe_filename(name: str) -> str:
    """Sanitize user-supplied filenames to prevent path traversal."""
    return Path(name).name or "upload"

router = APIRouter(prefix="/api/synth", tags=["synth"])
logger = logging.getLogger("utai.synth")

# --- G2P ---

_g2p_instance = None


def _get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        from g2p_en import G2p
        _g2p_instance = G2p()
    return _g2p_instance


@router.post("/g2p")
async def g2p_convert(text: str = Form(...), lang: str = Form("en")):
    """Convert text to phonemes for DiffSinger."""
    if lang == "en":
        g2p = _get_g2p()
        raw = g2p(text)
        words = []
        current = []
        for p in raw:
            if p.strip() == '':
                if current:
                    words.append(current)
                    current = []
            else:
                cleaned = re.sub(r'[0-9]', '', p).lower()
                if cleaned:
                    current.append(cleaned)
        if current:
            words.append(current)
        return {
            "lang": lang,
            "input": text,
            "words": [{"phonemes": " ".join(w)} for w in words],
            "all_phonemes": " ".join(" ".join(w) for w in words),
        }
    elif lang in ("zh", "ja"):
        chars = [c for c in text if c.strip()]
        return {
            "lang": lang,
            "input": text,
            "words": [{"phonemes": c} for c in chars],
            "all_phonemes": " ".join(chars),
        }
    else:
        return {"error": f"Unsupported language: {lang}"}


# --- Score2HuBERT render (primary pipeline) ---

@router.post("/s2h/render")
async def s2h_render(
    notes: str = Form(...),
    bpm: float = Form(120.0),
    language: str = Form("zh"),
    rvc_model: str = Form(""),
    f0_up_key: int = Form(0),
    mute_strategy: str = Form("auto"),
    index_rate: float = Form(0.75),
    affine_transform: bool = Form(False),
):
    """Render notes via Score2HuBERT → RVC → post mute pipeline.

    notes: JSON array of {position, duration, tone, lyric}
    affine_transform: if True, use affine pitch transform for range extension
    """
    from fastapi import HTTPException

    try:
        notes_list = json.loads(notes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid notes JSON: {e}")

    if not notes_list:
        raise HTTPException(status_code=422, detail="No notes to render")

    logger.info("S2H render request: %d notes, lang=%s, rvc=%s, affine=%s",
                len(notes_list), language, rvc_model, affine_transform)

    from app.backend.wrappers.s2h_wrapper import get_s2h_wrapper, RVC_WEIGHTS_DIR

    synth_output = OUTPUT_DIR / "synth"
    synth_output.mkdir(parents=True, exist_ok=True)
    output_path = str(synth_output / "s2h_render.wav")

    range_data = None
    if affine_transform and rvc_model:
        clean = rvc_model.replace(".pth", "")
        range_json = RVC_WEIGHTS_DIR / f"{clean}_range.json"
        if range_json.exists():
            range_data = json.loads(range_json.read_text(encoding="utf-8"))

    try:
        wrapper = get_s2h_wrapper()
        result_path = wrapper.render_to_file(
            output_path,
            notes=notes_list,
            bpm=bpm,
            language=language,
            rvc_model=rvc_model,
            f0_up_key=f0_up_key,
            mute_strategy=mute_strategy,
            index_rate=index_rate,
            affine_transform=affine_transform,
            range_data=range_data,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"S2H render error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"S2H render failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(result_path, media_type="audio/wav", filename="s2h_render.wav")


# --- Render (DiffSinger legacy) ---

@router.post("/render/content")
async def render_content(
    content: str = Form(...),
    filename: str = Form("project.ustx"),
    track: Optional[int] = Form(None),
):
    """Render USTX content (YAML string) to WAV using DiffSinger."""
    wrapper = get_synth_wrapper()
    result = await wrapper.render_content(content, filename, track)

    if not result["success"]:
        return {"error": result["message"]}

    return FileResponse(
        result["output_path"],
        media_type="audio/wav",
        filename=f"{Path(filename).stem}_rendered.wav",
    )


@router.post("/render")
async def render_project(
    file: UploadFile = File(...),
    track: Optional[int] = Form(None),
):
    """Render an uploaded .ustx file to WAV."""
    synth_temp = TEMP_DIR / "synth"
    synth_temp.mkdir(parents=True, exist_ok=True)

    ustx_path = synth_temp / _safe_filename(file.filename)
    with open(ustx_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    wrapper = get_synth_wrapper()
    result = await wrapper.render(str(ustx_path), track=track)

    if not result["success"]:
        return {"error": result["message"]}

    return FileResponse(
        result["output_path"],
        media_type="audio/wav",
        filename=f"{ustx_path.stem}_rendered.wav",
    )


# --- Pipeline (DiffSinger + RVC) ---

@router.post("/pipeline/content")
async def run_pipeline_content(
    content: str = Form(...),
    filename: str = Form("project.ustx"),
    rvc_model: str = Form(...),
    track: Optional[int] = Form(None),
    f0_up_key: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    filter_radius: int = Form(3),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
):
    """Full pipeline: USTX content -> DiffSinger render -> RVC convert -> output WAV."""
    # Step 1: Render with DiffSinger
    wrapper = get_synth_wrapper()
    render_result = await wrapper.render_content(content, filename, track)

    if not render_result["success"]:
        return {"success": False, "stage": "render", "error": render_result["message"]}

    rendered_path = render_result["output_path"]

    # Step 2: RVC voice conversion (reuse main project's RVC wrapper)
    from app.backend.wrappers.rvc_wrapper import rvc_wrapper
    import soundfile as sf
    import numpy as np

    synth_output = OUTPUT_DIR / "synth"
    synth_output.mkdir(parents=True, exist_ok=True)
    output_path = str(synth_output / f"{Path(filename).stem}_final.wav")

    try:
        audio_data, sr = sf.read(rendered_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)

        output_audio, output_sr = await rvc_wrapper.convert(
            audio=audio_data,
            sr=sr,
            model_name=rvc_model.replace(".pth", ""),
            params={
                "f0_up_key": f0_up_key,
                "f0_method": f0_method,
                "index_file": "",
                "index_rate": index_rate,
                "filter_radius": filter_radius,
                "rms_mix_rate": rms_mix_rate,
                "protect": protect,
            },
        )

        sf.write(output_path, output_audio, output_sr)
    except Exception as e:
        logger.error(f"RVC conversion failed: {e}")
        return {"success": False, "stage": "convert", "error": str(e)}

    return FileResponse(output_path, media_type="audio/wav", filename=f"{Path(filename).stem}_final.wav")


# --- Convert (RVC only, for standalone audio conversion) ---

@router.post("/convert")
async def convert_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    f0_up_key: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    filter_radius: int = Form(3),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
):
    """Convert uploaded audio using the shared RVC wrapper."""
    import soundfile as sf
    import numpy as np

    synth_temp = TEMP_DIR / "synth"
    synth_temp.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    input_path = synth_temp / f"rvc_input_{_safe_filename(file.filename)}"
    input_path.write_bytes(contents)

    if len(contents) < 100:
        return {"error": f"Input file too small ({len(contents)} bytes)"}

    from app.backend.wrappers.rvc_wrapper import rvc_wrapper

    synth_output = OUTPUT_DIR / "synth"
    synth_output.mkdir(parents=True, exist_ok=True)
    output_path = str(synth_output / f"rvc_output_{_safe_filename(file.filename)}")

    try:
        audio_data, sr = sf.read(str(input_path))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)

        output_audio, output_sr = await rvc_wrapper.convert(
            audio=audio_data,
            sr=sr,
            model_name=model.replace(".pth", ""),
            params={
                "f0_up_key": f0_up_key,
                "f0_method": f0_method,
                "index_file": "",
                "index_rate": index_rate,
                "filter_radius": filter_radius,
                "rms_mix_rate": rms_mix_rate,
                "protect": protect,
            },
        )

        sf.write(output_path, output_audio, output_sr)
    except Exception as e:
        logger.error(f"RVC conversion failed: {e}")
        return {"error": str(e)}

    return FileResponse(output_path, media_type="audio/wav", filename=f"converted_{file.filename}")


# --- Models & Info ---

@router.get("/singers")
async def list_singers():
    """List available DiffSinger voice banks."""
    wrapper = get_synth_wrapper()
    try:
        singers = await wrapper.list_singers()
    except FileNotFoundError:
        singers = wrapper.list_singers_from_dir()
    return singers


@router.post("/stop")
async def force_stop():
    """Force stop any running render/convert process."""
    wrapper = get_synth_wrapper()
    stopped = await wrapper.force_stop()
    return {"stopped": stopped}


@router.get("/logs")
async def get_logs(limit: int = 50):
    """Get recent synth backend log entries (newest first)."""
    entries = list(SYNTH_LOG_BUFFER)[-limit:]
    entries.reverse()
    return {"logs": entries}


# ══════════════════════════════════════════════════════════════
# Synth project save / load / autosave / export-audio
# ══════════════════════════════════════════════════════════════

AUTOSAVE_DIR = TEMP_DIR / "synth_autosave"
AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)


class ProjectSaveRequest(BaseModel):
    project: dict
    trackEffects: dict = {}
    settings: dict = {}
    path: str = ""


def _pack_synth_project(project: dict, track_effects: dict, settings: dict) -> bytes:
    """Pack synth project into .utaiSynth zip bytes."""
    buf = io.BytesIO()
    registry: dict[str, dict] = {}

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project, ensure_ascii=False, indent=2))
        zf.writestr("effects.json", json.dumps(track_effects, ensure_ascii=False, indent=2))
        zf.writestr("settings.json", json.dumps(settings, ensure_ascii=False, indent=2))
        zf.writestr("meta.json", json.dumps({
            "version": "1.0",
            "format": "utaiSynth",
        }, indent=2))

        seen_buffers: set[str] = set()
        for wp in project.get("wave_parts", []):
            bid = wp.get("buffer_id")
            if bid and bid not in seen_buffers:
                wav_bytes = buffer_manager.get_wav_bytes(bid)
                if wav_bytes:
                    zf.writestr(f"audio/{bid}.wav", wav_bytes)
                    audio = buffer_manager.get(bid)
                    registry[bid] = {
                        "name": audio.name if audio else bid,
                        "sample_rate": audio.sample_rate if audio else 44100,
                    }
                    seen_buffers.add(bid)

        if registry:
            zf.writestr("audio/registry.json", json.dumps(registry, ensure_ascii=False, indent=2))

    buf.seek(0)
    return buf.read()


def _unpack_synth_project(data: bytes) -> dict:
    """Unpack .utaiSynth zip, restore audio buffers, return project data."""
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        if "project.json" not in zf.namelist():
            raise ValueError("Invalid .utaiSynth file: no project.json")

        project = json.loads(zf.read("project.json"))
        track_effects = {}
        settings = {}
        if "effects.json" in zf.namelist():
            track_effects = json.loads(zf.read("effects.json"))
        if "settings.json" in zf.namelist():
            settings = json.loads(zf.read("settings.json"))

        registry: dict[str, dict] = {}
        if "audio/registry.json" in zf.namelist():
            registry = json.loads(zf.read("audio/registry.json"))

        id_map: dict[str, str] = {}
        for name in zf.namelist():
            if name.startswith("audio/") and name.endswith(".wav"):
                old_id = name[len("audio/"):-len(".wav")]
                try:
                    wav_bytes = zf.read(name)
                    samples, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                    meta = registry.get(old_id, {})
                    audio = AudioData(
                        samples=samples.astype(np.float32),
                        sample_rate=sr,
                        name=meta.get("name", old_id),
                    )
                    new_id = buffer_manager.store(audio)
                    id_map[old_id] = new_id
                except Exception as e:
                    logger.warning(f"Failed to restore audio buffer {old_id}: {e}")

        if id_map:
            for wp in project.get("wave_parts", []):
                old_bid = wp.get("buffer_id")
                if old_bid and old_bid in id_map:
                    wp["buffer_id"] = id_map[old_bid]

        return {"project": project, "trackEffects": track_effects, "settings": settings, "bufferIdMap": id_map}


@router.post("/project/save")
async def save_project(req: ProjectSaveRequest):
    """Save synth project to .utaiSynth file at given path."""
    if not req.path:
        raise HTTPException(400, "No save path specified")
    try:
        data = _pack_synth_project(req.project, req.trackEffects, req.settings)
        save_path = Path(req.path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(data)
        return {"success": True, "path": str(save_path), "size": len(data)}
    except Exception as e:
        logger.exception("Failed to save synth project")
        raise HTTPException(500, str(e))


@router.post("/project/save-dialog")
async def save_project_dialog(req: ProjectSaveRequest):
    """Open native save dialog, then save synth project."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title="Save Synth Project",
        defaultextension=".utaiSynth",
        filetypes=[("Utai Synth Project", "*.utaiSynth"), ("All Files", "*.*")],
        initialfile=f"{req.project.get('name', 'project')}.utaiSynth",
    )
    root.destroy()
    if not path:
        return {"cancelled": True}
    try:
        data = _pack_synth_project(req.project, req.trackEffects, req.settings)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        return {"success": True, "path": path, "size": len(data)}
    except Exception as e:
        logger.exception("Failed to save synth project")
        raise HTTPException(500, str(e))


@router.post("/project/open-dialog")
async def open_project_dialog():
    """Open native file dialog, load .utaiSynth project."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Open Synth Project",
        filetypes=[("Utai Synth Project", "*.utaiSynth"), ("All Files", "*.*")],
    )
    root.destroy()
    if not path:
        return {"cancelled": True}
    try:
        data = Path(path).read_bytes()
        result = _unpack_synth_project(data)
        result["path"] = path
        return result
    except Exception as e:
        logger.exception("Failed to open synth project")
        raise HTTPException(500, str(e))


@router.post("/project/open-path")
async def open_project_path(path: str = Body(..., embed=True)):
    """Load .utaiSynth project from given path (for Electron file association)."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    try:
        result = _unpack_synth_project(p.read_bytes())
        result["path"] = str(p)
        return result
    except Exception as e:
        logger.exception("Failed to open synth project")
        raise HTTPException(500, str(e))


@router.post("/project/open-upload")
async def open_project_upload(file: UploadFile = File(...)):
    """Load .utaiSynth project from uploaded file."""
    try:
        data = await file.read()
        result = _unpack_synth_project(data)
        return result
    except Exception as e:
        logger.exception("Failed to open synth project")
        raise HTTPException(500, str(e))


@router.post("/project/autosave")
async def autosave_project(req: ProjectSaveRequest):
    """Save project state to temp storage for crash recovery."""
    try:
        state = json.dumps({
            "project": req.project,
            "trackEffects": req.trackEffects,
            "settings": req.settings,
        }, ensure_ascii=False)
        (AUTOSAVE_DIR / "state.json").write_text(state, encoding="utf-8")
        buffer_ids = [wp.get("buffer_id") for wp in req.project.get("wave_parts", []) if wp.get("buffer_id")]
        for bid in buffer_ids:
            wav_path = AUTOSAVE_DIR / f"{bid}.wav"
            if not wav_path.exists():
                wav_bytes = buffer_manager.get_wav_bytes(bid)
                if wav_bytes:
                    wav_path.write_bytes(wav_bytes)
        return {"success": True}
    except Exception as e:
        logger.warning(f"Autosave failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/project/autosave")
async def load_autosave():
    """Load autosaved project state if available."""
    state_path = AUTOSAVE_DIR / "state.json"
    if not state_path.exists():
        return {"available": False}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        project = state.get("project", {})
        for wp in project.get("wave_parts", []):
            bid = wp.get("buffer_id")
            if bid:
                wav_path = AUTOSAVE_DIR / f"{bid}.wav"
                if wav_path.exists() and not buffer_manager.get(bid):
                    samples, sr = sf.read(str(wav_path), dtype="float32")
                    audio = AudioData(samples=samples.astype(np.float32), sample_rate=sr, name=bid)
                    buffer_manager.store(audio, force_id=bid)
        return {
            "available": True,
            "project": project,
            "trackEffects": state.get("trackEffects", {}),
            "settings": state.get("settings", {}),
        }
    except Exception as e:
        logger.warning(f"Failed to load autosave: {e}")
        return {"available": False, "error": str(e)}


@router.delete("/project/autosave")
async def clear_autosave():
    """Clear autosaved state."""
    try:
        for f in AUTOSAVE_DIR.iterdir():
            f.unlink(missing_ok=True)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ExportAudioRequest(BaseModel):
    project: dict
    trackAudioUrls: dict = {}
    format: str = "wav"
    sample_rate: int = 44100
    bit_depth: int = 16


@router.post("/project/export-audio-dialog")
async def export_audio_dialog(req: ExportAudioRequest):
    """Mix all tracks and export as audio file via native save dialog."""
    import tkinter as tk
    from tkinter import filedialog
    fmt_ext = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}
    ext = fmt_ext.get(req.format, ".wav")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title="Export Audio",
        defaultextension=ext,
        filetypes=[(f"{req.format.upper()} Audio", f"*{ext}"), ("All Files", "*.*")],
        initialfile=f"{req.project.get('name', 'export')}{ext}",
    )
    root.destroy()
    if not path:
        return {"cancelled": True}

    try:
        mixed = _mix_project_audio(req.project, req.trackAudioUrls, req.sample_rate)
        if mixed is None:
            raise ValueError("No audio to export")

        out_path = Path(path)
        subtype_map = {"wav": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}, "flac": {16: "PCM_16", 24: "PCM_24"}}
        subtype = subtype_map.get(req.format, {}).get(req.bit_depth, "PCM_16")

        if req.format == "mp3":
            tmp_wav = TEMP_DIR / "export_tmp.wav"
            sf.write(str(tmp_wav), mixed, req.sample_rate)
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", str(tmp_wav), "-b:a", "320k", str(out_path)], check=True, capture_output=True)
            tmp_wav.unlink(missing_ok=True)
        elif req.format == "ogg":
            tmp_wav = TEMP_DIR / "export_tmp.wav"
            sf.write(str(tmp_wav), mixed, req.sample_rate)
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libvorbis", "-q:a", "6", str(out_path)], check=True, capture_output=True)
            tmp_wav.unlink(missing_ok=True)
        else:
            sf.write(str(out_path), mixed, req.sample_rate, subtype=subtype)

        return {"success": True, "path": str(out_path)}
    except Exception as e:
        logger.exception("Failed to export audio")
        raise HTTPException(500, str(e))


def _mix_project_audio(project: dict, track_audio_urls: dict, target_sr: int) -> Optional[np.ndarray]:
    """Mix all track audio buffers into a single stereo array."""
    from app.backend.config import TEMP_DIR as _td
    tracks = project.get("tracks", [])
    bpm = project.get("tempos", [{}])[0].get("bpm", 120)
    resolution = 480
    tps = bpm / 60 * resolution

    max_samples = 0
    segments: list[tuple[np.ndarray, int, float, float]] = []

    for wp in project.get("wave_parts", []):
        bid = wp.get("buffer_id")
        if not bid:
            continue
        audio = buffer_manager.get(bid)
        if not audio:
            continue
        track_idx = wp.get("track_no", 0)
        track = tracks[track_idx] if track_idx < len(tracks) else None
        if track and track.get("mute"):
            continue

        sr = audio.sample_rate
        samples = audio.samples
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        skip_samples = int((wp.get("skip", 0) / 1000) * sr)
        dur_samples = int((wp.get("file_duration_ms", 0) / 1000) * sr)
        clip = samples[:, skip_samples:skip_samples + dur_samples]

        start_sec = wp.get("position", 0) / tps
        start_sample = int(start_sec * target_sr)

        if sr != target_sr:
            try:
                import soxr
                clip = np.stack([soxr.resample(clip[ch], sr, target_sr) for ch in range(clip.shape[0])])
            except ImportError:
                pass

        track_vol = 10 ** ((track.get("volume", 0) if track else 0) / 20) if (track and track.get("volume", 0) > -24) else 0
        clip_vol = 10 ** ((wp.get("volume", 0) or 0) / 20) if (wp.get("volume", 0) or 0) > -24 else (0 if (wp.get("volume", 0) or 0) <= -24 else 1)
        clip = clip * track_vol * clip_vol

        end_sample = start_sample + clip.shape[-1]
        if end_sample > max_samples:
            max_samples = end_sample
        segments.append((clip, start_sample, track.get("pan", 0) if track else 0, wp.get("pan", 0) or 0))

    if max_samples == 0:
        return None

    mixed = np.zeros((2, max_samples), dtype=np.float32)
    for clip, start, track_pan, clip_pan in segments:
        pan = max(-100, min(100, track_pan + clip_pan)) / 100
        l_gain = min(1.0, 1.0 - pan)
        r_gain = min(1.0, 1.0 + pan)
        end = start + clip.shape[-1]
        if clip.shape[0] == 1:
            mixed[0, start:end] += clip[0] * l_gain
            mixed[1, start:end] += clip[0] * r_gain
        else:
            mixed[0, start:end] += clip[0] * l_gain
            mixed[1, start:end] += clip[1] * r_gain

    return mixed.T
