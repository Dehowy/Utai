"""Synth API routes — DiffSinger rendering, RVC conversion pipeline, G2P, and model listing.

All endpoints are prefixed with /api/synth.
RVC conversion reuses the existing rvc_wrapper from the main Utai backend.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import FileResponse

from app.backend.config import TEMP_DIR, OUTPUT_DIR
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


# --- Render (DiffSinger only) ---

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
