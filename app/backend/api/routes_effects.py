"""Audio effects processing API."""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.backend.wrappers import effects_wrapper
from app.backend.config import TEMP_DIR

logger = logging.getLogger("utai.api.effects")
router = APIRouter(prefix="/api/effects", tags=["effects"])


class EffectItem(BaseModel):
    type: str
    params: dict = {}
    enabled: bool = True


class ApplyRequest(BaseModel):
    """Apply effects to an audio buffer by ID or path."""
    audio_path: str
    effects: list[EffectItem]
    sample_rate: int = 44100


@router.get("/types")
async def list_effect_types():
    """Return all available effect types with their parameter definitions."""
    return effects_wrapper.get_effect_defs()


@router.post("/apply")
async def apply_effects(req: ApplyRequest):
    """Apply a chain of effects to an audio file. Returns processed WAV."""
    if not os.path.exists(req.audio_path):
        raise HTTPException(404, f"Audio file not found: {req.audio_path}")
    try:
        result_path = effects_wrapper.apply_effects(
            req.audio_path,
            [e.dict() for e in req.effects],
            req.sample_rate,
        )
        return FileResponse(result_path, media_type="audio/wav", filename="processed.wav")
    except Exception as e:
        logger.exception("Effects processing failed")
        raise HTTPException(500, str(e))


@router.post("/apply-upload")
async def apply_effects_upload(
    file: UploadFile = File(...),
    effects_json: str = Form(...),
):
    """Apply effects to an uploaded audio file. Returns processed WAV."""
    import json
    try:
        effects = json.loads(effects_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid effects JSON")

    # Save uploaded file
    tmp_dir = TEMP_DIR / "effects"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{os.getpid()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        result_path = effects_wrapper.apply_effects(str(tmp_path), effects)
        return FileResponse(result_path, media_type="audio/wav", filename="processed.wav")
    except Exception as e:
        logger.exception("Effects processing failed")
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
