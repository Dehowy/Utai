"""Audio upload, streaming, download, and video serving endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.backend.config import TEMP_DIR
from app.backend.core.audio_buffer import AudioData, buffer_manager

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file. Returns buffer_id for use in workflow."""
    import logging
    logger = logging.getLogger(__name__)

    contents = await file.read()
    logger.info(f"Upload: filename={file.filename}, size={len(contents)}, content_type={file.content_type}")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Write to temp file first (some formats need seekable file / file extension)
    import os
    import tempfile
    ext = os.path.splitext(file.filename or "")[-1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=str(TEMP_DIR))
    try:
        tmp.write(contents)
    finally:
        tmp.close()
    tmp_path = tmp.name

    try:
        try:
            samples, sr = sf.read(tmp_path)
            logger.info(f"soundfile OK: sr={sr}, shape={samples.shape}")
        except Exception as e1:
            logger.warning(f"soundfile failed: {e1}")
            try:
                import librosa
                samples, sr = librosa.load(tmp_path, sr=None, mono=False)
                # librosa returns (channels, samples), convert to (samples, channels)
                if samples.ndim == 2 and samples.shape[0] < samples.shape[1]:
                    samples = samples.T
                logger.info(f"librosa OK: sr={sr}, shape={samples.shape}")
            except Exception as e2:
                logger.error(f"librosa also failed: {e2}")
                raise HTTPException(status_code=400, detail=f"Cannot read audio: {e2}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    audio = AudioData(
        samples=samples.astype(np.float32),
        sample_rate=sr,
        name=file.filename or "uploaded",
    )
    buffer_id = buffer_manager.store(audio)

    return {
        "buffer_id": buffer_id,
        "filename": file.filename,
        "sample_rate": sr,
        "duration": len(samples) / sr,
        "channels": samples.shape[1] if samples.ndim == 2 else 1,
    }


@router.get("/{buffer_id}")
async def stream_audio(buffer_id: str, request: Request):
    """Stream audio as WAV with Range request support for seeking."""
    wav_bytes = buffer_manager.get_wav_bytes(buffer_id)
    if wav_bytes is None:
        raise HTTPException(status_code=404, detail="Audio buffer not found")

    total = len(wav_bytes)
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        try:
            range_spec = range_header.replace("bytes=", "").strip()
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total - 1
        except (ValueError, IndexError):
            start = 0
            end = total - 1
        end = min(end, total - 1)
        if start > end or start >= total:
            raise HTTPException(status_code=416, detail="Range Not Satisfiable")
        length = end - start + 1

        return Response(
            content=wav_bytes[start:end + 1],
            status_code=206,
            media_type="audio/wav",
            headers={
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total),
            "Content-Disposition": f"inline; filename={buffer_id}.wav",
        },
    )


@router.get("/{buffer_id}/download")
async def download_audio(buffer_id: str, format: str = "wav"):
    """Download audio in specified format."""
    audio = buffer_manager.get(buffer_id)
    if audio is None:
        raise HTTPException(status_code=404, detail="Audio buffer not found")

    buf = io.BytesIO()

    try:
        if format == "wav":
            sf.write(buf, audio.samples, audio.sample_rate, format="WAV")
            media_type = "audio/wav"
        elif format == "flac":
            sf.write(buf, audio.samples, audio.sample_rate, format="FLAC")
            media_type = "audio/flac"
        else:
            sf.write(buf, audio.samples, audio.sample_rate, format="WAV")
            media_type = "audio/wav"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot encode audio as {format}: {e}")

    buf.seek(0)
    filename = f"{audio.name or buffer_id}.{format}"

    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/video/{filename:path}")
async def serve_video(filename: str):
    """Serve a video file from temp storage."""
    video_dir = TEMP_DIR / "video_output"
    file_path = video_dir / filename
    if not file_path.exists():
        raise HTTPException(404, "Video not found")
    # Determine media type from extension
    ext = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".flv": "video/x-flv",
    }
    media_type = media_types.get(ext, "video/mp4")
    return FileResponse(str(file_path), media_type=media_type)


@router.get("/{buffer_id}/info")
async def audio_info(buffer_id: str):
    """Get audio metadata."""
    audio = buffer_manager.get(buffer_id)
    if audio is None:
        raise HTTPException(status_code=404, detail="Audio buffer not found")

    return {
        "buffer_id": buffer_id,
        "name": audio.name,
        "sample_rate": audio.sample_rate,
        "duration": len(audio.samples) / audio.sample_rate,
        "channels": audio.samples.shape[1] if audio.samples.ndim == 2 else 1,
    }
