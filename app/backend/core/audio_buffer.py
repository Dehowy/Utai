"""Disk-backed audio buffer management for passing data between nodes.

Audio data is written to temporary WAV files on disk and loaded on-demand.
Memory is released immediately after writing; reads load from disk then free.
"""

from __future__ import annotations

import io
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.backend.config import TEMP_DIR

logger = logging.getLogger(__name__)

BUFFER_DIR = TEMP_DIR / "audio_buffers"


@dataclass
class AudioData:
    """Audio data with metadata."""
    samples: np.ndarray  # shape: (num_samples,) for mono, (num_samples, channels) for stereo
    sample_rate: int
    name: str = ""


class AudioBufferManager:
    """Manages audio buffers during workflow execution.

    Audio data is stored as WAV files on disk identified by unique IDs.
    Memory is only used transiently during read/write operations.
    """

    def __init__(self):
        self._metadata: dict[str, dict] = {}  # buffer_id -> {path, sample_rate, name}
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)

    def store(self, audio: AudioData) -> str:
        """Store audio data to disk, return buffer ID. Memory is released immediately."""
        import soundfile as sf

        buffer_id = uuid.uuid4().hex[:12]
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = BUFFER_DIR / f"{buffer_id}.wav"

        sf.write(str(wav_path), audio.samples, audio.sample_rate, format="WAV")

        self._metadata[buffer_id] = {
            "path": str(wav_path),
            "sample_rate": audio.sample_rate,
            "name": audio.name,
        }

        logger.debug(f"Buffer stored: {buffer_id} -> {wav_path.name}")
        return buffer_id

    def get(self, buffer_id: str) -> AudioData | None:
        """Load audio data from disk. Caller receives data in memory."""
        meta = self._metadata.get(buffer_id)
        if meta is None:
            return None

        path = Path(meta["path"])
        if not path.exists():
            logger.warning(f"Buffer file missing: {buffer_id} ({path})")
            self._metadata.pop(buffer_id, None)
            return None

        import soundfile as sf
        samples, sr = sf.read(str(path), dtype="float32")
        return AudioData(samples=samples, sample_rate=sr, name=meta.get("name", ""))

    def get_wav_bytes(self, buffer_id: str) -> bytes | None:
        """Get audio as WAV bytes for streaming to browser (reads directly from disk)."""
        meta = self._metadata.get(buffer_id)
        if meta is None:
            return None

        path = Path(meta["path"])
        if not path.exists():
            self._metadata.pop(buffer_id, None)
            return None

        return path.read_bytes()

    def remove(self, buffer_id: str):
        """Remove a single buffer (disk file + metadata)."""
        meta = self._metadata.pop(buffer_id, None)
        if meta:
            path = Path(meta["path"])
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self):
        """Remove all buffers (disk files + metadata)."""
        self._metadata.clear()
        if BUFFER_DIR.exists():
            shutil.rmtree(str(BUFFER_DIR), ignore_errors=True)
            BUFFER_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def buffer_ids(self) -> list[str]:
        return list(self._metadata.keys())


# Global buffer manager instance
buffer_manager = AudioBufferManager()
