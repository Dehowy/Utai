"""Wrapper for RVC Voice Conversion."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.backend.config import RVC_ROOT, TEMP_DIR

logger = logging.getLogger(__name__)


@contextmanager
def _rvc_context(rvc_root: str):
    """Context manager to set up RVC's expected environment.

    RVC expects:
    - Specific env vars (weight_root, index_root, etc.)
    - os.getcwd() to be the RVC root
    - sys.argv to be minimal (to avoid argparse conflicts)
    """
    old_cwd = os.getcwd()
    old_argv = sys.argv
    old_path = sys.path.copy()

    try:
        os.chdir(rvc_root)
        sys.argv = ["rvc"]
        if rvc_root not in sys.path:
            sys.path.insert(0, rvc_root)

        os.environ.setdefault("weight_root", os.path.join(rvc_root, "assets", "weights"))
        os.environ.setdefault("weight_uvr5_root", os.path.join(rvc_root, "assets", "uvr5_weights"))
        os.environ.setdefault("index_root", os.path.join(rvc_root, "logs"))
        os.environ.setdefault("outside_index_root", os.path.join(rvc_root, "assets", "indices"))
        os.environ.setdefault("rmvpe_root", os.path.join(rvc_root, "assets", "rmvpe"))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        yield
    finally:
        sys.path = old_path
        os.chdir(old_cwd)
        sys.argv = old_argv


class RVCWrapper:
    def __init__(self):
        self._rvc_root = str(RVC_ROOT)
        self._vc = None
        self._config = None
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized:
            with _rvc_context(self._rvc_root):
                from configs.config import Config
                self._config = Config()
                from infer.modules.vc.modules import VC
                self._vc = VC(self._config)
            self._initialized = True

    def list_models(self) -> list[dict[str, Any]]:
        """List available RVC voice models, including range data if available."""
        import json as _json
        models = []
        weights_dir = Path(self._rvc_root) / "assets" / "weights"
        if not weights_dir.exists():
            return models

        for model_file in sorted(weights_dir.glob("*.pth")):
            index_files = self._find_index_files(model_file.stem)
            entry: dict[str, Any] = {
                "name": model_file.stem,
                "path": str(model_file),
                "index_files": index_files,
            }
            range_json = weights_dir / f"{model_file.stem}_range.json"
            if range_json.exists():
                try:
                    rd = _json.loads(range_json.read_text(encoding="utf-8"))
                    entry["range"] = {
                        "range_low": rd.get("range_low"),
                        "range_high": rd.get("range_high"),
                        "comfort_low": rd.get("comfort_low"),
                        "comfort_high": rd.get("comfort_high"),
                        "range_label": rd.get("range_label"),
                        "comfort_label": rd.get("comfort_label"),
                    }
                except Exception:
                    pass
            models.append(entry)

        return models

    def _find_index_files(self, model_name: str) -> list[str]:
        """Find .index files that match a model name.

        Searches both logs/ and assets/indices/ directories.
        Uses base model name (without epoch/step suffixes like _e175_s6300)
        for broader matching.
        """
        index_files = []

        # Extract base name: strip _e{N}_s{N} training suffixes
        import re
        base_name = re.sub(r'_e\d+_s\d+$', '', model_name)
        search_names = {model_name.lower(), base_name.lower()}

        search_dirs = [
            Path(self._rvc_root) / "logs",
            Path(self._rvc_root) / "assets" / "indices",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for idx_file in search_dir.rglob("*.index"):
                stem_lower = idx_file.stem.lower()
                if any(name in stem_lower for name in search_names):
                    path_str = str(idx_file)
                    if path_str not in index_files:
                        index_files.append(path_str)

        return index_files

    async def convert(
        self,
        audio: np.ndarray,
        sr: int,
        model_name: str,
        params: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run RVC voice conversion."""
        self._ensure_init()

        if progress_callback:
            dur_s = audio.shape[-1] / sr if audio.ndim == 1 else audio.shape[-1] / sr
            await progress_callback(0.1, f"Input: {dur_s:.1f}s @ {sr}Hz")

        # Write audio to temp file (RVC requires file path input)
        import soundfile as sf
        import uuid as _uuid
        temp_wav = os.path.join(str(TEMP_DIR), f"rvc_input_{_uuid.uuid4().hex[:12]}.wav")
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio_write = audio.T
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio_write = audio
        else:
            audio_write = audio
        sf.write(temp_wav, audio_write, sr)

        if progress_callback:
            await progress_callback(0.2, f"Loading model: {model_name}")

        index_file = params.get("index_file", "")
        with _rvc_context(self._rvc_root):
            self._vc.get_vc(f"{model_name}.pth")

            if progress_callback:
                index_msg = f"Index: {Path(index_file).name}" if index_file and os.path.exists(index_file) else "Index: not used"
                await progress_callback(0.3, f"Model loaded (sr={self._vc.tgt_sr}, v={self._vc.version}). {index_msg}")
                f0_method = params.get("f0_method", "rmvpe")
                f0_key = int(params.get("f0_up_key", 0))
                await progress_callback(0.4, f"Converting: f0={f0_method}, pitch={f0_key:+d}st")

            # Run conversion in thread pool to not block event loop
            def _do_convert():
                return self._vc.vc_single(
                    sid=0,
                    input_audio_path=temp_wav,
                    f0_up_key=int(params.get("f0_up_key", 0)),
                    f0_file=None,
                    f0_method=params.get("f0_method", "rmvpe"),
                    file_index=params.get("index_file", ""),
                    file_index2="",
                    index_rate=float(params.get("index_rate", 0.75)),
                    filter_radius=int(params.get("filter_radius", 3)),
                    resample_sr=int(params.get("resample_sr", 0)),
                    rms_mix_rate=float(params.get("rms_mix_rate", 0.25)),
                    protect=float(params.get("protect", 0.33)),
                )

            loop = asyncio.get_running_loop()
            info, (out_sr, out_audio) = await loop.run_in_executor(None, _do_convert)

        # Clean up temp file
        try:
            os.remove(temp_wav)
        except OSError:
            pass

        if out_audio is None:
            logger.error(f"RVC conversion failed: {info}")
            raise RuntimeError(f"RVC conversion failed: {info}")

        logger.info(f"RVC info: {info}")

        if progress_callback:
            await progress_callback(0.9, f"Done: {out_audio.shape[0]/out_sr:.1f}s @ {out_sr}Hz")

        # Normalize output
        if out_audio.dtype == np.int16:
            out_audio = out_audio.astype(np.float32) / 32768.0

        return out_audio, out_sr


# Global instance
rvc_wrapper = RVCWrapper()
