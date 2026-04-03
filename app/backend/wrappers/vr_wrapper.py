"""Wrapper for UVR Vocal Remover models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from contextlib import contextmanager

import numpy as np

from app.backend.config import MSST_ROOT
from app.backend.core.model_manager import model_manager

logger = logging.getLogger(__name__)


@contextmanager
def _msst_cwd():
    """Temporarily chdir to MSST root (many MSST internals use relative paths)."""
    old = os.getcwd()
    os.chdir(str(MSST_ROOT))
    try:
        yield
    finally:
        os.chdir(old)


class VRWrapper:
    def __init__(self):
        self._msst_root = Path(MSST_ROOT)
        self._initialized = False
        self._models_info: dict[str, Any] = {}
        self._load_models_info()

    def _load_models_info(self):
        info_path = self._msst_root / "data" / "models_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self._models_info = json.load(f)

    def _ensure_init(self):
        if not self._initialized:
            root = str(self._msst_root)
            if root not in sys.path:
                sys.path.insert(0, root)

            old_cwd = os.getcwd()
            os.chdir(root)

            try:
                # Patch MSST's logger
                msst_logger = logging.getLogger("logger")
                if not hasattr(msst_logger, "console_handler"):
                    old_propagate = msst_logger.propagate
                    msst_logger.propagate = False
                    from utils.logger import get_logger as msst_get_logger
                    msst_get_logger()
                    msst_logger.propagate = old_propagate

                import modules.vocal_remover.vr_separator  # noqa: F401
            except Exception as e:
                logger.warning(f"VR init issue: {e}")
                msst_logger = logging.getLogger("logger")
                if not hasattr(msst_logger, "console_handler"):
                    handler = logging.StreamHandler()
                    handler.setLevel(logging.INFO)
                    msst_logger.console_handler = handler
            finally:
                os.chdir(old_cwd)

            self._initialized = True

    def list_models(self) -> list[dict[str, Any]]:
        """List VR models from models_info.json with their original names."""
        models = []
        for filename, info in self._models_info.items():
            if info.get("model_class") != "VR_Models":
                continue

            target = info.get("target_position", "")
            if target:
                full_path = self._msst_root / target.lstrip("./")
            else:
                full_path = None

            installed = full_path and full_path.exists()

            models.append({
                "name": filename,  # Original filename e.g. "1_HP-UVR.pth"
                "installed": installed,
                "path": str(full_path) if full_path else "",
                "primary_stem": info.get("primary_stem", "Vocals"),
                "secondary_stem": info.get("secondary_stem", "Instrumental"),
            })

        return models

    async def separate(
        self,
        audio: np.ndarray,
        sr: int,
        model_name: str,
        params: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, tuple[np.ndarray, int]]:
        """Run VR separation."""
        self._ensure_init()

        # model_name is already a full filename like "1_HP-UVR.pth"
        info = self._models_info.get(model_name, {})
        target = info.get("target_position", "")
        if target:
            model_path = self._msst_root / target.lstrip("./")
        else:
            model_path = self._msst_root / "pretrain" / "VR_Models" / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"VR model not found: {model_path}")

        if progress_callback:
            await progress_callback(0.2, "Loading VR model...")

        from app.backend.config import get_device

        device = get_device()
        use_cpu = ("cpu" in str(device))

        vr_params = {
            "batch_size": 2,
            "window_size": int(params.get("window_size", 512)),
            "aggression": int(params.get("aggression", 5)),
            "enable_tta": params.get("use_tta", False),
            "enable_post_process": params.get("post_process", False),
            "post_process_threshold": 0.2,
            "high_end_process": False,
        }

        def loader():
            with _msst_cwd():
                from inference.vr_infer import VRSeparator
                return VRSeparator(
                    model_file=str(model_path),
                    use_cpu=use_cpu,
                    vr_params=vr_params,
                )

        cache_key = f"vr:{model_name}:w{vr_params['window_size']}:a{vr_params['aggression']}:tta{vr_params['enable_tta']}:pp{vr_params['enable_post_process']}"
        separator = model_manager.get_or_load(cache_key, loader)

        if progress_callback:
            await progress_callback(0.4, "Running separation...")

        # Ensure correct format
        if audio.ndim == 1:
            audio_input = np.stack([audio, audio])
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio_input = audio.T
        else:
            audio_input = audio

        def run_separate():
            with _msst_cwd():
                return separator.separate(audio_input)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_separate)

        output = {}
        target_sr = separator.sample_rate if hasattr(separator, "sample_rate") else sr
        for stem_name, stem_audio in result.items():
            if stem_audio.ndim == 2 and stem_audio.shape[0] < stem_audio.shape[1]:
                stem_audio = stem_audio.T
            output[stem_name] = (stem_audio, target_sr)

        return output


# Global instance
vr_wrapper = VRWrapper()
