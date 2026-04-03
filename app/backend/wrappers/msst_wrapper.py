"""Wrapper for MSST Music Source Separation."""

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


class MSSTWrapper:
    def __init__(self):
        self._msst_root = Path(MSST_ROOT)
        self._initialized = False
        self._models_info: dict[str, Any] = {}
        self._load_models_info()

    def _load_models_info(self):
        """Load models_info.json from MSST data directory."""
        info_path = self._msst_root / "data" / "models_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self._models_info = json.load(f)

    def _ensure_init(self):
        if not self._initialized:
            root = str(self._msst_root)
            if root not in sys.path:
                sys.path.insert(0, root)

            # MSST modules use relative paths (e.g. 'data_backup/webui_config.json'),
            # so we must chdir to MSST root before importing them.
            old_cwd = os.getcwd()
            os.chdir(root)

            try:
                # Patch MSST's logger: its get_logger() checks hasHandlers() and
                # returns early if our FastAPI root logger already added handlers.
                # Fix: temporarily disable propagation so hasHandlers() returns False.
                msst_logger = logging.getLogger("logger")
                if not hasattr(msst_logger, "console_handler"):
                    old_propagate = msst_logger.propagate
                    msst_logger.propagate = False
                    from utils.logger import get_logger as msst_get_logger
                    msst_get_logger()
                    msst_logger.propagate = old_propagate

                # Pre-import core MSST modules so they resolve relative paths correctly
                import inference.msst_infer  # noqa: F401
                import modules.vocal_remover.vr_separator  # noqa: F401
            except Exception as e:
                logger.warning(f"MSST init issue: {e}")
                # Ensure console_handler exists even on failure
                msst_logger = logging.getLogger("logger")
                if not hasattr(msst_logger, "console_handler"):
                    handler = logging.StreamHandler()
                    handler.setLevel(logging.INFO)
                    msst_logger.console_handler = handler
            finally:
                os.chdir(old_cwd)

            self._initialized = True

    def list_models(self) -> list[dict[str, Any]]:
        """List installed MSST models (non-VR) with their original names."""
        models = []
        for filename, info in self._models_info.items():
            model_class = info.get("model_class", "")
            if model_class == "VR_Models":
                continue  # VR models listed separately

            # Check if actually installed on disk
            target = info.get("target_position", "")
            if target:
                full_path = self._msst_root / target.lstrip("./")
            else:
                full_path = None

            installed = full_path and full_path.exists()

            models.append({
                "name": filename,  # Original model filename e.g. "HTDemucs4.th"
                "model_class": model_class,
                "model_type": info.get("model_type", ""),
                "installed": installed,
                "path": str(full_path) if full_path else "",
                "stems": self._get_model_stems(filename),
            })

        return models

    def _get_model_stems(self, model_filename: str) -> list[str]:
        """Get output stem names by finding the matching config YAML."""
        config_path = self._find_config_for_model(model_filename)
        if config_path and config_path.exists():
            return self._read_stems_from_config(config_path)

        # Fallback: infer from model_class
        info = self._models_info.get(model_filename, {})
        model_class = info.get("model_class", "")
        if model_class == "multi_stem_models":
            return ["vocals", "drums", "bass", "other"]
        return ["vocals", "instrumental"]

    def get_model_stems(self, model_filename: str) -> list[str]:
        """Public API to get stems for a model."""
        return self._get_model_stems(model_filename)

    def _find_config_for_model(self, model_filename: str) -> Path | None:
        """Find the YAML config file that matches a model filename.

        MSST stores per-model configs under:
          configs/{model_class}/{model_filename}.yaml  (exact match, highest priority)
        and generic type configs under:
          configs/config_vocals_{model_type}.yaml  (fallback)
        """
        configs_dir = self._msst_root / "configs"
        if not configs_dir.exists():
            return None

        info = self._models_info.get(model_filename, {})
        model_class = info.get("model_class", "")
        model_type = info.get("model_type", "")

        # 1) Exact per-model config: configs/{model_class}/{model_filename}.yaml
        if model_class:
            exact = configs_dir / model_class / f"{model_filename}.yaml"
            if exact.exists():
                return exact

        # 2) Search all subdirectories for {model_filename}.yaml
        for cfg in configs_dir.rglob(f"{model_filename}.yaml"):
            return cfg

        # 3) Fallback: generic config by model_type in configs/*.yaml
        if model_type:
            candidates = []
            for cfg in configs_dir.glob("*.yaml"):
                cfg_stem = cfg.stem.lower()
                if model_type in cfg_stem:
                    candidates.append(cfg)
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                candidates.sort(key=lambda c: len(c.stem))
                return candidates[0]

        return None

    def _read_stems_from_config(self, config_path: Path) -> list[str]:
        """Read instrument list from a config YAML.

        MSST's MSSeparator.separate() returns keys from config.training.instruments.
        For target_instrument models, it outputs {target: ..., other_instrument: ...}
        where other_instrument is the first item in instruments that isn't the target.
        We must return stems that exactly match those keys.
        """
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.full_load(f)  # MSST configs use !!python/tuple
            training = config.get("training", {})
            instruments = training.get("instruments", [])
            target = training.get("target_instrument")

            if target and instruments:
                # Separator returns {target: ..., other_instruments[0]: ...}
                others = [i for i in instruments if i != target]
                return [target] + others if others else [target]
            if instruments:
                return instruments
            if target:
                return [target]
        except Exception:
            pass
        return ["vocals", "instrumental"]

    async def separate(
        self,
        audio: np.ndarray,
        sr: int,
        model_name: str,
        use_tta: bool = False,
        progress_callback: Callable | None = None,
    ) -> dict[str, tuple[np.ndarray, int]]:
        """Run separation. model_name is the original filename (e.g. 'HTDemucs4.th')."""
        self._ensure_init()

        info = self._models_info.get(model_name, {})
        model_type = info.get("model_type", "")

        # Find model weights
        target = info.get("target_position", "")
        if target:
            model_path = self._msst_root / target.lstrip("./")
        else:
            model_path = self._find_model_file_fallback(model_name)

        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_name}")

        # Find config
        config_path = self._find_config_for_model(model_name)
        if not config_path or not config_path.exists():
            raise FileNotFoundError(f"Config not found for model: {model_name}")

        if progress_callback:
            await progress_callback(0.2, f"Loading {model_name}...")

        from app.backend.config import get_device

        def loader():
            with _msst_cwd():
                from inference.msst_infer import MSSeparator
                return MSSeparator(
                    model_type=model_type,
                    config_path=str(config_path),
                    model_path=str(model_path),
                    device=get_device(),
                    device_ids=[0],
                    output_format="wav",
                    use_tta=use_tta,
                )

        separator = model_manager.get_or_load(f"msst:{model_name}", loader)

        if progress_callback:
            await progress_callback(0.4, "Running separation...")

        # Convert to (channels, samples) for MSST separator
        if audio.ndim == 1:
            audio_input = np.stack([audio, audio])  # mono -> stereo
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio_input = audio.T  # (samples, channels) -> (channels, samples)
        else:
            audio_input = audio

        # Resample to model's expected sample rate if needed
        target_sr = getattr(separator.config.audio, "sample_rate", 44100)
        if sr != target_sr:
            import librosa as _lr
            # librosa.resample expects (channels, samples)
            audio_input = _lr.resample(audio_input, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        def run_separate():
            with _msst_cwd():
                return separator.separate(audio_input)

        # Run the synchronous separation in a thread pool so the event loop
        # remains free to send WebSocket progress updates.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_separate)

        if progress_callback:
            await progress_callback(0.9, "Processing outputs...")

        output = {}
        target_sr = separator.sample_rate if hasattr(separator, "sample_rate") else sr
        for stem_name, stem_audio in result.items():
            if stem_audio.ndim == 2 and stem_audio.shape[0] < stem_audio.shape[1]:
                stem_audio = stem_audio.T
            output[stem_name] = (stem_audio, target_sr)

        return output

    def _find_model_file_fallback(self, model_name: str) -> Path | None:
        """Fallback: search pretrain subdirectories."""
        pretrain_dir = self._msst_root / "pretrain"
        if not pretrain_dir.exists():
            return None
        for subdir in pretrain_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / model_name
                if candidate.exists():
                    return candidate
        return None


# Global instance
msst_wrapper = MSSTWrapper()
