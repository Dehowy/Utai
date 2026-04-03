"""LRU model cache to manage GPU memory."""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import Any, Callable

from app.backend.config import MAX_CACHED_MODELS

logger = logging.getLogger(__name__)


class ModelManager:
    """LRU cache for loaded ML models.

    Evicts the oldest model when the cache exceeds max_cached_models.
    On eviction, deletes the model and clears GPU memory.
    """

    def __init__(self, max_cached: int = MAX_CACHED_MODELS):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max = max_cached

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Get a cached model or load it. Evicts oldest if cache is full."""
        if key in self._cache:
            self._cache.move_to_end(key)
            logger.info(f"Model cache hit: {key}")
            return self._cache[key]

        # Evict if needed
        while len(self._cache) >= self._max:
            old_key, old_model = self._cache.popitem(last=False)
            logger.info(f"Evicting model: {old_key}")
            self._unload(old_model)

        logger.info(f"Loading model: {key}")
        model = loader()
        self._cache[key] = model
        return model

    def remove(self, key: str):
        if key in self._cache:
            model = self._cache.pop(key)
            self._unload(model)

    def clear(self):
        for key in list(self._cache.keys()):
            self.remove(key)

    def _unload(self, model: Any):
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def cached_keys(self) -> list[str]:
        return list(self._cache.keys())


# Global model manager instance
model_manager = ModelManager()
