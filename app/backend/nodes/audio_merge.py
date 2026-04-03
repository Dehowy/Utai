"""Audio Merge node - mix multiple audio tracks into one."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.backend.core.node_registry import NodeRegistry
from app.backend.nodes.base import BaseNode, ExecutionContext
from app.backend.schemas.node import (
    NodeDefinition,
    ParameterDefinition,
    ParameterType,
    PortDefinition,
    PortType,
)


@NodeRegistry.register
class AudioMergeNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="audio_merge",
            label="Audio Merge",
            category="processing",
            description="Mix multiple audio tracks into a single output",
            # Dynamic inputs — the frontend controls how many tracks are shown.
            # The backend accepts audio_1..audio_N via params["track_count"].
            # We define a few default inputs; extras are handled dynamically.
            inputs=[
                PortDefinition(name="audio_1", port_type=PortType.AUDIO, label="Track 1", optional=True),
                PortDefinition(name="audio_2", port_type=PortType.AUDIO, label="Track 2", optional=True),
            ],
            outputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Mixed Audio"),
            ],
            parameters=[
                ParameterDefinition(
                    name="track_count",
                    label="Track Count",
                    type=ParameterType.INT,
                    default=2,
                    min=2,
                    max=8,
                    step=1,
                ),
            ],
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        track_count = int(params.get("track_count", 2))
        tracks: list[tuple[np.ndarray, int]] = []

        for i in range(1, track_count + 1):
            key = f"audio_{i}"
            data = inputs.get(key)
            if data is not None and isinstance(data, tuple) and len(data) == 2:
                samples, sr = data
                if isinstance(samples, np.ndarray):
                    tracks.append((samples, sr))

        if not tracks:
            raise ValueError("No audio tracks connected")

        await context.report_progress(0.1, f"Mixing {len(tracks)} tracks...")

        # Use the highest sample rate among inputs
        target_sr = max(sr for _, sr in tracks)

        # Resample all tracks to target_sr and find max length
        resampled: list[np.ndarray] = []
        for samples, sr in tracks:
            if sr != target_sr:
                try:
                    import soxr
                    samples = soxr.resample(samples, sr, target_sr)
                except ImportError:
                    # Fallback: simple linear interpolation
                    import scipy.signal
                    n_samples = int(len(samples) * target_sr / sr)
                    samples = scipy.signal.resample(samples, n_samples).astype(np.float32)
            # Ensure mono or consistent channel count
            if samples.ndim == 1:
                resampled.append(samples)
            else:
                resampled.append(samples)

        # Pad shorter tracks with silence to match longest
        max_len = max(len(s) if s.ndim == 1 else s.shape[0] for s in resampled)
        padded: list[np.ndarray] = []
        for s in resampled:
            if s.ndim == 1:
                if len(s) < max_len:
                    s = np.pad(s, (0, max_len - len(s)))
                padded.append(s)
            else:
                if s.shape[0] < max_len:
                    pad_shape = list(s.shape)
                    pad_shape[0] = max_len - s.shape[0]
                    s = np.concatenate([s, np.zeros(pad_shape, dtype=s.dtype)])
                padded.append(s)

        await context.report_progress(0.5, "Summing tracks...")

        # Mix by summing all tracks
        mixed = np.zeros_like(padded[0], dtype=np.float64)
        for s in padded:
            # Handle mono + stereo mixing: broadcast mono to match stereo
            if mixed.ndim != s.ndim:
                if s.ndim == 1 and mixed.ndim == 2:
                    s = np.stack([s] * mixed.shape[1], axis=-1)
                elif mixed.ndim == 1 and s.ndim == 2:
                    mixed = np.stack([mixed] * s.shape[1], axis=-1)
            mixed = mixed + s.astype(np.float64)

        # Normalize to prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak

        mixed = mixed.astype(np.float32)

        await context.report_progress(1.0, f"Mixed {len(tracks)} tracks")
        return {"audio": (mixed, target_sr)}
