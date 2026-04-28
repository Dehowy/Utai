"""Effects processing node — applies audio effect chains in the workflow."""

from __future__ import annotations

from typing import Any

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
class EffectsNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="effects",
            label="Effects",
            category="processing",
            description="Apply audio effects chain (pitch shift, reverb, compressor, etc.)",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            parameters=[
                ParameterDefinition(
                    name="effects",
                    type=ParameterType.TEXT,
                    label="Effects Chain (JSON)",
                    default="[]",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], params: dict[str, Any], ctx: ExecutionContext
    ) -> dict[str, Any]:
        from app.backend.wrappers.effects_wrapper import apply_effects_to_buffer

        import numpy as np

        audio_data = inputs.get("audio")
        if audio_data is None:
            raise ValueError("No audio input")

        if isinstance(audio_data, tuple):
            samples, sr = audio_data
        else:
            import soundfile as sf
            samples, sr = sf.read(audio_data, dtype="float32")

        effects = params.get("effects", [])
        if isinstance(effects, str):
            import json
            effects = json.loads(effects)

        if not effects:
            return {"audio": (samples, sr)}

        if samples.ndim == 1:
            audio = np.stack([samples, samples])
        else:
            audio = samples.T

        await ctx.report_progress(0.1, "Applying effects...")
        result = apply_effects_to_buffer(audio, effects, sr)
        await ctx.report_progress(1.0, "Done")

        return {"audio": (result.T.astype(np.float32), sr)}
