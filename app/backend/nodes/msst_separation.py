"""MSST Music Source Separation node."""

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
class MSSTSeparationNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="msst_separation",
            label="MSST Separation",
            category="separation",
            description="Separate audio into stems using MSST models",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[
                # Default outputs - will be dynamically updated based on model
                PortDefinition(name="vocals", port_type=PortType.AUDIO, label="Vocals"),
                PortDefinition(name="instrumental", port_type=PortType.AUDIO, label="Instrumental"),
            ],
            parameters=[
                ParameterDefinition(
                    name="model_name",
                    label="Model",
                    type=ParameterType.SELECT,
                    default="",
                    options=[],  # Populated dynamically via API
                ),
                ParameterDefinition(
                    name="use_tta",
                    label="Test-Time Augmentation",
                    type=ParameterType.CHECKBOX,
                    default=False,
                ),
            ],
            dynamic_outputs=True,
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        audio_data = inputs.get("audio")
        if audio_data is None:
            raise ValueError("No audio input connected")

        model_name = params.get("model_name", "")
        if not model_name:
            raise ValueError("No model selected")

        samples, sr = audio_data
        use_tta = params.get("use_tta", False)

        await context.report_progress(0.1, "Loading model...")

        from app.backend.wrappers.msst_wrapper import msst_wrapper

        results = await msst_wrapper.separate(
            samples, sr, model_name, use_tta=use_tta,
            progress_callback=context.report_progress,
        )

        # results is {stem_name: (ndarray, sr)}
        outputs = {}
        for stem_name, (stem_audio, stem_sr) in results.items():
            outputs[stem_name] = (stem_audio, stem_sr)

        await context.report_progress(1.0, "Separation complete")
        return outputs
