"""RVC Voice Conversion node."""

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
class RVCConversionNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="rvc_conversion",
            label="RVC Voice Conversion",
            category="conversion",
            description="Convert voice using RVC models",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            parameters=[
                ParameterDefinition(
                    name="model_name",
                    label="Voice Model",
                    type=ParameterType.SELECT,
                    default="",
                    options=[],
                ),
                ParameterDefinition(
                    name="index_file",
                    label="Index File",
                    type=ParameterType.SELECT,
                    default="",
                    options=[],
                ),
                ParameterDefinition(
                    name="f0_up_key",
                    label="Pitch Shift (semitones)",
                    type=ParameterType.SLIDER,
                    default=0,
                    min=-24,
                    max=24,
                    step=1,
                ),
                ParameterDefinition(
                    name="f0_method",
                    label="F0 Method",
                    type=ParameterType.SELECT,
                    default="rmvpe",
                    options=[
                        {"value": "rmvpe", "label": "RMVPE (Recommended)"},
                        {"value": "harvest", "label": "Harvest"},
                        {"value": "crepe", "label": "Crepe"},
                        {"value": "pm", "label": "PM (Fast)"},
                    ],
                ),
                ParameterDefinition(
                    name="index_rate",
                    label="Index Rate",
                    type=ParameterType.SLIDER,
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                ParameterDefinition(
                    name="filter_radius",
                    label="Filter Radius",
                    type=ParameterType.SLIDER,
                    default=3,
                    min=0,
                    max=7,
                    step=1,
                ),
                ParameterDefinition(
                    name="rms_mix_rate",
                    label="RMS Mix Rate",
                    type=ParameterType.SLIDER,
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                ParameterDefinition(
                    name="protect",
                    label="Protect Voiceless",
                    type=ParameterType.SLIDER,
                    default=0.33,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                ),
            ],
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
            raise ValueError("No voice model selected")

        samples, sr = audio_data
        await context.report_progress(0.1, "Loading RVC model...")

        from app.backend.wrappers.rvc_wrapper import rvc_wrapper

        result_samples, result_sr = await rvc_wrapper.convert(
            samples, sr, model_name, params,
            progress_callback=context.report_progress,
        )

        await context.report_progress(1.0, "Voice conversion complete")
        return {"audio": (result_samples, result_sr)}
