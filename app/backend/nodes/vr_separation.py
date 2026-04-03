"""VR (Vocal Remover) Separation node."""

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
class VRSeparationNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="vr_separation",
            label="VR Separation",
            category="separation",
            description="Separate audio using UVR Vocal Remover models",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[
                PortDefinition(name="primary", port_type=PortType.AUDIO, label="Primary Stem"),
                PortDefinition(name="secondary", port_type=PortType.AUDIO, label="Secondary Stem"),
            ],
            parameters=[
                ParameterDefinition(
                    name="model_name",
                    label="Model",
                    type=ParameterType.SELECT,
                    default="",
                    options=[],
                ),
                ParameterDefinition(
                    name="window_size",
                    label="Window Size",
                    type=ParameterType.SELECT,
                    default="512",
                    options=[
                        {"value": "320", "label": "320"},
                        {"value": "512", "label": "512 (Default)"},
                        {"value": "1024", "label": "1024"},
                    ],
                ),
                ParameterDefinition(
                    name="aggression",
                    label="Aggression",
                    type=ParameterType.SLIDER,
                    default=5,
                    min=0,
                    max=50,
                    step=1,
                ),
                ParameterDefinition(
                    name="use_tta",
                    label="Test-Time Augmentation",
                    type=ParameterType.CHECKBOX,
                    default=False,
                ),
                ParameterDefinition(
                    name="post_process",
                    label="Post Process",
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
        await context.report_progress(0.1, "Loading model...")

        from app.backend.wrappers.vr_wrapper import vr_wrapper

        results = await vr_wrapper.separate(
            samples, sr, model_name, params,
            progress_callback=context.report_progress,
        )

        # VR separator returns keys like "Vocals"/"Instrumental" (from model_data).
        # Map to port names "primary"/"secondary" so the executor can route data,
        # while also keeping original stem names for dynamic_outputs consumers.
        outputs: dict[str, Any] = {}
        keys = list(results.keys())
        if len(keys) >= 1:
            outputs["primary"] = results[keys[0]]
            outputs[keys[0]] = results[keys[0]]
        if len(keys) >= 2:
            outputs["secondary"] = results[keys[1]]
            outputs[keys[1]] = results[keys[1]]

        await context.report_progress(1.0, "Separation complete")
        return outputs
