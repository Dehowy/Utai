"""Unified separation node — dispatches to MSST or VR wrapper based on model_class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.backend.config import MSST_ROOT
from app.backend.core.node_registry import NodeRegistry
from app.backend.nodes.base import BaseNode, ExecutionContext
from app.backend.schemas.node import (
    NodeDefinition,
    ParameterDefinition,
    ParameterType,
    PortDefinition,
    PortType,
)

_msst_root = Path(MSST_ROOT)


def _get_model_class(model_name: str) -> str:
    """Look up model_class from models_info.json."""
    info_path = _msst_root / "data" / "models_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            models_info = json.load(f)
        info = models_info.get(model_name, {})
        return info.get("model_class", "")
    return ""


@NodeRegistry.register
class SeparationNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="separation",
            label="Separation",
            category="separation",
            description="Separate audio using MSST or VR models (category-based)",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[
                PortDefinition(name="vocals", port_type=PortType.AUDIO, label="Vocals"),
                PortDefinition(
                    name="instrumental", port_type=PortType.AUDIO, label="Instrumental"
                ),
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
                    name="category",
                    label="Category",
                    type=ParameterType.TEXT,
                    default="",
                ),
                ParameterDefinition(
                    name="use_tta",
                    label="TTA",
                    type=ParameterType.CHECKBOX,
                    default=False,
                ),
                # VR-specific params (only used if model is VR class)
                ParameterDefinition(
                    name="window_size",
                    label="Window Size",
                    type=ParameterType.SELECT,
                    default="512",
                    options=[
                        {"value": "320", "label": "320"},
                        {"value": "512", "label": "512"},
                        {"value": "1024", "label": "1024"},
                    ],
                ),
                ParameterDefinition(
                    name="aggression",
                    label="Aggression",
                    type=ParameterType.INT,
                    default=5,
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
        use_tta = params.get("use_tta", False)
        model_class = _get_model_class(model_name)

        await context.report_progress(0.1, "Loading model...")

        if model_class == "VR_Models":
            from app.backend.wrappers.vr_wrapper import vr_wrapper

            results = await vr_wrapper.separate(
                samples,
                sr,
                model_name,
                params={
                    "use_tta": use_tta,
                    "window_size": int(params.get("window_size", 512)),
                    "aggression": int(params.get("aggression", 5)),
                    "post_process": params.get("post_process", False),
                },
                progress_callback=context.report_progress,
            )
        else:
            from app.backend.wrappers.msst_wrapper import msst_wrapper

            results = await msst_wrapper.separate(
                samples,
                sr,
                model_name,
                use_tta=use_tta,
                progress_callback=context.report_progress,
            )

        outputs = {}
        for stem_name, (stem_audio, stem_sr) in results.items():
            outputs[stem_name] = (stem_audio, stem_sr)

        # Also map to default port names ("vocals"/"instrumental") for VR models
        # whose stem names (e.g. "Vocals"/"Instrumental") differ from the port definitions.
        if model_class == "VR_Models":
            for key, value in results.items():
                key_lower = key.lower()
                if "vocal" in key_lower and "vocals" not in outputs:
                    outputs["vocals"] = value
                elif "instrument" in key_lower and "instrumental" not in outputs:
                    outputs["instrumental"] = value

        await context.report_progress(1.0, "Separation complete")
        return outputs
