"""Audio Output node - makes audio available for playback and download."""

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
class AudioOutputNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="audio_output",
            label="Audio Output",
            category="output",
            description="Play and download processed audio",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            outputs=[],
            parameters=[
                ParameterDefinition(
                    name="format",
                    label="Output Format",
                    type=ParameterType.SELECT,
                    default="wav",
                    options=[
                        {"value": "wav", "label": "WAV"},
                        {"value": "flac", "label": "FLAC"},
                        {"value": "mp3", "label": "MP3"},
                    ],
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

        samples, sr = audio_data
        await context.report_progress(1.0, "Output ready")
        # The executor will detect this is audio and store it in the buffer manager
        return {"audio": (samples, sr)}
