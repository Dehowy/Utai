"""Audio Input node - loads uploaded audio files."""

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
class AudioInputNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="audio_input",
            label="Audio Input",
            category="input",
            description="Load an audio file by drag & drop or upload",
            inputs=[],
            outputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
            ],
            parameters=[
                ParameterDefinition(
                    name="buffer_id",
                    label="Uploaded File",
                    type=ParameterType.TEXT,
                    default="",
                ),
            ],
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        buffer_id = params.get("buffer_id", "")
        if not buffer_id:
            raise ValueError("No audio file uploaded")

        from app.backend.core.audio_buffer import buffer_manager

        audio = buffer_manager.get(buffer_id)
        if audio is None:
            raise ValueError(f"Audio buffer not found: {buffer_id}")

        await context.report_progress(1.0, "Audio loaded")
        return {"audio": (audio.samples, audio.sample_rate)}
