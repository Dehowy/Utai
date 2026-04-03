"""Video Input node - provides a video file path."""

from __future__ import annotations
from typing import Any

from app.backend.core.node_registry import NodeRegistry
from app.backend.nodes.base import BaseNode, ExecutionContext
from app.backend.schemas.node import (
    NodeDefinition, ParameterDefinition, ParameterType,
    PortDefinition, PortType,
)

@NodeRegistry.register
class VideoInputNode(BaseNode):
    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="video_input",
            label="Video Input",
            category="input",
            description="Load a video file",
            inputs=[],
            outputs=[
                PortDefinition(name="video_path", port_type=PortType.STRING, label="Video"),
            ],
            parameters=[
                ParameterDefinition(name="video_path", label="Video File", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="video_name", label="Video Name", type=ParameterType.TEXT, default=""),
            ],
        )

    async def execute(self, inputs: dict[str, Any], params: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        video_path = params.get("video_path", "")
        if not video_path:
            raise ValueError("No video file specified")
        import os
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        await context.report_progress(1.0, "Video loaded")
        return {"video_path": video_path}
