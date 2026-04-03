"""Video Output Node -- saves/displays video files."""
from __future__ import annotations

import logging
import os
import shutil
from typing import Any

from app.backend.config import TEMP_DIR
from app.backend.core.node_registry import NodeRegistry
from app.backend.nodes.base import BaseNode, ExecutionContext
from app.backend.schemas.node import (
    NodeDefinition,
    ParameterDefinition,
    ParameterType,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


@NodeRegistry.register
class VideoOutputNode(BaseNode):
    """Outputs/saves a video file."""

    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="video_output",
            label="Video Output",
            category="output",
            description="Output/save a video file",
            inputs=[
                PortDefinition(name="video_path", port_type=PortType.STRING, label="Video"),
            ],
            outputs=[],
            parameters=[],
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        video_path = inputs.get("video_path")
        if not video_path:
            raise ValueError("No video input connected")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Copy to a stable location
        output_dir = str(TEMP_DIR / "video_output")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(video_path)
        dest = os.path.join(output_dir, filename)
        if os.path.abspath(video_path) != os.path.abspath(dest):
            shutil.copy2(video_path, dest)

        await context.report_progress(1.0, "Video saved")

        return {
            "video_path": dest,
            "video_name": filename,
        }
