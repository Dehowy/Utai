"""yt-dlp Download Node -- downloads audio/video from URLs."""
from __future__ import annotations

import logging
from typing import Any

import soundfile as sf

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
class YtdlpDownloadNode(BaseNode):
    """Downloads media from URL using yt-dlp."""

    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="ytdlp_download",
            label="yt-dlp Download",
            category="input",
            description="Download audio/video from URL using yt-dlp",
            inputs=[],
            outputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
                PortDefinition(name="video_path", port_type=PortType.STRING, label="Video"),
            ],
            parameters=[
                ParameterDefinition(name="url", label="URL", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="output_audio", label="Output Audio", type=ParameterType.CHECKBOX, default=True),
                ParameterDefinition(name="output_video", label="Output Video", type=ParameterType.CHECKBOX, default=True),
                ParameterDefinition(name="format", label="Format", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="format_sort", label="Format Sort", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="playlist_items", label="Playlist Items", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="max_downloads", label="Max Downloads", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="match_filter", label="Match Filter", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="dateafter", label="Date After", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="proxy", label="Proxy", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="limit_rate", label="Rate Limit", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="socket_timeout", label="Socket Timeout", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="source_address", label="Source Address", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="force_ipv4", label="Force IPv4", type=ParameterType.CHECKBOX, default=False),
                ParameterDefinition(name="force_ipv6", label="Force IPv6", type=ParameterType.CHECKBOX, default=False),
                ParameterDefinition(name="geo_bypass", label="Geo Bypass", type=ParameterType.CHECKBOX, default=False),
                ParameterDefinition(name="cookies_from_browser", label="Cookies From Browser", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="cookies_file", label="Cookies File", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="username", label="Username", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="password", label="Password", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="audio_format", label="Audio Format", type=ParameterType.TEXT, default="wav"),
                ParameterDefinition(name="recode_video", label="Recode Video", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="sponsorblock", label="SponsorBlock", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="no_playlist", label="No Playlist", type=ParameterType.CHECKBOX, default=False),
                ParameterDefinition(name="strip_video_audio", label="Strip Video Audio", type=ParameterType.CHECKBOX, default=False),
            ],
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        from app.backend.wrappers.ytdlp_wrapper import download_media, extract_audio_from_video

        url = params.get("url", "").strip()
        if not url:
            raise ValueError("No URL provided")

        output_audio = params.get("output_audio", True)
        output_video = params.get("output_video", True)

        options = {
            "output_audio": output_audio,
            "output_video": output_video,
            "format": params.get("format", ""),
            "format_sort": params.get("format_sort", ""),
            "playlist_items": params.get("playlist_items", ""),
            "max_downloads": params.get("max_downloads", ""),
            "match_filter": params.get("match_filter", ""),
            "dateafter": params.get("dateafter", ""),
            "proxy": params.get("proxy", ""),
            "limit_rate": params.get("limit_rate", ""),
            "socket_timeout": params.get("socket_timeout", ""),
            "source_address": params.get("source_address", ""),
            "force_ipv4": params.get("force_ipv4", False),
            "force_ipv6": params.get("force_ipv6", False),
            "geo_bypass": params.get("geo_bypass", False),
            "cookies_from_browser": params.get("cookies_from_browser", ""),
            "cookies_file": params.get("cookies_file", ""),
            "username": params.get("username", ""),
            "password": params.get("password", ""),
            "audio_format": params.get("audio_format", "wav"),
            "recode_video": params.get("recode_video", ""),
            "sponsorblock": params.get("sponsorblock", ""),
            "no_playlist": params.get("no_playlist", False),
            "strip_video_audio": params.get("strip_video_audio", False),
        }

        result = await download_media(url, options, context._progress_callback, process_tracker=context)

        outputs: dict[str, Any] = {}

        # Process audio output (only if toggled on)
        if output_audio:
            audio_path = result.get("audio_path")
            if not audio_path and result.get("video_path"):
                # Extract audio from video
                await context.report_progress(0.95, "Extracting audio...")
                audio_path = await extract_audio_from_video(
                    result["video_path"], options.get("audio_format", "wav")
                )

            if audio_path:
                try:
                    samples, sr = sf.read(audio_path, dtype="float32")
                    # sf.read returns (num_samples,) for mono or (num_samples, channels) for stereo
                    # Keep as-is — this is the standard format used throughout the app
                    outputs["audio"] = (samples, sr)
                except Exception as e:
                    logger.warning(f"Could not read audio file {audio_path}: {e}")

        # Video output (only if toggled on)
        if output_video and result.get("video_path"):
            outputs["video_path"] = result["video_path"]
            outputs["video_title"] = result.get("title", "")

        if not outputs:
            raise RuntimeError("No media was downloaded")

        return outputs
