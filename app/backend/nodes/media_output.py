"""Media Output Node — unified audio/video output via ffmpeg.

Accepts audio input, video input, or both. When both are provided,
replaces the video's audio track with the incoming audio and muxes
into the selected container format.

All encoding parameters map directly to ffmpeg options, making this
node a visual ffmpeg frontend.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.backend.config import TEMP_DIR, PROJECT_ROOT
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

# Audio-only containers (no video stream expected)
AUDIO_CONTAINERS = {"wav", "flac", "mp3", "ogg", "m4a", "opus", "aac", "wma"}

# Default audio codec per container when set to "auto"
AUTO_AUDIO_CODEC = {
    "mp4": "aac",
    "mkv": "aac",
    "webm": "libopus",
    "avi": "libmp3lame",
    "mov": "aac",
    "flv": "aac",
    "ts": "aac",
    "wav": "pcm_s16le",
    "flac": "flac",
    "mp3": "libmp3lame",
    "ogg": "libvorbis",
    "m4a": "aac",
    "opus": "libopus",
    "aac": "aac",
    "wma": "wmav2",
}

# Default video codec per container when set to "auto"
AUTO_VIDEO_CODEC = {
    "mp4": "libx264",
    "mkv": "libx264",
    "webm": "libvpx-vp9",
    "avi": "libx264",
    "mov": "libx264",
    "flv": "libx264",
    "ts": "libx264",
}


def _find_ffmpeg() -> str:
    """Find ffmpeg — prefer project-local bin/ over system PATH."""
    local = PROJECT_ROOT / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    return "ffmpeg"


@NodeRegistry.register
class MediaOutputNode(BaseNode):
    """Unified media output node — visual ffmpeg interface."""

    @classmethod
    def definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type="media_output",
            label="Media Output",
            category="output",
            description="Export audio/video with full encoding control (ffmpeg)",
            inputs=[
                PortDefinition(name="audio", port_type=PortType.AUDIO, label="Audio"),
                PortDefinition(name="video_path", port_type=PortType.STRING, label="Video"),
            ],
            outputs=[],
            parameters=[
                # Container
                ParameterDefinition(name="container", label="Container", type=ParameterType.SELECT, default="mp4",
                                    options=[
                                        {"value": "mp4", "label": "MP4"},
                                        {"value": "mkv", "label": "MKV"},
                                        {"value": "webm", "label": "WebM"},
                                        {"value": "avi", "label": "AVI"},
                                        {"value": "mov", "label": "MOV"},
                                        {"value": "wav", "label": "WAV"},
                                        {"value": "flac", "label": "FLAC"},
                                        {"value": "mp3", "label": "MP3"},
                                        {"value": "ogg", "label": "OGG"},
                                        {"value": "m4a", "label": "M4A"},
                                        {"value": "opus", "label": "Opus"},
                                    ]),
                # Audio
                ParameterDefinition(name="audio_codec", label="Audio Codec", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="audio_bitrate", label="Audio Bitrate", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="sample_rate", label="Sample Rate", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="audio_channels", label="Audio Channels", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="volume", label="Volume", type=ParameterType.TEXT, default=""),
                # Video
                ParameterDefinition(name="video_codec", label="Video Codec", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="video_bitrate", label="Video Bitrate", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="crf", label="CRF", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="preset", label="Preset", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="resolution", label="Resolution", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="fps", label="FPS", type=ParameterType.TEXT, default="auto"),
                ParameterDefinition(name="pix_fmt", label="Pixel Format", type=ParameterType.TEXT, default="auto"),
                # Filters
                ParameterDefinition(name="audio_filter", label="Audio Filter", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="video_filter", label="Video Filter", type=ParameterType.TEXT, default=""),
                # Advanced
                ParameterDefinition(name="start_time", label="Start Time", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="duration", label="Duration", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="metadata", label="Metadata", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="extra_args", label="Extra Args", type=ParameterType.TEXT, default=""),
                ParameterDefinition(name="overwrite", label="Overwrite", type=ParameterType.CHECKBOX, default=True),
            ],
        )

    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        audio_data = inputs.get("audio")     # (samples, sr) tuple or None
        video_path = inputs.get("video_path")  # str path or None

        if audio_data is None and not video_path:
            raise ValueError("No audio or video input connected")

        container = params.get("container", "mp4")
        is_audio_only = container in AUDIO_CONTAINERS

        # ── Prepare output directory ──
        output_dir = TEMP_DIR / "media_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Write audio to temp WAV if present ──
        audio_tmp_path = None
        if audio_data is not None:
            samples, sr = audio_data
            audio_tmp_path = str(output_dir / f"_tmp_audio_{os.getpid()}.wav")
            sf.write(audio_tmp_path, samples, sr, format="WAV")

        # ── Build ffmpeg command ──
        ffmpeg = _find_ffmpeg()
        cmd: list[str] = [ffmpeg]

        # Overwrite
        if params.get("overwrite", True):
            cmd.append("-y")

        # Start time / duration (global input options)
        if params.get("start_time"):
            cmd += ["-ss", params["start_time"]]
        if params.get("duration"):
            cmd += ["-t", params["duration"]]

        # ── Input streams ──
        # Video input first (if present), then audio
        input_idx_video = -1
        input_idx_audio = -1

        if video_path and not is_audio_only:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            cmd += ["-i", video_path]
            input_idx_video = 0

        if audio_tmp_path:
            cmd += ["-i", audio_tmp_path]
            input_idx_audio = 1 if input_idx_video >= 0 else 0
        elif audio_data is None and video_path and not is_audio_only:
            # Video only — keep its audio track
            pass

        # ── Stream mapping ──
        if input_idx_video >= 0 and input_idx_audio >= 0:
            # Video + replacement audio: take video from video input, audio from audio input
            cmd += ["-map", f"{input_idx_video}:v", "-map", f"{input_idx_audio}:a"]
        elif input_idx_video >= 0:
            # Video only — keep all streams
            cmd += ["-map", f"{input_idx_video}"]
        elif input_idx_audio >= 0:
            # Audio only
            cmd += ["-map", f"{input_idx_audio}:a"]

        # ── Video codec ──
        if input_idx_video >= 0 and not is_audio_only:
            vc = params.get("video_codec", "auto")
            if vc == "auto":
                vc = AUTO_VIDEO_CODEC.get(container, "libx264")
            cmd += ["-c:v", vc]

            if vc != "copy":
                # Video bitrate
                vb = params.get("video_bitrate", "")
                if vb:
                    cmd += ["-b:v", vb]

                # CRF
                crf = params.get("crf", "")
                if crf:
                    cmd += ["-crf", crf]

                # Preset
                preset = params.get("preset", "auto")
                if preset and preset != "auto":
                    cmd += ["-preset", preset]

                # Resolution (scale filter)
                resolution = params.get("resolution", "auto")
                if resolution and resolution != "auto":
                    # Add as video filter
                    scale_filter = f"scale={resolution}"
                    existing_vf = params.get("video_filter", "")
                    if existing_vf:
                        params["video_filter"] = f"{scale_filter},{existing_vf}"
                    else:
                        params["video_filter"] = scale_filter

                # FPS
                fps = params.get("fps", "auto")
                if fps and fps != "auto":
                    cmd += ["-r", fps]

                # Pixel format
                pix_fmt = params.get("pix_fmt", "auto")
                if pix_fmt and pix_fmt != "auto":
                    cmd += ["-pix_fmt", pix_fmt]

        # ── Audio codec ──
        has_audio = input_idx_audio >= 0 or (input_idx_video >= 0 and input_idx_audio < 0)
        if has_audio:
            ac = params.get("audio_codec", "auto")
            if ac == "auto":
                ac = AUTO_AUDIO_CODEC.get(container, "aac")
            cmd += ["-c:a", ac]

            if ac != "copy":
                # Audio bitrate
                ab = params.get("audio_bitrate", "")
                if ab:
                    cmd += ["-b:a", ab]

                # Sample rate
                sr_param = params.get("sample_rate", "auto")
                if sr_param and sr_param != "auto":
                    cmd += ["-ar", sr_param]

                # Channels
                channels = params.get("audio_channels", "auto")
                if channels and channels != "auto":
                    cmd += ["-ac", channels]

        # ── Filters ──
        af_parts = []
        # Volume adjustment
        vol = params.get("volume", "")
        if vol:
            af_parts.append(f"volume={vol}")
        # User audio filter
        user_af = params.get("audio_filter", "")
        if user_af:
            af_parts.append(user_af)
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]

        # Video filter
        vf = params.get("video_filter", "")
        if vf:
            cmd += ["-vf", vf]

        # ── Metadata ──
        meta = params.get("metadata", "")
        if meta:
            for pair in meta.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    cmd += ["-metadata", pair]

        # ── Extra args (raw ffmpeg flags) ──
        extra = params.get("extra_args", "")
        if extra:
            cmd += shlex.split(extra)

        # ── Output file ──
        # Generate output filename
        if video_path and not is_audio_only:
            base_name = Path(video_path).stem
        elif audio_tmp_path:
            base_name = "output"
        else:
            base_name = "output"

        output_filename = f"{base_name}_out.{container}"
        output_path = str(output_dir / output_filename)
        cmd.append(output_path)

        # ── Execute ffmpeg ──
        await context.report_progress(0.1, "Running ffmpeg...")
        logger.info(f"ffmpeg command: {' '.join(cmd)}")

        import sys as _sys
        _kw: dict = {}
        if _sys.platform == "win32":
            _kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_kw,
            )

            _, stderr_data = await process.communicate()
            stderr_text = stderr_data.decode("utf-8", errors="replace")

            if process.returncode != 0:
                # Log last 10 lines of stderr
                err_lines = stderr_text.strip().split("\n")[-10:]
                err_msg = "\n".join(err_lines)
                raise RuntimeError(f"ffmpeg failed (code {process.returncode}):\n{err_msg}")

        finally:
            # Clean up temp audio file
            if audio_tmp_path and os.path.exists(audio_tmp_path):
                try:
                    os.unlink(audio_tmp_path)
                except OSError:
                    pass

        if not os.path.exists(output_path):
            raise RuntimeError("ffmpeg produced no output file")

        await context.report_progress(0.95, "Processing output...")

        # ── Build outputs ──
        outputs: dict[str, Any] = {}

        if is_audio_only:
            # Read back as audio buffer for preview/download
            try:
                out_samples, out_sr = sf.read(output_path, dtype="float32")
                outputs["audio"] = (out_samples, out_sr)
            except Exception as e:
                logger.warning(f"Could not read output audio {output_path}: {e}")
        else:
            # Video output — copy to stable location for serving
            stable_dir = TEMP_DIR / "video_output"
            stable_dir.mkdir(parents=True, exist_ok=True)
            stable_path = str(stable_dir / output_filename)
            if os.path.abspath(output_path) != os.path.abspath(stable_path):
                import shutil
                shutil.move(output_path, stable_path)
            else:
                stable_path = output_path
            outputs["video_path"] = stable_path
            outputs["video_name"] = output_filename

            # Also provide audio preview if we had audio input
            if audio_data is not None:
                samples, sr = audio_data
                outputs["audio"] = (samples, sr)

        await context.report_progress(1.0, "Done")
        return outputs
