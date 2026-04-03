"""Wrapper for yt-dlp (run from source)."""
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from app.backend.config import TEMP_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

# yt-dlp source root (contains yt_dlp/ package)
YTDLP_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "yt-dlp"


def _find_ffmpeg() -> str:
    """Find ffmpeg executable — prefer project-local bin/ over system PATH."""
    local_ffmpeg = PROJECT_ROOT / "bin" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    # Also check yt-dlp bundled ffmpeg
    ytdlp_ffmpeg = YTDLP_ROOT / "ffmpeg.exe"
    if ytdlp_ffmpeg.exists():
        return str(ytdlp_ffmpeg)
    return "ffmpeg"  # fallback to system PATH


async def download_media(
    url: str,
    options: dict[str, Any] | None = None,
    progress_callback: Callable | None = None,
    process_tracker: Any = None,
) -> dict[str, str | None]:
    """Download media from URL using yt-dlp.

    Returns dict with keys:
      - audio_path: path to extracted audio file (or None)
      - video_path: path to video file (or None)
      - title: media title
      - thumbnail: thumbnail URL
    """
    options = options or {}
    output_dir = str(TEMP_DIR / "ytdlp_downloads")
    os.makedirs(output_dir, exist_ok=True)

    # Build yt-dlp command
    python = sys.executable
    cmd = [python, "-m", "yt_dlp"]

    # Output template
    cmd += ["-o", os.path.join(output_dir, "%(title).80s.%(ext)s")]

    # Print JSON info for metadata
    cmd += ["--print-json"]

    # ── Output mode ──
    output_audio = options.get("output_audio", True)
    output_video = options.get("output_video", True)

    fmt = options.get("format", "")
    if fmt:
        cmd += ["-f", fmt]

    if output_audio and not output_video:
        # Audio-only: extract audio
        cmd += ["-x", "--audio-format", options.get("audio_format", "wav")]

    # ── Format options ──
    if options.get("format_sort"):
        cmd += ["-S", options["format_sort"]]

    # ── Selection options ──
    if options.get("playlist_items"):
        cmd += ["--playlist-items", options["playlist_items"]]

    if options.get("max_downloads"):
        try:
            cmd += ["--max-downloads", str(int(options["max_downloads"]))]
        except (ValueError, TypeError):
            pass

    if options.get("match_filter"):
        cmd += ["--match-filter", options["match_filter"]]

    if options.get("dateafter"):
        cmd += ["--dateafter", options["dateafter"]]

    # ── Network options ──
    if options.get("proxy"):
        cmd += ["--proxy", options["proxy"]]

    if options.get("limit_rate"):
        cmd += ["--limit-rate", options["limit_rate"]]

    if options.get("socket_timeout"):
        try:
            cmd += ["--socket-timeout", str(int(options["socket_timeout"]))]
        except (ValueError, TypeError):
            pass

    if options.get("source_address"):
        cmd += ["--source-address", options["source_address"]]

    if options.get("force_ipv4"):
        cmd += ["--force-ipv4"]
    elif options.get("force_ipv6"):
        cmd += ["--force-ipv6"]

    if options.get("geo_bypass"):
        cmd += ["--geo-bypass"]

    # ── Authentication options ──
    if options.get("cookies_from_browser"):
        cmd += ["--cookies-from-browser", options["cookies_from_browser"]]

    if options.get("cookies_file"):
        cmd += ["--cookies", options["cookies_file"]]

    if options.get("username"):
        cmd += ["--username", options["username"]]

    if options.get("password"):
        cmd += ["--password", options["password"]]

    # ── Post-processing options ──
    if options.get("recode_video"):
        cmd += ["--recode-video", options["recode_video"]]

    sb = options.get("sponsorblock", "")
    if sb == "mark":
        cmd += ["--sponsorblock-mark", "all"]
    elif sb == "remove":
        cmd += ["--sponsorblock-remove", "all"]

    # ── Playlist control ──
    if options.get("no_playlist"):
        cmd += ["--no-playlist"]

    cmd.append(url)

    # Environment with yt-dlp source path
    env = os.environ.copy()
    ytdlp_parent = str(YTDLP_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ytdlp_parent}{os.pathsep}{existing}" if existing else ytdlp_parent
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if progress_callback:
        await progress_callback(0.0, "Starting download...")

    result: dict[str, str | None] = {
        "audio_path": None,
        "video_path": None,
        "title": "",
        "thumbnail": None,
    }

    # Hide console windows on Windows
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(YTDLP_ROOT),
        env=env,
        **kwargs,
    )

    # Register process for cleanup on cancellation
    if process_tracker is not None and hasattr(process_tracker, 'register_process'):
        process_tracker.register_process(process)

    stdout_data = b""
    stderr_lines: list[str] = []

    async def read_stderr():
        assert process.stderr is not None
        async for line in process.stderr:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                stderr_lines.append(text)
                # Parse progress
                pct_match = re.search(r"\[download\]\s+([\d.]+)%", text)
                if pct_match and progress_callback:
                    pct = float(pct_match.group(1)) / 100
                    await progress_callback(pct * 0.9, f"Downloading... {pct_match.group(1)}%")

    stderr_task = asyncio.create_task(read_stderr())
    assert process.stdout is not None
    stdout_data = await process.stdout.read()
    await stderr_task
    await process.wait()

    if process.returncode != 0:
        err_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
        raise RuntimeError(f"yt-dlp failed (code {process.returncode}): {err_msg}")

    # Parse JSON output for metadata
    stdout_text = stdout_data.decode("utf-8", errors="replace").strip()
    if stdout_text:
        try:
            # yt-dlp may output multiple JSON objects (one per format)
            for line in stdout_text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    info = json.loads(line)
                    result["title"] = info.get("title", "")
                    result["thumbnail"] = info.get("thumbnail")
                    # Find the actual downloaded file
                    filename = info.get("_filename") or info.get("filename", "")
                    if filename and os.path.exists(filename):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"):
                            result["audio_path"] = filename
                        else:
                            result["video_path"] = filename
                    break
        except json.JSONDecodeError:
            pass

    # If we didn't get paths from JSON, scan the output dir for recent files
    if not result["audio_path"] and not result["video_path"]:
        files = sorted(Path(output_dir).glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:5]:
            if f.is_file():
                ext = f.suffix.lower()
                if ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"):
                    if not result["audio_path"]:
                        result["audio_path"] = str(f)
                elif ext in (".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"):
                    if not result["video_path"]:
                        result["video_path"] = str(f)

    # Strip audio track from video if requested
    if options.get("strip_video_audio") and result.get("video_path"):
        if progress_callback:
            await progress_callback(0.95, "Stripping audio from video...")
        stripped = await strip_audio_from_video(result["video_path"])
        if stripped:
            result["video_path"] = stripped

    if progress_callback:
        await progress_callback(1.0, "Download complete")

    return result


async def strip_audio_from_video(video_path: str) -> str | None:
    """Remove audio track from video, keeping video stream only."""
    p = Path(video_path)
    output_path = str(p.with_stem(p.stem + ".noaudio"))

    cmd = [
        _find_ffmpeg(), "-y", "-i", video_path,
        "-an", "-c:v", "copy",
        output_path,
    ]
    _kw: dict = {}
    if sys.platform == "win32":
        _kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **_kw,
        )
        await proc.wait()
        if proc.returncode == 0 and os.path.exists(output_path):
            return output_path
    except FileNotFoundError:
        pass
    return None


async def extract_audio_from_video(video_path: str, output_format: str = "wav") -> str | None:
    """Extract audio from a video file using ffmpeg."""
    output_path = str(Path(video_path).with_suffix(f".audio.{output_format}"))

    # Use ffmpeg directly (more reliable for local files)
    ffmpeg_cmd = [
        _find_ffmpeg(), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
        output_path,
    ]
    _kw2: dict = {}
    if sys.platform == "win32":
        _kw2["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **_kw2,
        )
        await proc.wait()
        if proc.returncode == 0 and os.path.exists(output_path):
            return output_path
    except FileNotFoundError:
        pass

    return None
