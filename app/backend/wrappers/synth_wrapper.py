"""DiffSinger renderer wrapper.

Calls the utai-renderer C# CLI tool to render .ustx projects to WAV.
Manages singer discovery and USTX temp file handling.
"""

import collections
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from app.backend.config import PROJECT_ROOT, TEMP_DIR

logger = logging.getLogger(__name__)

# Synth paths
SYNTH_ROOT = PROJECT_ROOT / "synth"
RENDERER_EXE = SYNTH_ROOT / "utai-renderer" / "UtaiRenderer.exe"
SINGERS_DIR = SYNTH_ROOT / "Singers"

# In-memory log buffer for synth operations
SYNTH_LOG_BUFFER: collections.deque = collections.deque(maxlen=200)


class _SynthLogHandler(logging.Handler):
    """Captures synth log records into a ring buffer for /api/synth/logs."""
    def emit(self, record):
        try:
            SYNTH_LOG_BUFFER.append({
                "time": self.format(record).split(" [")[0] if " [" in self.format(record) else "",
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "formatted": self.format(record),
            })
        except Exception:
            pass


_synth_handler = _SynthLogHandler()
_synth_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.getLogger("utai.synth").addHandler(_synth_handler)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class SynthWrapper:
    """Wraps the utai-renderer CLI for .ustx -> WAV rendering."""

    def __init__(self):
        self._exe_path = RENDERER_EXE

    def _check_exe(self):
        if not self._exe_path.exists():
            raise FileNotFoundError(
                f"Renderer not found at {self._exe_path}. "
                "Place the pre-compiled utai-renderer in synth/utai-renderer/ "
                "or build from synth/utai-renderer-src/ with: dotnet build -c Release"
            )

    async def render_content(
        self,
        content: str,
        filename: str = "project.ustx",
        track: Optional[int] = None,
    ) -> dict:
        """Render USTX YAML content string to WAV."""
        synth_temp = TEMP_DIR / "synth"
        synth_temp.mkdir(parents=True, exist_ok=True)

        ustx_path = synth_temp / filename
        ustx_path.write_text(content, encoding="utf-8")

        return await self.render(str(ustx_path), track=track)

    async def render(
        self,
        ustx_path: str,
        output_path: Optional[str] = None,
        track: Optional[int] = None,
    ) -> dict:
        """Render a .ustx project file to WAV."""
        self._check_exe()

        ustx_path = Path(ustx_path).resolve()
        if not ustx_path.exists():
            return {
                "success": False,
                "output_path": None,
                "message": f"Project file not found: {ustx_path}",
            }

        synth_temp = TEMP_DIR / "synth"
        synth_temp.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = str(synth_temp / f"{ustx_path.stem}_rendered.wav")

        cmd = [str(self._exe_path), "render", str(ustx_path), output_path]
        if track is not None:
            cmd.extend(["--track", str(track)])

        synth_logger = logging.getLogger("utai.synth")
        synth_logger.info(f"Rendering: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            synth_logger.error("Render timed out (5 min)")
            return {"success": False, "output_path": None, "message": "Render timed out"}
        except Exception as e:
            synth_logger.error(f"Render subprocess error: {e}")
            return {"success": False, "output_path": None, "message": str(e)}

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Render failed"
            synth_logger.error(f"Render failed (exit {result.returncode}): {error_msg}")
            return {"success": False, "output_path": None, "message": error_msg}

        synth_logger.info(f"Render complete: {output_path}")
        return {
            "success": True,
            "output_path": output_path,
            "message": result.stdout.strip(),
        }

    async def list_singers(self) -> list[dict]:
        """List available DiffSinger voice banks via CLI."""
        self._check_exe()
        try:
            result = subprocess.run(
                [str(self._exe_path), "singers"],
                capture_output=True, text=True, timeout=30,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception as e:
            logger.error(f"List singers failed: {e}")
            return []

        singers = []
        current = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                if current:
                    singers.append(current)
                bracket_end = line.index("]")
                current = {
                    "id": line[1:bracket_end],
                    "name": line[bracket_end + 2:].strip(),
                }
            elif line.startswith("Author:") and current:
                current["author"] = line[7:].strip()
            elif line.startswith("Type:") and current:
                current["type"] = line[5:].strip()
        if current:
            singers.append(current)

        return singers

    def list_singers_from_dir(self) -> list[dict]:
        """List singers by scanning the Singers directory directly."""
        singers = []
        if not SINGERS_DIR.exists():
            return singers

        for subdir in sorted(SINGERS_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            char_txt = subdir / "character.txt"
            char_yaml = subdir / "character.yaml"
            dsconfig = subdir / "dsconfig.yaml"

            if char_txt.exists() or char_yaml.exists():
                info = {"path": str(subdir), "id": subdir.name}
                if char_yaml.exists():
                    info.update(self._parse_character_yaml(char_yaml))
                info["type"] = "diffsinger" if dsconfig.exists() else "utau"
                singers.append(info)

        return singers

    def _parse_character_yaml(self, path: Path) -> dict:
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "name": data.get("name", path.parent.name),
                "singer_type": data.get("singer_type", "unknown"),
                "phonemizer": data.get("default_phonemizer", ""),
            }
        except Exception:
            return {"name": path.parent.name}

    async def get_project_info(self, ustx_path: str) -> dict:
        """Get info about a .ustx project via CLI."""
        self._check_exe()
        try:
            result = subprocess.run(
                [str(self._exe_path), "info", ustx_path],
                capture_output=True, text=True, timeout=30,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception as e:
            return {"error": str(e)}

        info = {"raw": result.stdout}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Project:"):
                info["name"] = line[8:].strip()
            elif line.startswith("Tracks:"):
                info["track_count"] = int(line[7:].strip())
            elif line.startswith("Tempo:"):
                info["tempo"] = float(line[6:].strip().replace(" BPM", ""))
        return info


# Singleton
_wrapper: Optional[SynthWrapper] = None


def get_synth_wrapper() -> SynthWrapper:
    global _wrapper
    if _wrapper is None:
        _wrapper = SynthWrapper()
    return _wrapper
