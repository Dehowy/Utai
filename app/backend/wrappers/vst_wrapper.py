"""VST instrument plugin wrapper — loads VST2/VST3 plugins via dawdreamer,
renders MIDI events to audio, manages plugin parameters and state.

All plugin loading and rendering runs in-process (dawdreamer handles native DLL
loading internally).  For crash isolation, the API layer should call these
functions inside a subprocess when working with untrusted plugins.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.backend.config import VST_INSTRUMENTS_DIR, TEMP_DIR, PROJECT_ROOT

logger = logging.getLogger("utai.vst")

# Cache loaded engines to avoid reloading plugins on every render
_engine_cache: dict[str, Any] = {}  # name → (engine, plugin)
_MAX_CACHE = 2


def _find_plugin_dll(plugin_dir: Path) -> Path | None:
    """Find the .dll or .vst3 file inside a plugin directory."""
    for ext in ("*.dll", "*.vst3"):
        files = list(plugin_dir.glob(ext))
        if files:
            return files[0]
    return None


def list_plugins() -> list[dict]:
    """List all imported VST instrument plugins."""
    plugins = []
    if not VST_INSTRUMENTS_DIR.exists():
        return plugins
    for d in sorted(VST_INSTRUMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["name"] = d.name
                plugins.append(meta)
            except Exception:
                plugins.append({"name": d.name, "error": "bad metadata"})
        else:
            dll = _find_plugin_dll(d)
            plugins.append({"name": d.name, "dll": dll.name if dll else None})
    return plugins


def validate_and_import(src_path: str, name: str | None = None) -> dict:
    """Validate a VST plugin by loading it in a subprocess, then import it.

    Returns metadata dict on success, raises on failure.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Plugin file not found: {src}")

    if not name:
        name = src.stem

    # Sanitize name
    name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not name:
        raise ValueError("Invalid plugin name")

    dest_dir = VST_INSTRUMENTS_DIR / name
    if dest_dir.exists():
        raise FileExistsError(f"Plugin '{name}' already exists")

    # Validate by loading in subprocess (crash isolation)
    result = _subprocess_validate(str(src))
    if result.get("error"):
        raise RuntimeError(result["error"])

    # Reference-based import: store metadata pointing to the original DLL path.
    # VST plugins often need adjacent resource files (samples, presets) that can be
    # gigabytes in size — copying them is impractical. Instead we load from the
    # original location and only store a lightweight metadata entry.
    dest_dir.mkdir(parents=True)

    meta = {
        "dll": src.name,
        "dll_path": str(src.resolve()),  # absolute path to load from
        "original_dir": str(src.parent.resolve()),
        "params": result.get("params", 0),
        "channels": result.get("channels", 2),
        "plugin_name": result.get("plugin_name", name),
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(f"Imported VST plugin: {name} ({src.name})")
    meta["name"] = name
    return meta


def _subprocess_validate(dll_path: str) -> dict:
    """Load a plugin in a subprocess to validate it (crash-safe)."""
    script = f"""
import json, sys
try:
    import dawdreamer as daw
    engine = daw.RenderEngine(sample_rate=44100, block_size=512)
    p = engine.make_plugin_processor("test", r"{dll_path}")
    result = {{
        "ok": True,
        "params": p.get_plugin_parameter_size(),
        "channels": p.get_num_output_channels(),
        "plugin_name": p.get_name(),
    }}
    # Quick render test
    p.add_midi_note(60, 100, 0.0, 0.5)
    engine.load_graph([(p, [])])
    engine.render(1.0)
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=15,
            cwd=str(Path(dll_path).parent),
        )
        stdout = proc.stdout.strip()
        # Find the last JSON line (skip JUCE warnings)
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                if not data.get("ok"):
                    return {"error": data.get("error", "Validation failed")}
                return data
        if proc.returncode != 0:
            return {"error": f"Plugin crashed (exit code {proc.returncode})"}
        return {"error": "No output from validation"}
    except subprocess.TimeoutExpired:
        return {"error": "Plugin loading timed out (15s)"}
    except Exception as e:
        return {"error": str(e)}


def delete_plugin(name: str) -> bool:
    """Delete an imported plugin."""
    d = VST_INSTRUMENTS_DIR / name
    if d.exists() and d.is_dir():
        shutil.rmtree(d)
        _engine_cache.pop(name, None)
        logger.info(f"Deleted VST plugin: {name}")
        return True
    return False


def get_plugin_params(name: str) -> list[dict]:
    """Get parameter descriptions for a plugin."""
    _, plugin = _load_plugin(name)
    desc = plugin.get_plugin_parameters_description()
    return [
        {
            "index": d["index"],
            "name": d["name"],
            "defaultValue": d.get("defaultValue", 0),
            "currentValue": plugin.get_parameter(d["index"]),
        }
        for d in desc
    ]


def set_plugin_param(name: str, index: int, value: float):
    """Set a parameter value on a loaded plugin."""
    _, plugin = _load_plugin(name)
    plugin.set_parameter(index, value)


def render_midi(
    name: str,
    notes: list[dict],
    bpm: float = 120,
    duration_sec: float | None = None,
    sample_rate: int = 44100,
    params: dict[int, float] | None = None,
) -> str:
    """Render MIDI notes through a VST plugin and return path to WAV file.

    notes: [{"tone": 60, "velocity": 100, "start": 0.0, "duration": 0.5}, ...]
    Returns path to rendered WAV file in temp directory.
    """
    # Always create fresh engine for rendering (cached engines have advanced
    # time position from previous renders — notes at t=0 would be in the past)
    _engine_cache.pop(name, None)
    engine, plugin = _load_plugin(name, sample_rate=sample_rate)

    # Apply parameters if provided
    if params:
        for idx, val in params.items():
            plugin.set_parameter(idx, val)

    # Clear previous MIDI and add new notes
    plugin.clear_midi()
    max_end = 0.0
    for n in notes:
        start = float(n["start"])
        dur = float(n["duration"])
        plugin.add_midi_note(int(n["tone"]), int(n.get("velocity", 100)), start, dur)
        max_end = max(max_end, start + dur)

    # Render duration: auto-detect or use provided value (+1s tail for release)
    if duration_sec is None:
        duration_sec = max_end + 2.0
    duration_sec = max(duration_sec, 0.5)

    engine.set_bpm(bpm)
    engine.load_graph([(plugin, [])])
    engine.render(duration_sec)

    audio = engine.get_audio()  # shape: (channels, samples)

    # Mix down to stereo if needed
    if audio.shape[0] > 2:
        audio = audio[:2]
    elif audio.shape[0] == 1:
        audio = np.vstack([audio, audio])

    # Save to temp file
    out_dir = TEMP_DIR / "vst_render"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{os.getpid()}.wav"
    sf.write(str(out_path), audio.T, sample_rate)

    logger.info(f"VST render: {name}, {len(notes)} notes, {duration_sec:.1f}s → {out_path}")
    return str(out_path)


def _load_plugin(name: str, sample_rate: int = 44100) -> tuple:
    """Load a plugin (with caching). Returns (engine, plugin)."""
    import dawdreamer as daw

    if name in _engine_cache:
        return _engine_cache[name]

    # Evict oldest if cache is full
    while len(_engine_cache) >= _MAX_CACHE:
        oldest = next(iter(_engine_cache))
        _engine_cache.pop(oldest)

    plugin_dir = VST_INSTRUMENTS_DIR / name
    if not plugin_dir.exists():
        raise FileNotFoundError(f"Plugin '{name}' not found")

    meta_path = plugin_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Prefer dll_path (reference-based import) over local copy
        dll_path = Path(meta["dll_path"]) if "dll_path" in meta else (plugin_dir / meta.get("dll", ""))
    else:
        dll_path = _find_plugin_dll(plugin_dir)

    if not dll_path or not dll_path.exists():
        raise FileNotFoundError(f"Plugin DLL not found: {dll_path}")

    engine = daw.RenderEngine(sample_rate=sample_rate, block_size=512)
    plugin = engine.make_plugin_processor(name, str(dll_path.resolve()))

    # Apply saved editor state (presets, parameters set via GUI)
    _apply_state_if_exists(name, plugin)

    _engine_cache[name] = (engine, plugin)
    return engine, plugin


def clear_cache():
    """Clear the plugin cache."""
    _engine_cache.clear()


# ── Editor subprocess management ──

_editor_processes: dict[str, subprocess.Popen] = {}


def open_editor(name: str) -> dict:
    """Open the VST plugin's GUI editor in a subprocess (non-blocking).
    Returns immediately; the editor window runs independently.
    """
    if name in _editor_processes:
        proc = _editor_processes[name]
        if proc.poll() is None:
            return {"status": "already_open"}
        del _editor_processes[name]

    plugin_dir = VST_INSTRUMENTS_DIR / name
    meta_path = plugin_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Plugin '{name}' not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dll_path = meta.get("dll_path", "")
    if not dll_path or not Path(dll_path).exists():
        raise FileNotFoundError(f"Plugin DLL not found: {dll_path}")

    state_path = str(plugin_dir / "state.bin")
    ext = Path(dll_path).suffix.lower()

    if ext == ".dll":
        # VST2: Python ctypes host via vst_editor.py
        editor_script = str(PROJECT_ROOT / "app" / "backend" / "wrappers" / "vst_editor.py")
        python_exe = Path(sys.executable)
        pythonw = python_exe.parent / "pythonw.exe"
        if not pythonw.exists():
            pythonw = python_exe
        cmd = [str(pythonw), editor_script, dll_path, state_path]
    else:
        # VST3: JUCE host exe
        juce_exe = PROJECT_ROOT / "VST" / "host" / "UtaiVSTHost.exe"
        if not juce_exe.exists():
            raise FileNotFoundError("JUCE VST host not found. Build VST/host/ first.")
        cmd = [str(juce_exe), dll_path, state_path]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )
    _editor_processes[name] = proc

    # Invalidate cache so next render picks up new state
    _engine_cache.pop(name, None)

    logger.info(f"Opened VST editor: {name} (pid={proc.pid})")
    return {"status": "opened", "pid": proc.pid}


def close_editor(name: str) -> dict:
    """Close a running editor subprocess."""
    proc = _editor_processes.pop(name, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Invalidate cache to pick up state changes
        _engine_cache.pop(name, None)
        logger.info(f"Closed VST editor: {name}")
        return {"status": "closed"}
    return {"status": "not_running"}


def is_editor_open(name: str) -> bool:
    """Check if an editor subprocess is running."""
    proc = _editor_processes.get(name)
    return proc is not None and proc.poll() is None


def _apply_state_if_exists(name: str, plugin):
    """Load saved editor state into a plugin if the state file exists."""
    state_path = VST_INSTRUMENTS_DIR / name / "state.bin"
    if state_path.exists():
        try:
            plugin.load_state(str(state_path))
        except Exception:
            pass
