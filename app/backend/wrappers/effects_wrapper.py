"""Audio effects processing engine — combines dawdreamer processors + WORLD vocoder.

Supports chaining multiple effects on an audio buffer:
  - world_pitch: WORLD vocoder pitch shift (vocal, no chipmunk)
  - warp_pitch:  PlaybackWarp pitch/speed (general audio)
  - compressor:  Dynamic range compression
  - reverb:      Room reverb
  - filter:      High/low/band-pass filter
  - delay:       Echo/delay
  - panner:      Stereo panning
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.backend.config import TEMP_DIR

logger = logging.getLogger("utai.effects")

# Effect type definitions with default parameters
EFFECT_DEFS: dict[str, dict[str, Any]] = {
    "world_pitch": {
        "label": "WORLD Pitch/Speed",
        "category": "pitch",
        "params": {
            "semitones": {"default": 0, "min": -24, "max": 24, "step": 1, "label": "Semitones"},
            "formant_shift": {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "label": "Formant"},
            "speed": {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "label": "Speed"},
        },
    },
    "warp_pitch": {
        "label": "Pitch / Speed",
        "category": "pitch",
        "params": {
            "semitones": {"default": 0, "min": -24, "max": 24, "step": 1, "label": "Semitones"},
            "speed": {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "label": "Speed"},
        },
    },
    "compressor": {
        "label": "Compressor",
        "category": "dynamics",
        "params": {
            "threshold": {"default": -10, "min": -60, "max": 0, "step": 1, "label": "Threshold (dB)"},
            "ratio": {"default": 4, "min": 1, "max": 20, "step": 0.5, "label": "Ratio"},
            "attack": {"default": 5, "min": 0.1, "max": 100, "step": 0.1, "label": "Attack (ms)"},
            "release": {"default": 100, "min": 5, "max": 2000, "step": 5, "label": "Release (ms)"},
        },
    },
    "reverb": {
        "label": "Reverb",
        "category": "space",
        "params": {
            "room_size": {"default": 0.5, "min": 0, "max": 1, "step": 0.01, "label": "Room Size"},
            "damping": {"default": 0.5, "min": 0, "max": 1, "step": 0.01, "label": "Damping"},
            "wet": {"default": 0.3, "min": 0, "max": 1, "step": 0.01, "label": "Wet"},
            "dry": {"default": 0.7, "min": 0, "max": 1, "step": 0.01, "label": "Dry"},
            "width": {"default": 1.0, "min": 0, "max": 1, "step": 0.01, "label": "Width"},
        },
    },
    "filter": {
        "label": "Filter",
        "category": "eq",
        "params": {
            "mode": {"default": 0, "min": 0, "max": 2, "step": 1, "label": "Mode (0=low,1=high,2=band)"},
            "frequency": {"default": 1000, "min": 20, "max": 20000, "step": 10, "label": "Frequency (Hz)"},
            "q": {"default": 0.707, "min": 0.1, "max": 10, "step": 0.01, "label": "Q"},
            "gain": {"default": 1.0, "min": 0, "max": 4, "step": 0.1, "label": "Gain"},
        },
    },
    "delay": {
        "label": "Delay",
        "category": "space",
        "params": {
            "time_ms": {"default": 250, "min": 1, "max": 2000, "step": 1, "label": "Time (ms)"},
            "wet": {"default": 0.3, "min": 0, "max": 1, "step": 0.01, "label": "Wet"},
        },
    },
    "panner": {
        "label": "Panner",
        "category": "spatial",
        "params": {
            "pan": {"default": 0, "min": -1, "max": 1, "step": 0.01, "label": "Pan (L-R)"},
        },
    },
}

FILTER_MODES = {0: "low", 1: "high", 2: "bandpass"}


def get_effect_defs() -> dict:
    """Return all effect type definitions."""
    return EFFECT_DEFS


def apply_effects(
    audio_path: str,
    effects: list[dict],
    sample_rate: int = 44100,
) -> str:
    """Apply a chain of effects to an audio file.

    effects: [{"type": "reverb", "params": {"room_size": 0.7, ...}}, ...]
    Returns path to processed WAV file.
    """
    audio, sr = sf.read(audio_path, dtype="float64")
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    else:
        audio = audio.T  # (samples, ch) → (ch, samples)
    sample_rate = sr

    for fx in effects:
        fx_type = fx.get("type", "")
        params = fx.get("params", {})
        enabled = fx.get("enabled", True)
        if not enabled:
            continue

        if fx_type == "world_pitch":
            audio = _apply_world_pitch(audio, sr, params)
        elif fx_type == "warp_pitch":
            audio = _apply_warp_pitch(audio, sr, params)
        elif fx_type == "delay":
            audio = _apply_delay(audio, sr, params)
        elif fx_type == "reverb":
            audio = _apply_reverb(audio, sr, params)
        elif fx_type in ("compressor", "filter", "panner"):
            audio = _apply_dawdreamer_effect(audio, sr, fx_type, params)
        else:
            logger.warning(f"Unknown effect type: {fx_type}")

    # Save result
    out_dir = TEMP_DIR / "effects"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fx_{os.getpid()}_{id(audio)}.wav"
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    sf.write(str(out_path), audio.T, sr)
    logger.info(f"Applied {len(effects)} effects → {out_path}")
    return str(out_path)


def apply_effects_to_buffer(
    audio: np.ndarray,
    effects: list[dict],
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply effects to an in-memory audio array (channels, samples). Returns processed array."""
    for fx in effects:
        fx_type = fx.get("type", "")
        params = fx.get("params", {})
        enabled = fx.get("enabled", True)
        if not enabled:
            continue
        if fx_type == "world_pitch":
            audio = _apply_world_pitch(audio, sample_rate, params)
        elif fx_type == "warp_pitch":
            audio = _apply_warp_pitch(audio, sample_rate, params)
        elif fx_type == "delay":
            audio = _apply_delay(audio, sample_rate, params)
        elif fx_type == "reverb":
            audio = _apply_reverb(audio, sample_rate, params)
        elif fx_type in ("compressor", "filter", "panner"):
            audio = _apply_dawdreamer_effect(audio, sample_rate, fx_type, params)
    return audio


# ── WORLD vocoder pitch shift ──

def _apply_world_pitch(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """WORLD vocoder pitch/speed shift — vocal-specific, no chipmunk effect."""
    import pyworld as pw

    semitones = float(params.get("semitones", 0))
    formant_shift = float(params.get("formant_shift", 1.0))
    speed = float(params.get("speed", 1.0))

    if semitones == 0 and formant_shift == 1.0 and speed == 1.0:
        return audio

    mono = audio[0].astype(np.float64) if audio.shape[0] >= 1 else audio.astype(np.float64)

    f0, sp, ap = pw.wav2world(mono, sr)

    if semitones != 0:
        f0_new = f0.copy()
        f0_new[f0_new > 0] *= 2 ** (semitones / 12)
    else:
        f0_new = f0

    if formant_shift != 1.0:
        sp_new = np.zeros_like(sp)
        freq_axis = np.arange(sp.shape[1], dtype=np.float64)
        for i in range(sp.shape[0]):
            sp_new[i] = np.interp(freq_axis, freq_axis / formant_shift, sp[i])
    else:
        sp_new = sp

    # Time-stretch: resample frames to change duration without affecting pitch
    if speed != 1.0:
        n_old = len(f0_new)
        n_new = max(1, int(round(n_old / speed)))
        old_t = np.linspace(0, 1, n_old)
        new_t = np.linspace(0, 1, n_new)
        f0_new = np.interp(new_t, old_t, f0_new)
        sp_new = np.array([np.interp(new_t, old_t, sp_new[:, j]) for j in range(sp_new.shape[1])]).T
        ap = np.array([np.interp(new_t, old_t, ap[:, j]) for j in range(ap.shape[1])]).T

    y = pw.synthesize(
        np.ascontiguousarray(f0_new),
        np.ascontiguousarray(sp_new),
        np.ascontiguousarray(ap),
        sr,
    )
    y = y.astype(np.float32)

    return np.stack([y, y])


# ── PlaybackWarp pitch/speed ──

def _apply_warp_pitch(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Pitch/speed shift. Pure speed change uses fast resampling; pitch shift uses dawdreamer."""
    semitones = float(params.get("semitones", 0))
    speed = float(params.get("speed", 1.0))

    if semitones == 0 and speed == 1.0:
        return audio

    if semitones == 0:
        try:
            import soxr
            resampled = np.stack([
                soxr.resample(audio[ch].astype(np.float32), sr, int(sr * speed))
                for ch in range(audio.shape[0])
            ])
            return resampled
        except ImportError:
            pass

    import dawdreamer as daw
    engine = daw.RenderEngine(sample_rate=sr, block_size=512)
    pwp = engine.make_playbackwarp_processor("warp", audio.astype(np.float32))
    pwp.transpose = semitones
    pwp.time_ratio = 1.0 / speed
    engine.load_graph([(pwp, [])])

    duration = (audio.shape[1] / sr) / speed + 1.0
    engine.render(duration)
    result = engine.get_audio()

    peak = np.max(np.abs(result), axis=0)
    last_nonzero = np.where(peak > 0.0001)[0]
    if len(last_nonzero) > 0:
        result = result[:, :last_nonzero[-1] + sr]

    return result


# ── Delay (pure numpy — dawdreamer's native delay segfaults on Windows) ──

def _apply_delay(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Single-tap stereo delay with wet/dry mix."""
    time_ms = float(params.get("time_ms", 250))
    wet = float(params.get("wet", 0.3))

    delay_samples = int(sr * time_ms / 1000)
    if delay_samples <= 0 or wet <= 0:
        return audio

    n_ch, n_samples = audio.shape
    out = np.empty_like(audio)
    for ch in range(n_ch):
        delayed = np.zeros(n_samples, dtype=audio.dtype)
        if delay_samples < n_samples:
            delayed[delay_samples:] = audio[ch, :n_samples - delay_samples]
        out[ch] = audio[ch] * (1 - wet) + delayed * wet

    return out


# ── Reverb (Freeverb-style, pure scipy — dawdreamer reverb broken on Windows) ──

_COMB_TUNINGS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
_AP_TUNINGS = [556, 441, 341, 225]
_STEREO_SPREAD = 23

def _apply_reverb(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Freeverb-style algorithmic reverb using scipy.signal.lfilter."""
    from scipy.signal import lfilter

    room_size = float(params.get("room_size", 0.5))
    damping = float(params.get("damping", 0.5))
    wet = float(params.get("wet", 0.3))
    dry = float(params.get("dry", 0.7))
    width = float(params.get("width", 1.0))

    if wet <= 0:
        return audio

    feedback = room_size * 0.28 + 0.7
    damp = damping * 0.4
    scale = sr / 44100

    n_ch, n_samples = audio.shape
    mono_in = audio.mean(axis=0).astype(np.float32) if n_ch > 1 else audio[0].astype(np.float32)

    wet_l = np.zeros(n_samples, dtype=np.float32)
    wet_r = np.zeros(n_samples, dtype=np.float32)

    for base_delay in _COMB_TUNINGS:
        d_l = max(1, int(base_delay * scale))
        d_r = max(1, int((base_delay + _STEREO_SPREAD) * scale))
        fb = feedback * (1 - damp)
        for d, acc in [(d_l, wet_l), (d_r, wet_r)]:
            b = np.zeros(d + 1, dtype=np.float32)
            b[d] = 1.0
            a = np.zeros(d + 1, dtype=np.float32)
            a[0] = 1.0
            a[d] = -fb
            acc += lfilter(b, a, mono_in)

    wet_l /= len(_COMB_TUNINGS)
    wet_r /= len(_COMB_TUNINGS)

    for base_delay in _AP_TUNINGS:
        d_l = max(1, int(base_delay * scale))
        d_r = max(1, int((base_delay + _STEREO_SPREAD) * scale))
        g = 0.5
        for d, sig in [(d_l, wet_l), (d_r, wet_r)]:
            b = np.zeros(d + 1, dtype=np.float32)
            b[0] = -g
            b[d] = 1.0
            a = np.zeros(d + 1, dtype=np.float32)
            a[0] = 1.0
            a[d] = -g
            sig[:] = lfilter(b, a, sig)

    w1 = (1 + width) / 2
    w2 = (1 - width) / 2

    if n_ch >= 2:
        out = np.empty_like(audio)
        out[0] = audio[0] * dry + (wet_l * w1 + wet_r * w2) * wet
        out[1] = audio[1] * dry + (wet_r * w1 + wet_l * w2) * wet
    else:
        out = np.empty_like(audio)
        out[0] = audio[0] * dry + (wet_l + wet_r) * 0.5 * wet

    return out


# ── dawdreamer built-in effects ──

def _apply_dawdreamer_effect(audio: np.ndarray, sr: int, fx_type: str, params: dict) -> np.ndarray:
    """Apply a dawdreamer built-in effect (compressor, filter, panner)."""
    import dawdreamer as daw

    engine = daw.RenderEngine(sample_rate=sr, block_size=512)

    # Playback processor for input audio
    playback = engine.make_playback_processor("input", audio.astype(np.float32))

    # Create effect processor
    if fx_type == "compressor":
        fx = engine.make_compressor_processor("fx",
            threshold=float(params.get("threshold", -10)),
            ratio=float(params.get("ratio", 4)),
            attack=float(params.get("attack", 5)),
            release=float(params.get("release", 100)),
        )
    elif fx_type == "filter":
        mode_idx = int(params.get("mode", 0))
        mode = FILTER_MODES.get(mode_idx, "low")
        fx = engine.make_filter_processor("fx",
            mode=mode,
            freq=float(params.get("frequency", 1000)),
            q=float(params.get("q", 0.707)),
            gain=float(params.get("gain", 1.0)),
        )
    elif fx_type == "panner":
        fx = engine.make_panner_processor("fx",
            pan=float(params.get("pan", 0)) * 0.5 + 0.5,  # -1..1 → 0..1
        )
    else:
        return audio

    # Chain: input → effect
    engine.load_graph([
        (playback, []),
        (fx, ["input"]),
    ])

    duration = audio.shape[1] / sr + 0.5
    engine.render(duration)
    return engine.get_audio()
