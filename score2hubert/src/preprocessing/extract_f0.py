"""
F0 extraction for Score2HuBERT training data.

Supports multiple extraction methods to match RVC's internal processing:
- RMVPE (recommended, matches RVC default)
- Harvest (PyWorld)
- Parselmouth

All methods output at 100fps (10ms frame period) to match HuBERT frame rate.
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Union, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class F0Extractor:
    """
    Frame-level F0 extraction at 100fps, matching RVC's internal representation.

    Output format:
    - Continuous F0 in Hz, 0 for unvoiced frames
    - Frame rate: 100fps (hop_length=160 at 16kHz)
    """

    def __init__(
        self,
        method: str = "rmvpe",
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        sample_rate: int = 16000,
        hop_length: int = 160,
        device: str = "cuda",
        rmvpe_model_path: Optional[str] = None,
    ):
        self.method = method
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.device = device
        self.frame_period = hop_length / sample_rate * 1000  # ms

        if method == "rmvpe" and rmvpe_model_path:
            self._load_rmvpe(rmvpe_model_path)

    def _load_rmvpe(self, model_path: str):
        """Load RMVPE model (from RVC assets)."""
        import sys
        import os

        # Add RVC paths for RMVPE imports
        # RMVPE uses `from infer.lib import jit`, so RVC root must be on sys.path
        rvc_root = Path(model_path).parent.parent.parent
        if str(rvc_root) not in sys.path:
            sys.path.insert(0, str(rvc_root))
        rvc_infer = rvc_root / "infer"
        if str(rvc_infer) not in sys.path:
            sys.path.insert(0, str(rvc_infer))

        try:
            from lib.rmvpe import RMVPE
            self.rmvpe = RMVPE(model_path, is_half=True, device=self.device)
            logger.info(f"Loaded RMVPE from {model_path}")
        except ImportError:
            logger.warning("RMVPE import failed, falling back to parselmouth")
            self.method = "parselmouth"

    def extract(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Extract F0 from audio file.

        Returns:
            f0: [T_frame] tensor of F0 values in Hz (0 for unvoiced)
        """
        # Load audio at 16kHz
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        if self.method == "rmvpe":
            f0 = self._extract_rmvpe(audio)
        elif self.method == "harvest":
            f0 = self._extract_harvest(audio)
        elif self.method == "parselmouth":
            f0 = self._extract_parselmouth(audio)
        else:
            raise ValueError(f"Unknown F0 method: {self.method}")

        return torch.from_numpy(f0).float()

    def _extract_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using RMVPE (GPU-accelerated, matches RVC)."""
        f0 = self.rmvpe.infer_from_audio(audio, thred=0.03)
        return f0

    def _extract_harvest(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using PyWorld's Harvest algorithm."""
        import pyworld

        audio64 = audio.astype(np.float64)
        f0, t = pyworld.harvest(
            audio64,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.frame_period,
        )

        # Refine with DIO for better accuracy
        f0 = pyworld.stonemask(audio64, f0, t, self.sample_rate)

        # Apply median filter to smooth outliers
        from scipy.signal import medfilt
        f0 = medfilt(f0, kernel_size=3)

        return f0.astype(np.float32)

    def _extract_parselmouth(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using Parselmouth (Praat)."""
        import parselmouth

        snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        pitch = snd.to_pitch_ac(
            time_step=self.frame_period / 1000,
            voicing_threshold=0.6,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max,
        )

        # Get F0 values at regular intervals
        n_frames = int(len(audio) / self.hop_length)
        f0 = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            t = i * self.frame_period / 1000
            f0_val = pitch.get_value_at_time(t)
            if not np.isnan(f0_val):
                f0[i] = f0_val

        return f0

    def process_directory(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".wav", ".flac", ".ogg", ".mp3"),
    ):
        """Process all audio files in a directory."""
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))

        logger.info(f"Extracting F0 from {len(audio_files)} files using {self.method}")

        for audio_path in tqdm(audio_files, desc=f"F0 ({self.method})"):
            rel_path = audio_path.relative_to(audio_dir)
            out_path = output_dir / rel_path.with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                continue

            try:
                f0 = self.extract(audio_path)
                torch.save(f0, out_path)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")


def extract_energy(audio_path: Union[str, Path], hop_length: int = 160,
                   sample_rate: int = 16000) -> torch.Tensor:
    """Extract frame-level energy (RMS) from audio."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # Frame-level RMS energy
    n_frames = len(audio) // hop_length
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-8)

    # Log-scale
    energy = np.log(energy + 1e-8)

    return torch.from_numpy(energy).float()


def extract_breathiness(audio_path: Union[str, Path], f0: torch.Tensor,
                        hop_length: int = 160, sample_rate: int = 16000,
                        n_fft: int = 512) -> torch.Tensor:
    """
    Estimate breathiness (aperiodicity) from audio.

    Uses the spectral flatness of the harmonic residual as a proxy for breathiness.
    Higher values = more breathy/airy sound.
    """
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=True)
    mag = np.abs(stft)  # [n_freq, n_frames]

    n_frames = min(mag.shape[1], f0.size(0))
    breathiness = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        if f0[i] > 0:
            # Spectral flatness: geometric_mean / arithmetic_mean
            frame_mag = mag[:, i]
            frame_mag = frame_mag[frame_mag > 0]
            if len(frame_mag) > 0:
                geo_mean = np.exp(np.mean(np.log(frame_mag + 1e-8)))
                ari_mean = np.mean(frame_mag)
                breathiness[i] = geo_mean / (ari_mean + 1e-8)

    return torch.from_numpy(breathiness).float()
