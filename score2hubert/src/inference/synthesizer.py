"""
Score2HuBERT Inference Synthesizer.

Complete inference pipeline:
    Score → Score2HuBERT → HuBERT features + F0 → RVC decoder → Audio

Supports:
- Direct feature injection into RVC's net_g (bypassing HuBERT re-extraction)
- Chunked generation for long sequences
- RVC compensation post-processing
- User pitch curve override
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

from ..model.score2hubert import Score2HuBERT
from ..data.phoneme import PhonemeVocab

logger = logging.getLogger(__name__)


class Score2HuBERTSynthesizer:
    """
    End-to-end synthesizer: musical score → audio via Score2HuBERT + RVC.

    Usage:
        synth = Score2HuBERTSynthesizer(model_path, rvc_model_path, vocab)
        audio = synth.synthesize(phonemes, pitches, durations)
        sf.write("output.wav", audio, 44100)
    """

    def __init__(
        self,
        model_path: str,
        rvc_model_path: Optional[str] = None,
        vocab: Optional[PhonemeVocab] = None,
        device: str = "cuda",
        compensation_stats: Optional[str] = None,
    ):
        self.device = device
        self.vocab = vocab or PhonemeVocab()

        # Load Score2HuBERT model
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.config = ckpt["config"]
        self.model = Score2HuBERT(self.config).to(device)

        # Load EMA weights if available, otherwise regular weights
        if "ema_shadow" in ckpt:
            state_dict = {}
            for name, param in self.model.named_parameters():
                if name in ckpt["ema_shadow"]:
                    state_dict[name] = ckpt["ema_shadow"][name]
                else:
                    state_dict[name] = ckpt["model_state"][name]
            # Include non-parameter buffers
            for name, buf in self.model.named_buffers():
                if name in ckpt["model_state"]:
                    state_dict[name] = ckpt["model_state"][name]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(ckpt["model_state"])

        self.model.eval()
        logger.info(f"Loaded Score2HuBERT from {model_path} (step {ckpt.get('global_step', '?')})")

        # Load RVC decoder (optional)
        self.rvc_decoder = None
        self.rvc_sr = None
        if rvc_model_path:
            self._load_rvc_decoder(rvc_model_path)

        # Load compensation stats (optional)
        self.compensation = None
        if compensation_stats:
            self.compensation = torch.load(compensation_stats, weights_only=True)
            logger.info("Loaded RVC compensation stats")

    def _load_rvc_decoder(self, model_path: str):
        """Load RVC's net_g for direct feature → audio synthesis."""
        import sys

        model_p = Path(model_path)
        rvc_root = None
        for parent in model_p.parents:
            if (parent / "infer").exists():
                rvc_root = parent
                break

        if rvc_root is None:
            raise RuntimeError(f"Cannot find RVC root from {model_path}")

        rvc_infer = rvc_root / "infer"
        if str(rvc_infer) not in sys.path:
            sys.path.insert(0, str(rvc_infer))

        cpt = torch.load(model_path, map_location="cpu", weights_only=False)

        from lib.infer_pack.models import SynthesizerTrnMs768NSFsid

        config = cpt.get("config", None)
        self.rvc_sr = cpt.get("config", [0, 0, 0, 0, 0])[-1]
        if self.rvc_sr == 0:
            self.rvc_sr = 40000

        if config:
            net_g = SynthesizerTrnMs768NSFsid(*config, is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid(
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2],
                512, [16, 16, 4, 4], 109, self.rvc_sr, 1,
            )

        net_g.load_state_dict(cpt.get("weight", {}), strict=False)
        net_g = net_g.to(self.device).eval()
        self.rvc_decoder = net_g
        self.rvc_sid = torch.tensor([0], device=self.device)

        logger.info(f"Loaded RVC decoder: sr={self.rvc_sr}")

    @torch.no_grad()
    def synthesize(
        self,
        phonemes: List[str],
        pitches: List[int],
        durations: List[int],
        language: str = "ja",
        singing_ratio: float = 1.0,
        n_steps: int = 10,
        solver: str = "euler",
        cfg_scale: float = 2.0,
        temperature: float = 1.0,
        pitch_curve: Optional[np.ndarray] = None,
        pitch_shift: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Synthesize audio from score.

        Args:
            phonemes: list of phoneme strings (e.g., ["ja_k", "ja_a", ...])
            pitches: list of MIDI note numbers per phoneme
            durations: list of frame counts per phoneme (at 100fps)
            language: "ja" | "zh" | "en"
            singing_ratio: 0.0 (rap) to 1.0 (singing)
            n_steps: flow matching ODE steps
            solver: "euler" | "midpoint" | "rk4"
            cfg_scale: classifier-free guidance scale
            temperature: noise temperature for flow matching
            pitch_curve: optional [T_frame] array of F0 in Hz (overrides predictor)
            pitch_shift: semitone shift for RVC output (+/- 24)

        Returns:
            dict with:
                'audio': np.ndarray of audio samples
                'sr': sample rate
                'hubert': generated HuBERT features [T, 768]
                'f0': predicted F0 [T] in Hz
        """
        # Encode phonemes
        phone_ids = torch.tensor(
            self.vocab.encode(phonemes, language), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        pitch_t = torch.tensor(pitches, dtype=torch.long).unsqueeze(0).to(self.device)
        dur_t = torch.tensor(durations, dtype=torch.long).unsqueeze(0).to(self.device)
        sr_t = torch.tensor([singing_ratio], dtype=torch.float32).to(self.device)
        lang_id = torch.tensor([self.vocab.get_language_id(language)]).to(self.device)

        pitch_curve_t = None
        if pitch_curve is not None:
            pitch_curve_t = torch.from_numpy(pitch_curve).float().unsqueeze(0).to(self.device)

        # Generate HuBERT + F0
        output = self.model.synthesize(
            phonemes=phone_ids,
            pitch=pitch_t,
            duration=dur_t,
            singing_ratio=sr_t,
            lang_id=lang_id,
            n_steps=n_steps,
            solver=solver,
            cfg_scale=cfg_scale,
            temperature=temperature,
            pitch_curve=pitch_curve_t,
        )

        hubert_feat = output["hubert"][0]  # [T, 768]
        f0 = output["f0"][0]              # [T]

        # Apply RVC compensation if available
        if self.compensation is not None:
            from ..preprocessing.rvc_compensation import RVCCompensationAnalyzer
            hubert_feat = RVCCompensationAnalyzer.apply_compensation(
                hubert_feat, self.compensation
            )

        result = {
            "hubert": hubert_feat.cpu().numpy(),
            "f0": f0.cpu().numpy(),
        }

        # Convert to audio via RVC decoder
        if self.rvc_decoder is not None:
            audio = self._rvc_decode(hubert_feat, f0, pitch_shift)
            result["audio"] = audio.cpu().numpy()
            result["sr"] = self.rvc_sr
        else:
            result["audio"] = None
            result["sr"] = None

        return result

    def _rvc_decode(
        self,
        hubert_feat: torch.Tensor,  # [T, 768] at 50fps
        f0: torch.Tensor,           # [T] Hz at 50fps
        pitch_shift: int = 0,
    ) -> torch.Tensor:
        """
        Direct RVC decoder inference: HuBERT features + F0 -> audio.

        This BYPASSES RVC's HuBERT encoder entirely — we feed our generated
        features directly into net_g's decoder.

        Matches RVC's official `pipeline.py:247` interface:
          - feat upsampled 2x to 100fps (RVC's expected internal rate)
          - feat passed as [B, T, 768], NOT permuted to [B, 768, T]
          - f0 also upsampled 2x to match feat length
          - pitch_coarse and nsff0 derived from upsampled f0
        """
        # Apply pitch shift
        if pitch_shift != 0:
            ratio = 2.0 ** (pitch_shift / 12.0)
            f0 = f0.clone()
            voiced = f0 > 0
            f0[voiced] = f0[voiced] * ratio

        # Truncate to common length first
        T_min = min(hubert_feat.size(0), f0.size(0))
        feat = hubert_feat[:T_min].unsqueeze(0).to(self.device).float()  # [1, T, 768]
        f0_hz = f0[:T_min].to(self.device).float()                       # [T]

        # Upsample feat 2x: [1, T, 768] -> [1, 2T, 768] (matches RVC pipeline.py:247)
        feat_2x = F.interpolate(
            feat.permute(0, 2, 1), scale_factor=2, mode="nearest"
        ).permute(0, 2, 1)

        # Upsample F0 2x linearly to match feat length
        f0_2x = F.interpolate(
            f0_hz.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False
        ).view(-1)

        # F0 to coarse pitch index (RVC format)
        f0_mel = 1127.0 * torch.log(1.0 + f0_2x / 700.0)
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
        pitch_coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
        pitch_coarse[f0_2x < 50] = 0
        pitch_coarse = pitch_coarse.unsqueeze(0)  # [1, 2T]
        nsff0 = f0_2x.unsqueeze(0)                # [1, 2T]

        phone_lengths = torch.tensor([feat_2x.size(1)], device=self.device).long()

        # RVC decoder inference — feat as [B, T, 768], NOT permuted
        audio = self.rvc_decoder.infer(
            feat_2x,
            phone_lengths,
            pitch_coarse,
            nsff0,
            self.rvc_sid,
        )[0][0, 0]  # [samples]

        return audio.float()

    def save_audio(self, audio: np.ndarray, sr: int, path: str):
        """Save audio to WAV file."""
        # Normalize to prevent clipping
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
        sf.write(path, audio, sr)
        logger.info(f"Saved audio: {path} ({len(audio)/sr:.2f}s, {sr}Hz)")
