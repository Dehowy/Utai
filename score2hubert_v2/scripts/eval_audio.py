"""Generate audio from Score2HuBERT v2 checkpoint via RVC.

For each test sample, produces 4 ablation variants:
  gt_ceiling  : GT HuBERT  + GT F0    (best possible RVC quality)
  pred_full   : pred HuBERT + pred F0 (full pipeline)
  h_pred_f0gt : pred HuBERT + GT F0   (isolates HuBERT quality)
  hgt_f0pred  : GT HuBERT  + pred F0  (isolates F0 quality)

Usage:
    python scripts/eval_audio.py \
        --model checkpoints/best.pt \
        --rvc-model RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth \
        --output-dir test_outputs/ace_v2 --num-samples 5
"""
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.score2hubert import Score2HuBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--rvc-model", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="test_outputs/ace_v2")
    p.add_argument("--processed-dir", type=str, default="data/processed/ace_opencpop")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--min-frames", type=int, default=400)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--strict-load", action="store_true", default=False,
                   help="Strict state_dict loading (set True to catch arch mismatches)")
    p.add_argument("--hubert-mute", choices=["off", "midi", "combined"], default="off",
                   help="Zero pred_hubert at rest frames BEFORE feeding RVC. "
                        "midi=MIDI==0 only; combined=MIDI+voicing (phoneme-aware exempt).")
    p.add_argument("--hubert-mute-vthresh", type=float, default=0.15,
                   help="voicing_sig threshold for 'combined' mode.")
    p.add_argument("--hubert-mute-dilate", type=int, default=5,
                   help="Frames dilation around consonants (exempt from voicing gate).")
    return p.parse_args()


UNVOICED_PHONEMES = {
    "p", "pʰ", "t", "tʰ", "k", "kʰ", "ʔ",
    "f", "s", "ɕ", "ʂ", "ɸ", "h", "x", "ç", "θ",
    "ts", "tsʰ", "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "tʃ",
}
VOICED_CONSONANTS = {
    "b", "d", "g", "v", "z", "ʑ", "ɣ", "ð",
    "dz", "dʑ", "dʒ", "ɖʐ",
    "m", "n", "ŋ", "ɲ", "ɴ",
    "l", "r", "ɾ", "w", "j", "ɹ", "ɥ",
}
EXEMPT_PHONEMES = UNVOICED_PHONEMES | VOICED_CONSONANTS


def compute_hubert_mute_mask(phonemes, pitch, duration, vocab_path,
                              voicing_sig=None, mode="off",
                              vthresh=0.15, dilate=5):
    """Return [T] bool mask: True at frames to ZERO out in pred_hubert."""
    import yaml
    import numpy as np
    phonemes_np = phonemes.cpu().numpy()
    pitch_np = pitch.cpu().numpy().copy()
    duration_np = duration.cpu().numpy()
    # Force pitch=0 at SP phoneme positions (SOFA carryover fix)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    sp_id = vocab["phone2id"].get("SP", 3)
    pitch_np[phonemes_np == sp_id] = 0
    frame_ph = np.repeat(phonemes_np, duration_np)
    frame_midi = np.repeat(pitch_np, duration_np)
    T = len(frame_ph)

    if mode == "off":
        return torch.zeros(T, dtype=torch.bool)

    midi_rest = (frame_midi == 0)
    if mode == "midi":
        return torch.from_numpy(midi_rest)

    # combined: voicing gate with consonant exempt
    if voicing_sig is None:
        return torch.from_numpy(midi_rest)
    vsig = voicing_sig.cpu().numpy() if torch.is_tensor(voicing_sig) else voicing_sig
    if len(vsig) != T:
        idx = np.round(np.arange(T) * (len(vsig) - 1) / max(T - 1, 1)).astype(np.int64)
        vsig = vsig[idx]
    with open(vocab_path, encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    phone2id = vocab["phone2id"]
    exempt_ids = {phone2id[p] for p in EXEMPT_PHONEMES if p in phone2id}
    is_consonant = np.isin(frame_ph, list(exempt_ids))
    exempt = is_consonant.copy()
    for shift in range(1, dilate + 1):
        exempt[shift:] |= is_consonant[:-shift]
        exempt[:-shift] |= is_consonant[shift:]
    voicing_low = (vsig < vthresh) & ~exempt
    combined = midi_rest | voicing_low
    return torch.from_numpy(combined)


# ── Model loading ─────────────────────────────────────────────────────

def load_model(model_path: str, device: str, strict: bool = True):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})

    model = Score2HuBERT(
        n_phones=mc.get("n_phones", 80),
        hidden_dim=mc.get("hidden_dim", 512),
        hubert_dim=mc.get("hubert_dim", 768),
        n_enc_layers=mc.get("n_enc_layers", 4),
        n_heads=mc.get("n_heads", 8),
        enc_kernel=mc.get("enc_kernel", 31),
        n_det_blocks=mc.get("n_det_blocks", 4),
        det_kernel=mc.get("det_kernel", 7),
        dropout=0.0,  # no dropout at inference
    ).to(device)

    # Non-strict load lets older ckpts (e.g. Run 3, pre-LR-pos-features) use the current model:
    # missing keys (pos_pe/pos_proj/dur_proj) fall back to __init__ values. pos_proj and dur_proj
    # final layers are zero-init → feature contribution is 0 → forward = plain expansion.
    res = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict and (res.missing_keys or res.unexpected_keys):
        logger.warning(f"Non-strict load: missing={res.missing_keys}, unexpected={res.unexpected_keys}")
    model.eval()

    step = ckpt.get("step", "?")
    best = ckpt.get("best_val_loss", "?")
    logger.info(f"Loaded: {model_path} (step={step}, best_val={best})")
    params = model.count_params()
    logger.info(f"  Total params: {params['total']:,}")
    return model


# ── RVC loading ───────────────────────────────────────────────────────

def load_rvc(rvc_model_path: str, device: str):
    model_p = Path(rvc_model_path)
    rvc_root = None
    for parent in model_p.parents:
        if (parent / "infer").exists():
            rvc_root = parent
            break
    if rvc_root is None:
        raise RuntimeError(f"Can't find RVC root from {rvc_model_path}")

    for p in [rvc_root / "infer", rvc_root]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    cpt = torch.load(rvc_model_path, map_location="cpu", weights_only=False)
    from lib.infer_pack.models import SynthesizerTrnMs768NSFsid

    cfg = cpt.get("config", None)
    sr = cfg[-1] if cfg else 40000
    net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
    net_g.load_state_dict(cpt.get("weight", {}), strict=False)
    net_g = net_g.to(device).eval()
    sid = torch.tensor([0], device=device)
    logger.info(f"Loaded RVC: {rvc_model_path} (sr={sr})")
    return net_g, sid, sr


def f0_to_coarse(f0_hz: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127.0 * torch.log(1.0 + f0_hz / 700.0)
    f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
    coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
    coarse[f0_hz < 50] = 0
    return coarse


@torch.no_grad()
def rvc_decode(net_g, sid, hubert_feat, f0_hz, device):
    """RVC decode: feat/f0 at 50fps -> upsample 2x to 100fps -> net_g.infer."""
    T_min = min(hubert_feat.size(0), f0_hz.size(0))
    feat = hubert_feat[:T_min].unsqueeze(0).to(device).float()
    f0 = f0_hz[:T_min].to(device).float()

    feat_2x = F.interpolate(
        feat.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)

    f0_2x = F.interpolate(
        f0.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False
    ).view(-1)

    pitch_coarse = f0_to_coarse(f0_2x).unsqueeze(0)
    nsff0 = f0_2x.unsqueeze(0)
    p_len = torch.tensor([feat_2x.size(1)], device=device).long()

    audio = net_g.infer(feat_2x, p_len, pitch_coarse, nsff0, sid)[0][0, 0]
    return audio.float().cpu()


def save_audio(audio: torch.Tensor, sr: int, path: Path):
    a = audio.numpy()
    a = a / (np.abs(a).max() + 1e-8) * 0.95
    sf.write(str(path), a, sr)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.device, strict=args.strict_load)
    net_g, sid, sr = load_rvc(args.rvc_model, args.device)

    proc = Path(args.processed_dir)
    with open(proc / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Filter to test split, in frame range
    candidates = [
        m for m in meta
        if m.get("split") == "test"
        and args.min_frames <= m.get("n_frames", 0) <= args.max_frames
    ]
    if not candidates:
        logger.error(f"No test samples in frame range {args.min_frames}-{args.max_frames}")
        return

    # Pick longest N
    candidates.sort(key=lambda m: -m["n_frames"])
    picks = candidates[:args.num_samples]

    logger.info(f"Picked {len(picks)} samples ({picks[-1]['n_frames']//50}-{picks[0]['n_frames']//50}s)")

    for i, m in enumerate(picks, start=1):
        sid_str = m["id"]
        n_f = m["n_frames"]
        logger.info(f"[{i}/{len(picks)}] {sid_str} ({n_f} frames ~ {n_f/50:.1f}s)")

        # Load
        phonemes = torch.load(proc / "phonemes" / f"{sid_str}.pt", weights_only=True).long()
        pitch = torch.load(proc / "pitch" / f"{sid_str}.pt", weights_only=True).long()
        duration = torch.load(proc / "duration" / f"{sid_str}.pt", weights_only=True).long()
        gt_hubert = torch.load(proc / "hubert" / f"{sid_str}.pt", weights_only=True).float()
        gt_f0 = torch.load(proc / "f0" / f"{sid_str}.pt", weights_only=True).float()

        # Inference — use full forward for voicing_logit access
        with torch.no_grad():
            full_out = model(
                phonemes=phonemes.unsqueeze(0).to(args.device),
                pitch=pitch.unsqueeze(0).to(args.device),
                durations=duration.unsqueeze(0).to(args.device),
                f0_target=None,
            )
        pred_hubert = full_out["hubert_pred"][0].float().cpu()
        pred_f0 = full_out["f0_pred"][0].float().cpu()
        voicing_sig = torch.sigmoid(full_out["voicing_logit"][0]).cpu()

        # HuBERT-space mute: zero pred_hubert at rest frames BEFORE RVC
        if args.hubert_mute != "off":
            mute_mask = compute_hubert_mute_mask(
                phonemes, pitch, duration,
                vocab_path=str(proc / "vocab.yaml"),
                voicing_sig=voicing_sig,
                mode=args.hubert_mute,
                vthresh=args.hubert_mute_vthresh,
                dilate=args.hubert_mute_dilate,
            )
            T_mask = min(len(mute_mask), len(pred_hubert))
            pred_hubert_muted = pred_hubert.clone()
            pred_hubert_muted[:T_mask][mute_mask[:T_mask]] = 0.0
            pred_hubert = pred_hubert_muted
            logger.info(f"    hubert_mute={args.hubert_mute}: zeroed {int(mute_mask.sum())}/{len(mute_mask)} frames")

        # Metrics
        T = min(pred_hubert.size(0), gt_hubert.size(0))
        cos = F.cosine_similarity(
            pred_hubert[:T].unsqueeze(0), gt_hubert[:T].unsqueeze(0), dim=-1
        ).mean().item()
        l1 = F.l1_loss(pred_hubert[:T], gt_hubert[:T]).item()

        voiced = gt_f0[:T] > 50
        if voiced.any():
            pc = pred_f0[:T][voiced].clamp(min=1.0)
            tc = gt_f0[:T][voiced].clamp(min=1.0)
            f0_mae = (1200.0 * torch.abs(torch.log2(pc / tc))).mean().item()
        else:
            f0_mae = 0.0

        logger.info(f"    cos={cos:.4f}  l1={l1:.4f}  f0_mae={f0_mae:.1f}c")

        # Generate audio variants
        T_common = min(pred_hubert.size(0), pred_f0.size(0),
                       gt_hubert.size(0), gt_f0.size(0))
        ph = pred_hubert[:T_common]
        pf = pred_f0[:T_common]
        gh = gt_hubert[:T_common]
        gf = gt_f0[:T_common]

        variants = {
            "gt_ceiling":  (gh, gf),
            "pred_full":   (ph, pf),
            "h_pred_f0gt": (ph, gf),
            "hgt_f0pred":  (gh, pf),
        }
        for name, (h, f) in variants.items():
            audio = rvc_decode(net_g, sid, h, f, args.device)
            out_path = out_dir / f"sample{i}_{name}.wav"
            save_audio(audio, sr, out_path)
            logger.info(f"    {out_path.name} ({len(audio)/sr:.2f}s)")

    logger.info(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
