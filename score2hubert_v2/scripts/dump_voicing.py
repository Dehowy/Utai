"""For a list of utt_ids, run Score2HuBERT forward and save pred_f0 + voicing_sigmoid
per frame as .pt tensors. Used by mute scripts to detect label-voice-but-model-unvoiced frames.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from src.model.score2hubert import Score2HuBERT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(V2 / "configs" / "train_run8.yaml"))
    ap.add_argument("--processed-dir", required=True, help="data/processed/<dataset>/")
    ap.add_argument("--utt-ids", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    device = "cuda"
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    model = Score2HuBERT(
        n_phones=mc["n_phones"], hidden_dim=mc["hidden_dim"],
        hubert_dim=mc["hubert_dim"], n_enc_layers=mc["n_enc_layers"],
        n_heads=mc["n_heads"], enc_kernel=mc["enc_kernel"],
        n_det_blocks=mc["n_det_blocks"], det_kernel=mc["det_kernel"],
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    proc = Path(args.processed_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for utt in args.utt_ids:
            phonemes = torch.load(proc / "phonemes" / f"{utt}.pt", weights_only=False).long().unsqueeze(0).to(device)
            pitch = torch.load(proc / "pitch" / f"{utt}.pt", weights_only=False).long().unsqueeze(0).to(device)
            duration = torch.load(proc / "duration" / f"{utt}.pt", weights_only=False).long().unsqueeze(0).to(device)
            result = model(phonemes=phonemes, pitch=pitch, durations=duration, f0_target=None)
            pred_f0 = result["f0_pred"][0].cpu()  # [T_frame]
            voicing_logit = result["voicing_logit"][0].cpu()  # [T_frame]
            voicing_sig = torch.sigmoid(voicing_logit)
            torch.save({"pred_f0": pred_f0, "voicing_sig": voicing_sig}, out / f"{utt}.pt")
            print(f"  dumped {utt}: T={pred_f0.shape[0]} voicing_mean={voicing_sig.mean():.3f}")


if __name__ == "__main__":
    main()
