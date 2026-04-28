"""Post-Run-8 pipeline.

Steps A (ACE singer1 RVC training) → optional B (JA mixed RVC).
Step C (Yachiyo baseline audio) is already done by post_run8_pipeline.sh.

Each RVC subprocess MUST cd into RVC_ROOT (so its `from infer.lib...` works)
and reference assets/* via relative paths.

Usage:
    python scripts/post_run8_pipeline.py [--skip-b]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

V2 = Path(r"D:\MyDev\MSSTRVC\score2hubert_v2")
RVC_ROOT = V2 / "RVC" / "RVC20240604Nvidia"
LOGS_DIR = RVC_ROOT / "logs"
PY = str(V2 / "venv" / "Scripts" / "python.exe")
RUN8_FINAL = V2 / "checkpoints" / "final.pt"
YACHIYO_RVC = RVC_ROOT / "assets" / "weights" / "yachiyo_runami_e350_s10500.pth"
PRETRAIN_G = RVC_ROOT / "assets" / "pretrained_v2" / "f0G40k.pth"
PRETRAIN_D = RVC_ROOT / "assets" / "pretrained_v2" / "f0D40k.pth"
SR = 40000
VERSION = "v2"
BATCH = 12  # raised from 8 — Epoch 1 timing showed 8 min/epoch, GPU only 59% saturated → bigger batch helps


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_rvc(cmd: list[str], cwd: Path = RVC_ROOT, env_extra: dict | None = None) -> bool:
    """Run a subprocess in RVC_ROOT (so `from infer.*` imports work)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # Windows + PyTorch 2.x distributed init fixes:
    env["USE_LIBUV"] = "0"  # libuv not built-in
    env["MASTER_ADDR"] = "127.0.0.1"  # avoid hostname binding (Windows firewall blocks 10013)
    env["MASTER_PORT"] = "29500"  # explicit common port to avoid auto-pick collision
    if env_extra:
        env.update(env_extra)
    log(f"  $ (in {cwd}) {' '.join(map(str, cmd))}")
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                       capture_output=True, text=True, errors="replace")
    if p.stdout:
        for line in p.stdout.splitlines()[-15:]:
            print(f"    | {line}", flush=True)
    if p.returncode != 0:
        log(f"  FAILED rc={p.returncode}")
        if p.stderr:
            for line in p.stderr.splitlines()[-15:]:
                print(f"    | STDERR {line}", flush=True)
        return False
    return True


def gen_audio(model_pth: Path, rvc_pth: Path, processed_dir: Path, out_dir: Path,
              num_samples: int, min_frames: int, max_frames: int) -> bool:
    """Run eval_audio.py to generate audio variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(V2 / "scripts" / "eval_audio.py"),
           "--model", str(model_pth),
           "--rvc-model", str(rvc_pth),
           "--output-dir", str(out_dir),
           "--processed-dir", str(processed_dir),
           "--num-samples", str(num_samples),
           "--min-frames", str(min_frames),
           "--max-frames", str(max_frames),
           "--device", "cuda"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([str(c) for c in cmd], env=env,
                       capture_output=True, text=True, errors="replace")
    if p.stdout:
        for line in p.stdout.splitlines()[-3:]:
            print(f"    | {line}", flush=True)
    if p.returncode != 0:
        log(f"  audio gen FAILED rc={p.returncode}")
        if p.stderr:
            for line in p.stderr.splitlines()[-5:]:
                print(f"    | STDERR {line}", flush=True)
        return False
    return True


def write_filelist(exp_dir: Path) -> bool:
    """Write training filelist.txt referencing extracted features."""
    feat_dir = exp_dir / "3_feature768"
    gt_dir = exp_dir / "0_gt_wavs"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    if not gt_dir.exists() or not feat_dir.exists():
        log(f"  filelist: missing dirs (gt={gt_dir.exists()} feat={feat_dir.exists()})")
        return False
    samples = sorted(f.stem for f in gt_dir.glob("*.wav"))
    lines = [
        f"{gt_dir / f'{s}.wav'}|{feat_dir / f'{s}.npy'}|{f0_dir / f'{s}.wav.npy'}|{f0nsf_dir / f'{s}.wav.npy'}|0"
        for s in samples
    ]
    out = exp_dir / "filelist.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"  wrote {len(lines)} entries to filelist.txt")
    return True


def build_index(exp_dir: Path, exp_name: str, max_files: int = 2000) -> bool:
    import numpy as np
    import faiss
    feat_dir = exp_dir / "3_feature768"
    feat_files = sorted(feat_dir.glob("*.npy"))[:max_files]
    if not feat_files:
        log(f"  index: no features at {feat_dir}")
        return False
    feats = []
    for fp in feat_files:
        try:
            feats.append(np.load(fp).astype(np.float32))
        except Exception as e:
            log(f"  index: skip {fp.name}: {e}")
    if not feats:
        return False
    big = np.concatenate(feats, axis=0)
    log(f"  index: features shape {big.shape}")
    n_ivf = max(4, min(big.shape[0] // 39, 256))
    index = faiss.index_factory(big.shape[1], f"IVF{n_ivf},Flat")
    index.train(big)
    index.add(big)
    out = exp_dir / f"added_IVF{n_ivf}_Flat_v2.index"
    faiss.write_index(index, str(out))
    log(f"  index: wrote {out.name}")
    return True


def train_rvc(exp_name: str, audio_dir: Path, epochs: int = 200, skip_prep: bool = False) -> Path | None:
    """Full RVC training pipeline. Returns path to latest weight .pth.

    skip_prep: if True, skip A.1-A.3 (assume already done from prior run).
    """
    exp_dir = LOGS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== Train RVC: exp={exp_name}, epochs={epochs}, data={audio_dir}, skip_prep={skip_prep} ===")

    if not skip_prep:
        log(f"{exp_name}.1 preprocess (slice + resample to {SR}Hz)")
        if not run_rvc([PY, "infer/modules/train/preprocess.py",
                        str(audio_dir), str(SR), "4", str(exp_dir), "False", "3.7"]):
            return None
        gt_count = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
        log(f"  preprocess output: {gt_count} sliced wavs")
        if gt_count == 0:
            return None

        log(f"{exp_name}.2 extract F0 (RMVPE GPU)")
        if not run_rvc([PY, "infer/modules/train/extract/extract_f0_rmvpe.py",
                        "1", "0", "0", str(exp_dir), "True"]):
            return None

        log(f"{exp_name}.3 extract features (HuBERT v2)")
        if not run_rvc([PY, "infer/modules/train/extract_feature_print.py",
                        "cuda", "1", "0", "0", str(exp_dir), VERSION, "True"]):
            return None
    else:
        gt_count = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
        feat_count = len(list((exp_dir / "3_feature768").glob("*.npy")))
        log(f"  skip_prep: gt={gt_count}, features={feat_count}")
        if gt_count == 0 or feat_count == 0:
            log(f"  ERROR: skip_prep requested but data missing")
            return None

    log(f"{exp_name}.4 write filelist")
    if not write_filelist(exp_dir):
        return None

    log(f"{exp_name}.5 copy config")
    src = RVC_ROOT / "configs" / VERSION / "40k.json"
    shutil.copy(src, exp_dir / "config.json")

    log(f"{exp_name}.6 TRAIN ({epochs} epochs)")
    if not run_rvc([PY, "-u", "infer/modules/train/train.py",
                    "-e", exp_name, "-sr", "40k", "-f0", "1",
                    "-bs", str(BATCH), "-g", "0",
                    "-te", str(epochs), "-se", "5",  # save state every 5 epochs (faster checkpoints)
                    "-pg", str(PRETRAIN_G), "-pd", str(PRETRAIN_D),
                    "-l", "0", "-c", "0", "-sw", "1", "-v", VERSION],  # sw=1: also save inference weights at each save_every interval
                   env_extra={"RVC_VERSION": VERSION}):
        return None

    log(f"{exp_name}.7 build index")
    build_index(exp_dir, exp_name)

    weights = sorted((RVC_ROOT / "assets" / "weights").glob(f"{exp_name}_*.pth"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if weights:
        log(f"  latest weight: {weights[0].name}")
        return weights[0]
    log(f"  WARN: no weight files found for {exp_name}")
    return None


def gen_audio_with_rvc(rvc_pth: Path, out_root: Path, datasets: list[tuple]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for ds_name, dsname_out, min_f, max_f, n in datasets:
        log(f"  audio gen: {ds_name} -> {dsname_out}")
        gen_audio(
            model_pth=RUN8_FINAL,
            rvc_pth=rvc_pth,
            processed_dir=V2 / "data" / "processed" / ds_name,
            out_dir=out_root / dsname_out,
            num_samples=n, min_frames=min_f, max_frames=max_f,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--a-epochs", type=int, default=200)
    ap.add_argument("--b-epochs", type=int, default=100)
    ap.add_argument("--skip-a-prep", action="store_true",
                    help="Skip A preprocess/F0/features (assume prior run did them)")
    args = ap.parse_args()

    if not RUN8_FINAL.exists():
        log(f"ERROR: Run 8 final.pt not found at {RUN8_FINAL}")
        sys.exit(1)

    # ── Step A ──
    a_data = LOGS_DIR / "ace_zh_singer1" / "audio_raw"
    if not a_data.exists() or not list(a_data.glob("*.wav")):
        log("ERROR: A data missing")
        sys.exit(1)

    a_weight = train_rvc("ace_zh_singer1", a_data, epochs=args.a_epochs,
                         skip_prep=args.skip_a_prep)
    if a_weight:
        gen_audio_with_rvc(
            a_weight,
            V2 / "test_outputs" / "run8_final_ace_singer1_rvc",
            [
                ("ace_opencpop", "ace_zh", 200, 600, 3),
                ("kiritan", "kiritan_ja", 400, 1000, 2),
                ("ofuton", "ofuton_ja", 400, 1000, 2),
            ],
        )
    else:
        log("WARN: A training did not produce a weight; skipping A audio gen")

    # ── Step B (optional, time-bounded) ──
    if args.skip_b:
        log("B skipped (--skip-b)")
    else:
        now_h = datetime.now().hour
        if now_h >= 10:
            log(f"B skipped (hour={now_h} ≥ 10, user may be waking)")
        else:
            b_data = LOGS_DIR / "ja_mixed_5singer" / "audio_raw"
            if not b_data.exists() or not list(b_data.glob("*.wav")):
                log("WARN: B data missing, skipping")
            else:
                b_epochs = args.b_epochs
                # If we're starting late, reduce epochs
                if now_h >= 9:
                    b_epochs = min(b_epochs, 30)
                elif now_h >= 8:
                    b_epochs = min(b_epochs, 60)
                log(f"B starting at hour {now_h} with {b_epochs} epochs")
                b_weight = train_rvc("ja_mixed_5singer", b_data, epochs=b_epochs)
                if b_weight:
                    gen_audio_with_rvc(
                        b_weight,
                        V2 / "test_outputs" / "run8_final_ja_mixed_rvc",
                        [
                            ("kiritan", "kiritan_ja", 400, 1000, 2),
                            ("ofuton", "ofuton_ja", 400, 1000, 2),
                            ("itako", "itako_ja", 400, 1000, 2),
                            ("natsume", "natsume_ja", 400, 1000, 2),
                            ("oniku", "oniku_ja", 400, 1000, 2),
                        ],
                    )

    log("=== ALL DONE ===")
    log(f"  C: {V2 / 'test_outputs' / 'run8_final_yachiyo'}")
    log(f"  A: {V2 / 'test_outputs' / 'run8_final_ace_singer1_rvc'}")
    if not args.skip_b:
        log(f"  B: {V2 / 'test_outputs' / 'run8_final_ja_mixed_rvc'}")


if __name__ == "__main__":
    main()
