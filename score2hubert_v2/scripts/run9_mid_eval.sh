#!/bin/bash
# Run 9 mid-training eval at step 10000 (JA plateau confirmed).
# CPU-only to avoid GPU contention with running training.
# Mirrors Run 8 step 25K procedure: 6 datasets × 2 samples × 4 variants.

cd D:/MyDev/MSSTRVC/score2hubert_v2

CKPT="archive/run9_snapshots/step_10000_ja_plateau_confirmed.pt"
RVC="D:/MyDev/MSSTRVC/RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth"
OUT_BASE="test_outputs/run9_mid_step10000"

mkdir -p "$OUT_BASE"
t_total=$(date +%s)

# (dataset, out_suffix, min_frames, max_frames)
DATASETS=(
  "ace_opencpop ace_zh 200 600"
  "kiritan kiritan_ja 400 1000"
  "itako itako_ja 400 1000"
  "oniku oniku_ja 400 1000"
  "ofuton ofuton_ja 400 1000"
  "natsume natsume_ja 400 1000"
)

for line in "${DATASETS[@]}"; do
  set -- $line
  ds="$1"; suffix="$2"; minf="$3"; maxf="$4"
  out="$OUT_BASE/$suffix"
  echo "=== $ds → $out (min=$minf max=$maxf) ==="
  t0=$(date +%s)
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -u scripts/eval_audio.py \
    --model "$CKPT" \
    --rvc-model "$RVC" \
    --processed-dir "data/processed/$ds" \
    --output-dir "$out" \
    --num-samples 2 \
    --min-frames "$minf" \
    --max-frames "$maxf" \
    --device cpu
  echo "=== $ds done in $(( $(date +%s) - t0 ))s ==="
done

echo "=== ALL done in $(( $(date +%s) - t_total ))s ==="
