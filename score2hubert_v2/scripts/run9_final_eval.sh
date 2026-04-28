#!/bin/bash
# Run 9 final eval — 4-way mute strategy A/B:
#   raw    = eval_audio.py --hubert-mute off          (baseline, no mute)
#   inline = eval_audio.py --hubert-mute combined     (HuBERT-level SP zero)
#   post   = raw + apply_midi_mute.py                  (amplitude-gate post-process)
#   both   = inline + apply_midi_mute.py               (double safety net)
#
# Per dataset: 2 samples × 4 internal variants × 4 mute strategies = 32 wavs/dataset
# 6 datasets = 192 wavs total
# Expected wall: ~15-20 min on GPU

cd D:/MyDev/MSSTRVC/score2hubert_v2
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

PY=venv/Scripts/python.exe
CKPT="archive/run9_final/best.pt"
RVC="D:/MyDev/MSSTRVC/RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth"
OUT_ROOT="test_outputs/run9_final_eval"
mkdir -p "$OUT_ROOT"

# (dataset_dir, out_suffix, min_frames, max_frames)
DATASETS=(
  "ace_opencpop ace_zh 200 600"
  "kiritan kiritan_ja 400 1000"
  "itako itako_ja 400 1000"
  "oniku oniku_ja 400 1000"
  "ofuton ofuton_ja 400 1000"
  "natsume natsume_ja 400 1000"
)

total_t0=$(date +%s)

for line in "${DATASETS[@]}"; do
  set -- $line
  ds="$1"; suffix="$2"; minf="$3"; maxf="$4"

  raw_dir="$OUT_ROOT/raw/$suffix"
  inline_dir="$OUT_ROOT/inline/$suffix"
  post_dir="$OUT_ROOT/post/$suffix"
  both_dir="$OUT_ROOT/both/$suffix"
  mkdir -p "$raw_dir" "$inline_dir" "$post_dir" "$both_dir"

  echo "======================================"
  echo "=== $ds → 4-way ==="
  echo "======================================"

  # 1. Raw (no mute)
  t0=$(date +%s)
  echo "--- [1/4] raw ---"
  $PY -u scripts/eval_audio.py \
    --model "$CKPT" --rvc-model "$RVC" \
    --processed-dir "data/processed/$ds" \
    --output-dir "$raw_dir" \
    --num-samples 2 \
    --min-frames "$minf" --max-frames "$maxf" \
    --device cuda 2>&1 | tee "logs/eval_raw_${suffix}.log"

  # Extract utt_ids from log output
  utt_ids=$(grep -oE '\[[0-9]+/[0-9]+\] [^ ]+ \(' "logs/eval_raw_${suffix}.log" | awk '{print $2}')
  echo "  utt_ids: $utt_ids"

  # 2. Inline (--hubert-mute combined)
  echo "--- [2/4] inline (--hubert-mute combined) ---"
  $PY -u scripts/eval_audio.py \
    --model "$CKPT" --rvc-model "$RVC" \
    --processed-dir "data/processed/$ds" \
    --output-dir "$inline_dir" \
    --num-samples 2 \
    --min-frames "$minf" --max-frames "$maxf" \
    --hubert-mute combined \
    --device cuda 2>&1 | tee "logs/eval_inline_${suffix}.log"

  # 3. Post mute on raw
  echo "--- [3/4] post (apply_midi_mute on raw) ---"
  $PY -u scripts/apply_midi_mute.py \
    --in-dir "$raw_dir" --out-dir "$post_dir" \
    --dataset "$ds" --utt-ids $utt_ids

  # 4. Both (inline + post mute)
  echo "--- [4/4] both (apply_midi_mute on inline) ---"
  $PY -u scripts/apply_midi_mute.py \
    --in-dir "$inline_dir" --out-dir "$both_dir" \
    --dataset "$ds" --utt-ids $utt_ids

  echo "=== $ds done in $(( $(date +%s) - t0 ))s ==="
done

echo "==============================================="
echo "ALL DONE in $(( $(date +%s) - total_t0 ))s"
echo "Outputs: $OUT_ROOT/{raw,inline,post,both}/<dataset>/"
