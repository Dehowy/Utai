#!/bin/bash
# Re-apply post mute with fix (no 音头 fade) to existing raw/ and inline/ outputs.
# Regenerates post/ and both/ only. No model re-inference needed.

cd D:/MyDev/MSSTRVC/score2hubert_v2
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe

OUT_ROOT="test_outputs/run9_final_eval"

DATASETS=(
  "ace_opencpop ace_zh"
  "kiritan kiritan_ja"
  "itako itako_ja"
  "oniku oniku_ja"
  "ofuton ofuton_ja"
  "natsume natsume_ja"
)

for line in "${DATASETS[@]}"; do
  set -- $line
  ds="$1"; suffix="$2"

  # Extract utt_ids from existing raw log
  log="logs/eval_raw_${suffix}.log"
  if [ ! -f "$log" ]; then
    echo "MISSING log $log — skip $ds"
    continue
  fi
  utt_ids=$(grep -oE '\[[0-9]+/[0-9]+\] [^ ]+ \(' "$log" | awk '{print $2}')
  echo "=== $ds utt_ids: $utt_ids ==="

  # Re-do post (raw → post)
  $PY -u scripts/apply_midi_mute.py \
    --in-dir "$OUT_ROOT/raw/$suffix" \
    --out-dir "$OUT_ROOT/post/$suffix" \
    --dataset "$ds" --utt-ids $utt_ids

  # Re-do both (inline → both)
  $PY -u scripts/apply_midi_mute.py \
    --in-dir "$OUT_ROOT/inline/$suffix" \
    --out-dir "$OUT_ROOT/both/$suffix" \
    --dataset "$ds" --utt-ids $utt_ids
done

echo "=== DONE — post/ and both/ regenerated with 音头保留 fix ==="
