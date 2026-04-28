#!/bin/bash
# Run SOFA-Mandarin forced alignment on ACE-Opencpop per-singer.
# Chunked over 30 singer subdirs to:
#   - Save htk/phones/*.lab outputs incrementally per-singer (not all-or-nothing)
#   - Bound in-memory prediction accumulation (~3500 samples per call, not 105K)
#   - Resume-safe: skip singers whose htk/phones/ already has >= wav count labs
#
# Expected ~10h total on 3080 Ti. Each singer ~20min.
# Launch:  bash scripts/run_sofa_mandarin_ace.sh

export PYTHONUNBUFFERED=1  # fix Windows block-buffered stdout (per pipeline.md memory)

SOFA_DIR="D:/MyDev/MSSTRVC/SOFA"
CKPT="ckpt/Mandarin/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt"
BASE_FOLDER="segments/sofa_run/ace_opencpop"

cd "$SOFA_DIR"

total_t0=$(date +%s)
singers=$(ls "$BASE_FOLDER" | sort -V)
n=$(echo "$singers" | wc -l)
echo "== SOFA-Mandarin on ACE, $n singers =="

i=0
n_ok=0
n_skip=0
n_fail=0
for singer in $singers; do
    i=$((i+1))
    folder="$BASE_FOLDER/$singer"
    if [ ! -d "$folder" ]; then continue; fi

    done_dir="$folder/htk/phones"
    n_wav=$(ls "$folder"/*.wav 2>/dev/null | wc -l)
    if [ -d "$done_dir" ]; then
        n_lab=$(ls "$done_dir"/*.lab 2>/dev/null | wc -l)
        if [ "$n_lab" -ge "$n_wav" ] && [ "$n_wav" -gt 0 ]; then
            echo "[$i/$n] $singer: SKIP already done ($n_lab/$n_wav)"
            n_skip=$((n_skip+1))
            continue
        fi
    fi

    t0=$(date +%s)
    echo "[$i/$n] $singer: starting ($n_wav wavs)..."
    venv/Scripts/python.exe -u infer.py \
      --ckpt "$CKPT" \
      --folder "$folder" \
      --g2p PhonemeG2P \
      --ap_detector NoneAPDetector \
      --mode force \
      --in_format lab \
      --out_formats htk
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        n_lab=$(ls "$done_dir"/*.lab 2>/dev/null | wc -l)
        echo "[$i/$n] $singer: DONE in ${dt}s ($n_lab labs)"
        n_ok=$((n_ok+1))
    else
        echo "[$i/$n] $singer: FAIL rc=$rc (${dt}s)"
        n_fail=$((n_fail+1))
    fi
done

total_dt=$(( $(date +%s) - total_t0 ))
echo "========================================"
echo "ALL DONE: ok=$n_ok skip=$n_skip fail=$n_fail total=$n wall=${total_dt}s"
