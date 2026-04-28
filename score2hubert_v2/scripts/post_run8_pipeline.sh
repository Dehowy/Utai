#!/bin/bash
# Post-Run-8 pipeline:
#   C (baseline Yachiyo RVC audio on Run 8 final.pt) →
#   A (ACE singer 1 Chinese RVC training) →
#   (if time) B (JA mixed 5-singer RVC training) →
#   Generate audio samples with each RVC × best Run 8 ckpt
#
# Assumptions:
#  - Run 8 has completed (checkpoints/final.pt exists, logs show "Training complete.")
#  - Pipeline data already prepped (ace_zh_singer1/ + ja_mixed_5singer/)
#  - GPU free after Run 8
#
# Usage:
#   bash scripts/post_run8_pipeline.sh [--skip-b] [--time-limit-hours HOURS]

set -e
export PYTHONIOENCODING=utf-8
V2=/d/MyDev/MSSTRVC/score2hubert_v2
PY=$V2/venv/Scripts/python.exe
RVC_ROOT=$V2/RVC/RVC20240604Nvidia
LOGS_DIR=$RVC_ROOT/logs
YACHIYO_RVC=$RVC_ROOT/assets/weights/yachiyo_runami_e350_s10500.pth
RUN8_FINAL=$V2/checkpoints/final.pt

SKIP_B=0
USER_WAKE_DEADLINE_HOURS=8  # rough deadline to have all done
for arg in "$@"; do
    case $arg in
        --skip-b) SKIP_B=1 ;;
        --time-limit-hours) shift; USER_WAKE_DEADLINE_HOURS=$2 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }

# =========================================================================
# Step 0: Wait for Run 8 + sanity check
# =========================================================================
if [ ! -f "$RUN8_FINAL" ]; then
    log "ERROR: Run 8 final.pt not found at $RUN8_FINAL"
    exit 1
fi
log "Run 8 final.pt confirmed: $RUN8_FINAL"

# =========================================================================
# Step C: Baseline audio with Yachiyo RVC + Run 8 final.pt
# =========================================================================
log "=== STEP C: Baseline Yachiyo RVC audio on Run 8 final.pt ==="
C_OUT=$V2/test_outputs/run8_final_yachiyo
mkdir -p $C_OUT

for ds in kiritan ofuton itako natsume oniku ace_opencpop jvs_music; do
    if [ "$ds" = "ace_opencpop" ]; then
        dsname=ace_zh
        MIN=200; MAX=600
    elif [ "$ds" = "jvs_music" ]; then
        dsname=jvs_ja
        MIN=1000; MAX=2500
    else
        dsname=${ds}_ja
        MIN=400; MAX=1000
    fi
    log "  C: $ds ($dsname)"
    $PY $V2/scripts/eval_audio.py \
        --model $RUN8_FINAL \
        --rvc-model $YACHIYO_RVC \
        --output-dir $C_OUT/$dsname \
        --processed-dir $V2/data/processed/$ds \
        --num-samples 3 --min-frames $MIN --max-frames $MAX \
        --device cuda 2>&1 | tail -8 || log "  WARN: eval_audio $ds failed"
done
log "C done: $C_OUT"

# =========================================================================
# Step A: Train ACE singer 1 Chinese RVC
# =========================================================================
log "=== STEP A: Train ACE singer 1 Chinese RVC ==="
A_EXP=ace_zh_singer1
A_INPUT=$LOGS_DIR/$A_EXP/audio_raw
A_EXPDIR=$LOGS_DIR/$A_EXP

# Check wavs prepped
if [ ! -d "$A_INPUT" ] || [ $(ls $A_INPUT/*.wav 2>/dev/null | wc -l) -lt 100 ]; then
    log "ERROR: A wav data not found at $A_INPUT"
    exit 1
fi

SR=40000
VERSION=v2
EPOCHS=200
BATCH=8
PRETR_G=$RVC_ROOT/assets/pretrained_${VERSION}/f0G40k.pth
PRETR_D=$RVC_ROOT/assets/pretrained_${VERSION}/f0D40k.pth

log "A.1 preprocess (slice + resample to ${SR}Hz)"
$PY $RVC_ROOT/infer/modules/train/preprocess.py \
    $A_INPUT $SR 4 $A_EXPDIR False 3.7 2>&1 | tail -5

log "A.2 extract F0 (RMVPE GPU)"
$PY $RVC_ROOT/infer/modules/train/extract/extract_f0_rmvpe.py \
    1 0 0 $A_EXPDIR True 2>&1 | tail -5

log "A.3 extract features (HuBERT v2)"
$PY $RVC_ROOT/infer/modules/train/extract_feature_print.py \
    cuda 1 0 0 $A_EXPDIR $VERSION True 2>&1 | tail -5

log "A.4 write filelist"
$PY -c "
import os, random
exp = r'$A_EXPDIR'
sr = '${SR}'
feat_dir = exp + '/3_feature768'
gt_dir = exp + '/0_gt_wavs'
f0_dir = exp + '/2a_f0'
f0nsf_dir = exp + '/2b-f0nsf'
samples = [f.replace('.wav','') for f in os.listdir(gt_dir)]
random.seed(0); random.shuffle(samples)
lines = []
for s in samples:
    lines.append(f'{gt_dir}/{s}.wav|{feat_dir}/{s}.npy|{f0_dir}/{s}.wav.npy|{f0nsf_dir}/{s}.wav.npy|0')
with open(exp + '/filelist.txt','w',encoding='utf-8') as f:
    f.write('\\n'.join(lines))
print(f'wrote {len(lines)} entries to filelist.txt')
"

log "A.5 copy config"
cp $RVC_ROOT/configs/$VERSION/40k.json $A_EXPDIR/config.json

log "A.6 TRAIN (${EPOCHS} epochs, batch ${BATCH})"
cd $RVC_ROOT
$PY -u infer/modules/train/train.py \
    -e $A_EXP \
    -sr 40k \
    -f0 1 \
    -bs $BATCH \
    -g 0 \
    -te $EPOCHS \
    -se 25 \
    -pg $PRETR_G \
    -pd $PRETR_D \
    -l 0 -c 0 -sw 0 -v $VERSION 2>&1 | tail -30
cd -

log "A.7 build index"
$PY -u -c "
import sys, numpy as np, faiss, os, glob, traceback
exp = r'$A_EXPDIR'
feat_dir = exp + '/3_feature768'
feats = []
for f in sorted(glob.glob(feat_dir + '/*.npy'))[:2000]:
    try:
        x = np.load(f).astype(np.float32)
        feats.append(x)
    except Exception as e:
        print('skip', f, e)
big = np.concatenate(feats, axis=0)
print('feats shape', big.shape)
n_ivf = min(big.shape[0]//39, 256)
index = faiss.index_factory(big.shape[1], f'IVF{n_ivf},Flat')
index.train(big)
index.add(big)
idx_name = f'{exp}/added_IVF{n_ivf}_Flat_v2.index'
faiss.write_index(index, idx_name)
print('wrote', idx_name)
"

log "A done: RVC weights at $RVC_ROOT/assets/weights/${A_EXP}_*.pth"

# Generate audio with A's RVC
A_LATEST=$(ls -t $RVC_ROOT/assets/weights/${A_EXP}_*.pth 2>/dev/null | head -1)
if [ -n "$A_LATEST" ]; then
    log "=== Generating audio with ACE singer1 RVC: $A_LATEST ==="
    A_AUDIO=$V2/test_outputs/run8_final_ace_singer1_rvc
    mkdir -p $A_AUDIO
    # ZH focus (since this RVC is trained on Chinese singer)
    for ds in ace_opencpop kiritan ofuton; do
        if [ "$ds" = "ace_opencpop" ]; then
            dsname=ace_zh
            MIN=200; MAX=600
        else
            dsname=${ds}_ja
            MIN=400; MAX=1000
        fi
        log "  A_eval: $ds ($dsname)"
        $PY $V2/scripts/eval_audio.py \
            --model $RUN8_FINAL \
            --rvc-model $A_LATEST \
            --output-dir $A_AUDIO/$dsname \
            --processed-dir $V2/data/processed/$ds \
            --num-samples 3 --min-frames $MIN --max-frames $MAX \
            --device cuda 2>&1 | tail -5 || log "  WARN"
    done
fi

# =========================================================================
# Step B (optional): Train JA mixed 5-singer RVC
# =========================================================================
CURRENT_HOUR=$(date +%H)
if [ $SKIP_B -eq 1 ]; then
    log "B skipped (--skip-b)"
elif [ $CURRENT_HOUR -ge 9 ]; then
    log "B skipped (already past 9AM, user may be waking up)"
else
    log "=== STEP B: Train JA mixed 5-singer RVC ==="
    B_EXP=ja_mixed_5singer
    B_INPUT=$LOGS_DIR/$B_EXP/audio_raw
    B_EXPDIR=$LOGS_DIR/$B_EXP

    if [ ! -d "$B_INPUT" ]; then
        log "WARN: B wav data not found at $B_INPUT, skipping"
    else
        log "B.1 preprocess"
        $PY $RVC_ROOT/infer/modules/train/preprocess.py \
            $B_INPUT $SR 4 $B_EXPDIR False 3.7 2>&1 | tail -5

        log "B.2 F0"
        $PY $RVC_ROOT/infer/modules/train/extract/extract_f0_rmvpe.py \
            1 0 0 $B_EXPDIR True 2>&1 | tail -5

        log "B.3 features"
        $PY $RVC_ROOT/infer/modules/train/extract_feature_print.py \
            cuda 1 0 0 $B_EXPDIR $VERSION True 2>&1 | tail -5

        log "B.4 filelist"
        $PY -c "
import os, random
exp = r'$B_EXPDIR'
feat_dir = exp + '/3_feature768'
gt_dir = exp + '/0_gt_wavs'
f0_dir = exp + '/2a_f0'
f0nsf_dir = exp + '/2b-f0nsf'
samples = [f.replace('.wav','') for f in os.listdir(gt_dir)]
random.seed(0); random.shuffle(samples)
lines = []
for s in samples:
    lines.append(f'{gt_dir}/{s}.wav|{feat_dir}/{s}.npy|{f0_dir}/{s}.wav.npy|{f0nsf_dir}/{s}.wav.npy|0')
with open(exp + '/filelist.txt','w',encoding='utf-8') as f:
    f.write('\\n'.join(lines))
print(f'wrote {len(lines)} entries')
"
        cp $RVC_ROOT/configs/$VERSION/40k.json $B_EXPDIR/config.json

        # Shorter epochs if starting late
        B_EPOCHS=100
        CUR_HR=$(date +%H)
        if [ $CUR_HR -ge 7 ]; then B_EPOCHS=50; fi
        log "B.6 TRAIN ($B_EPOCHS epochs, starting at ${CUR_HR}:00)"
        cd $RVC_ROOT
        $PY -u infer/modules/train/train.py \
            -e $B_EXP -sr 40k -f0 1 -bs $BATCH -g 0 \
            -te $B_EPOCHS -se 25 \
            -pg $PRETR_G -pd $PRETR_D \
            -l 0 -c 0 -sw 0 -v $VERSION 2>&1 | tail -30
        cd -

        # skip index for B (time-limited)
        B_LATEST=$(ls -t $RVC_ROOT/assets/weights/${B_EXP}_*.pth 2>/dev/null | head -1)
        if [ -n "$B_LATEST" ]; then
            log "=== Generating audio with JA mixed RVC: $B_LATEST ==="
            B_AUDIO=$V2/test_outputs/run8_final_ja_mixed_rvc
            mkdir -p $B_AUDIO
            for ds in kiritan ofuton itako natsume oniku; do
                log "  B_eval: $ds"
                $PY $V2/scripts/eval_audio.py \
                    --model $RUN8_FINAL \
                    --rvc-model $B_LATEST \
                    --output-dir $B_AUDIO/${ds}_ja \
                    --processed-dir $V2/data/processed/$ds \
                    --num-samples 2 --min-frames 400 --max-frames 1000 \
                    --device cuda 2>&1 | tail -5 || log "  WARN"
            done
        fi
    fi
fi

log "=== ALL DONE ==="
log "Listen at:"
log "  C (baseline): $V2/test_outputs/run8_final_yachiyo/"
log "  A (ACE zh RVC): $V2/test_outputs/run8_final_ace_singer1_rvc/"
[ $SKIP_B -eq 0 ] && log "  B (JA mixed RVC): $V2/test_outputs/run8_final_ja_mixed_rvc/ (if trained)"
