#!/bin/bash
set -e

ROOT=/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training
CODE=$ROOT/OmniLottie
DATA_MERGED=/opt/liblibai-models/user-workspace2/dataset/MMLottie-2M/data/Lottie_merged_70_30
PYTHON=/opt/liblibai-models/user-workspace2/anaconda3/envs/omnilottie_qwen35/bin/python
ACCELERATE=/opt/liblibai-models/user-workspace2/anaconda3/envs/omnilottie_qwen35/bin/accelerate

GPUS=${GPUS:-1,2,4,5,6}
NPROC=${NPROC:-5}
MODEL=Qwen/Qwen3.5-9B
MIXED_RATIO_STRATEGY=${MIXED_RATIO_STRATEGY:-adaptive_stage_loss}

STAGE=${1:-1}   # pass 1/2/3/4 to start from a specific stage

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_stage() {
    local STAGE_NUM=$1
    local TASK_MODE=$2
    local OUT_DIR=$3
    local INIT_WEIGHTS=$4   # empty = fresh; non-empty = warm-start from previous stage

    log "======== STAGE $STAGE_NUM: $TASK_MODE ========"
    mkdir -p "$OUT_DIR"

    EXTRA_ARGS=()
    if [ -n "$INIT_WEIGHTS" ]; then
        EXTRA_ARGS+=(--init_weights "$INIT_WEIGHTS")
    fi
    if [ "$TASK_MODE" = "mixed" ]; then
        EXTRA_ARGS+=(--mixed_ratio_strategy "$MIXED_RATIO_STRATEGY" --mixed_ratio_stage_root "$ROOT/outputs")
    fi

    CUDA_VISIBLE_DEVICES=$GPUS \
    PYTHONPATH=$CODE \
    $ACCELERATE launch \
        --num_processes $NPROC \
        --mixed_precision bf16 \
        $CODE/train.py \
            --model_path "$MODEL" \
            --data_path "$DATA_MERGED" \
            --output_dir "$OUT_DIR" \
            --task_mode "$TASK_MODE" \
            --max_seq_len 4096 \
            --num_epochs 5 \
            --per_device_batch 2 \
            --grad_accum 1 \
            --max_steps_per_epoch 10000 \
            --lora_rank 64 \
            --lora_alpha 128 \
            --lora_dropout 0.05 \
            --lora_lr 2e-5 \
            --lottie_lr 5e-4 \
            --save_steps 2000 \
            --eval_steps 1000 \
            --logging_steps 20 \
            --warmup_ratio 0.03 \
            --num_workers 4 \
            --seed 42 \
            --early_stopping_patience 5 \
            --skip_audit \
            "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${OUT_DIR}/train.log"

    log "Stage $STAGE_NUM complete. Best weights: $OUT_DIR/best"
}

# Stage 1 — text-to-lottie
if [ "$STAGE" -le 1 ]; then
    run_stage 1 text \
        "$ROOT/outputs_stage1_text" \
        ""
fi

# Stage 2 — text+image-to-lottie
if [ "$STAGE" -le 2 ]; then
    run_stage 2 image \
        "$ROOT/outputs_stage2_image" \
        "$ROOT/outputs_stage1_text/best"
fi

# Stage 3 — video-to-lottie
if [ "$STAGE" -le 3 ]; then
    run_stage 3 video \
        "$ROOT/outputs_stage3_video" \
        "$ROOT/outputs_stage2_image/best"
fi

# Stage 4 — mixed
if [ "$STAGE" -le 4 ]; then
    run_stage 4 mixed \
        "$ROOT/outputs_stage4_mixed" \
        "$ROOT/outputs_stage3_video/best"
fi

log "All stages complete."
