#!/usr/bin/env bash
# =============================================================
# OmniLottie 4-Stage Training Pipeline
# Stage 1: text-to-lottie
# Stage 2: text-image-to-lottie  (init from Stage 1 best)
# Stage 3: video-to-lottie        (init from Stage 2 best)
# Stage 4: mixed mode             (init from Stage 3 best)
# =============================================================
set -euo pipefail

PY=/opt/liblibai-models/user-workspace2/anaconda3/envs/omnilottie_qwen35/bin/python
ACCELERATE=/opt/liblibai-models/user-workspace2/anaconda3/envs/omnilottie_qwen35/bin/accelerate
TRAIN=/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training/OmniLottie/train.py
DATA=/opt/liblibai-models/user-workspace2/dataset/MMLottie-2M/data/Lottie_merged_70_30
OUT_ROOT=/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training/outputs
MODEL=Qwen/Qwen3.5-9B
MIXED_RATIO_STRATEGY=${MIXED_RATIO_STRATEGY:-adaptive_stage_loss}
select_idle_gpus() {
    local selected=()
    while IFS=',' read -r idx mem_used util; do
        idx=${idx// /}
        util=${util// /}
        if [ "$idx" = "0" ]; then
            continue
        fi
        if [ "$util" = "0" ]; then
            selected+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

    if [ ${#selected[@]} -eq 0 ]; then
        echo "No idle GPUs available outside GPU 0." >&2
        exit 1
    fi

    local joined
    joined=$(IFS=,; printf '%s' "${selected[*]}")
    printf '%s\n' "$joined"
}

if [ -z "${GPUS:-}" ]; then
    GPUS=$(select_idle_gpus)
fi
NPROC=${NPROC:-$(awk -F',' '{print NF}' <<< "$GPUS")}

# Hyperparameters
MAX_SEQ_LEN=4096
PER_DEVICE_BATCH=2
LORA_RANK=64
LORA_ALPHA=128
LORA_LR=2e-5
LOTTIE_LR=5e-4
SAVE_STEPS=500
EVAL_STEPS=250
LOGGING_STEPS=10
WARMUP_RATIO=0.03
NUM_WORKERS=4

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_stage() {
    local stage_num=$1
    local task_mode=$2
    local num_epochs=$3
    local init_from=$4   # empty string = no init
    local out_dir="${OUT_ROOT}/stage${stage_num}_${task_mode//-/_}"
    local log_file="${OUT_ROOT}/stage${stage_num}_${task_mode//-/_}.log"

    log "======================================================"
    log "Starting Stage ${stage_num}: ${task_mode}  (${num_epochs} epochs)"
    log "  Output: ${out_dir}"
    log "  Log:    ${log_file}"
    if [ -n "${init_from}" ]; then
        log "  Init weights from: ${init_from}"
    fi
    log "======================================================"
    log "  Audit: skipped during staged training (use valid_auditing.sh for standalone audit)"

    # Build extra args
    local extra=()
    if [ -n "${init_from}" ]; then
        extra+=(--init_weights "${init_from}")
    fi

    # map task_mode to train.py --task_mode value
    local tm
    case "${task_mode}" in
        text-to-lottie)        tm=text  ;;
        text-image-to-lottie)  tm=image ;;
        video-to-lottie)       tm=video ;;
        mixed)                 tm=mixed ;;
        *) echo "Unknown task_mode: ${task_mode}"; exit 1 ;;
    esac

    local cmd=(
        ${ACCELERATE} launch
        --num_processes ${NPROC}
        --mixed_precision bf16
        ${TRAIN}
        --model_path "${MODEL}"
        --data_path "${DATA}"
        --output_dir "${out_dir}"
        --task_mode "${tm}"
        --max_seq_len ${MAX_SEQ_LEN}
        --num_epochs ${num_epochs}
        --per_device_batch ${PER_DEVICE_BATCH}
        --lora_rank ${LORA_RANK}
        --lora_alpha ${LORA_ALPHA}
        --lora_lr ${LORA_LR}
        --lottie_lr ${LOTTIE_LR}
        --save_steps ${SAVE_STEPS}
        --eval_steps ${EVAL_STEPS}
        --logging_steps ${LOGGING_STEPS}
        --warmup_ratio ${WARMUP_RATIO}
        --num_workers ${NUM_WORKERS}
        --seed 42
        --skip_audit
    )
    if [ "${tm}" = "mixed" ]; then
        cmd+=(--mixed_ratio_strategy "${MIXED_RATIO_STRATEGY}" --mixed_ratio_stage_root "${OUT_ROOT}")
    fi
    cmd+=("${extra[@]}")

    CUDA_VISIBLE_DEVICES=${GPUS} \
    PYTHONPATH=/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training/OmniLottie:${PYTHONPATH:-} \
    "${cmd[@]}" \
        2>&1 | tee "${log_file}"

    log "Stage ${stage_num} complete. Best checkpoint at: ${out_dir}/best"
}

mkdir -p "${OUT_ROOT}"

# --- Stage 1: text-to-lottie ---
run_stage 1 "text-to-lottie" 3 ""

STAGE1_BEST="${OUT_ROOT}/stage1_text_to_lottie/best"

# --- Stage 2: text-image-to-lottie ---
run_stage 2 "text-image-to-lottie" 3 "${STAGE1_BEST}"

STAGE2_BEST="${OUT_ROOT}/stage2_text_image_to_lottie/best"

# --- Stage 3: video-to-lottie ---
run_stage 3 "video-to-lottie" 3 "${STAGE2_BEST}"

STAGE3_BEST="${OUT_ROOT}/stage3_video_to_lottie/best"

# --- Stage 4: mixed mode ---
run_stage 4 "mixed" 5 "${STAGE3_BEST}"

log "All 4 stages completed successfully!"
log "Final model: ${OUT_ROOT}/stage4_mixed/best"
