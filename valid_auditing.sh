#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training"
AUDIT_TASK_MODE="${AUDIT_TASK_MODE:-text}"
OUT_DIR="$SCRIPT_DIR/outputs_audit_${AUDIT_TASK_MODE}"
POLL_INTERVAL="${GPU_POLL_INTERVAL:-1}"
mkdir -p "$OUT_DIR"

select_idle_gpu() {
  while true; do
    mapfile -t idle_gpus < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 != 0 && $2 == 0 && $3 == 0 {print $1}')

    if (( ${#idle_gpus[@]} > 0 )); then
      echo "${idle_gpus[0]}"
      return 0
    fi

    echo "No non-gpu0 idle GPU found (util=0%, mem=0 MiB); retrying in ${POLL_INTERVAL}s..." >&2
    sleep "$POLL_INTERVAL"
  done
}

GPU_ID=$(select_idle_gpu)
echo "Using GPU ${GPU_ID}" >&2
export CUDA_VISIBLE_DEVICES="$GPU_ID"

EXTRA_ARGS=()
if [ "$AUDIT_TASK_MODE" = "mixed" ]; then
  EXTRA_ARGS+=(--mixed_ratio_strategy adaptive_stage_loss --mixed_ratio_stage_root "$SCRIPT_DIR/outputs")
fi

accelerate launch --num_processes 1 --main_process_port 0 --mixed_precision bf16 "$SCRIPT_DIR/OmniLottie/train.py" \
  --model_path Qwen/Qwen3.5-9B \
  --data_path /opt/liblibai-models/user-workspace2/dataset/MMLottie-2M/data/Lottie_merged_70_30 \
  --output_dir "$OUT_DIR" \
  --task_entrypoint "$AUDIT_TASK_MODE" \
  --audit_only \
  --valid_indices_path "$OUT_DIR/valid_indices.json" \
  --audit_max_samples 2000 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/audit.log"
