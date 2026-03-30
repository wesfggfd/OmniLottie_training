#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/liblibai-models/user-workspace2/users/Sean_CHEN/OmniLottie_training"
LOG="$ROOT/gpu_watcher.log"
POLL_INTERVAL="${GPU_POLL_INTERVAL:-1}"
LOCK_DIR="${GPU_WATCHER_LOCK_DIR:-/tmp/omnilottie_gpu_claims}"
mkdir -p "$LOCK_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

claim_idle_gpu() {
    while true; do
        while IFS=',' read -r idx util mem; do
            idx="${idx// /}"
            util="${util// /}"
            mem="${mem// /}"

            if [[ -z "$idx" || "$idx" == "0" ]]; then
                continue
            fi
            if [[ "$util" != "0" || "$mem" != "0" ]]; then
                continue
            fi

            lock_file="$LOCK_DIR/gpu_${idx}.lock"
            if ( set -o noclobber; > "$lock_file" ) 2>/dev/null; then
                echo "$idx"
                return 0
            fi
        done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)

        log "No non-gpu0 idle GPU available (util=0%, mem=0 MiB); retrying in ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

GPU_ID=$(claim_idle_gpu)
trap 'rm -f "$LOCK_DIR/gpu_'"${GPU_ID}"'.lock"' EXIT

log "Claimed GPU ${GPU_ID}. Holding it until this watcher exits."
log "If you want to pin a process immediately, launch it with: CUDA_VISIBLE_DEVICES=${GPU_ID} ..."

while true; do
    sleep "$POLL_INTERVAL"
done
