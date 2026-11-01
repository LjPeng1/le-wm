#!/usr/bin/env bash
# Submit stage1 CPU preparation and dependent GPU official-checkpoint evaluation.
# Default is PLAN ONLY. Explicit --submit + approval flags are required.
set -Eeuo pipefail
mode="${1:---plan}"
[[ "$mode" = --plan || "$mode" = --submit ]] || { echo "Usage: bash jobs/submit_stage1.sh [--plan|--submit]" >&2; exit 2; }
repo="$(cd "$(dirname "$0")/.." && pwd)"
export PUSHT_REPO="$repo"
[[ -n "${STABLEWM_HOME:-}" && "$STABLEWM_HOME" = /* ]] || { echo "Set STABLEWM_HOME to approved absolute persistent project storage" >&2; exit 2; }
[[ -n "${PUSHT_GPU_PARTITION:-}" ]] || { echo "Set PUSHT_GPU_PARTITION after jobs/00_inventory.sh" >&2; exit 2; }
[[ -d "$STABLEWM_HOME" ]] || { echo "Create/confirm the approved STABLEWM_HOME before submission" >&2; exit 2; }
export PUSHT_CODE_SHA="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" diff --quiet 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac HEAD -- train.py eval.py jepa.py module.py utils.py config || {
  echo "Author scientific files have changed; refuse submission" >&2; exit 2;
}
if [[ -z "${PUSHT_RUN_ID:-}" ]]; then
  export PUSHT_RUN_ID="pusht_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
[[ "$PUSHT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
export PUSHT_RUN_DIR="$STABLEWM_HOME/workflows/$PUSHT_RUN_ID"
cpu_time="${PUSHT_CPU_TIME:-08:00:00}"
gpu_time="${PUSHT_GPU1_TIME:-12:00:00}"
cpu_mem="${PUSHT_CPU_MEM:-64G}"
gpu_mem="${PUSHT_GPU_MEM:-64G}"
cpu_cores="${PUSHT_CPU_CORES:-8}"
gpu_cores="${PUSHT_GPU_CORES:-8}"
gpu_request="${PUSHT_GPU_GRES:-gpu:1}"

echo "PLAN stage1: run=$PUSHT_RUN_ID"
echo "HEAD=$PUSHT_CODE_SHA"
echo "CPU partition=${PUSHT_CPU_PARTITION:-cluster default}, cpus=$cpu_cores, mem=$cpu_mem, time=$cpu_time"
echo "GPU partition=$PUSHT_GPU_PARTITION, GRES=$gpu_request, cpus=$gpu_cores, mem=$gpu_mem, time=$gpu_time"
echo "Persistent run=$PUSHT_RUN_DIR"
echo "CPU -> afterok: GPU official two-episode smoke + 50-episode evaluation"
if [[ "$mode" = --plan ]]; then echo "PLAN ONLY: no sbatch calls made."; exit 0; fi
[[ "${PUSHT_CONFIRM_STORAGE:-}" = YES && "${PUSHT_CONFIRM_DATA_DOWNLOAD:-}" = YES && "${PUSHT_CONFIRM_GPU_EVAL:-}" = YES ]] || {
  echo "Required: PUSHT_CONFIRM_STORAGE=YES, PUSHT_CONFIRM_DATA_DOWNLOAD=YES, PUSHT_CONFIRM_GPU_EVAL=YES" >&2; exit 2;
}
command -v sbatch >/dev/null
[[ ! -e "$PUSHT_RUN_DIR" ]] || { echo "Workflow output path exists; never overwrite" >&2; exit 2; }
mkdir -p "$PUSHT_RUN_DIR/logs" "$PUSHT_RUN_DIR/status"
common=(--parsable --nodes=1 --ntasks=1 --chdir="$repo" --export=ALL)
if [[ -n "${PUSHT_ACCOUNT:-}" ]]; then common+=(--account="$PUSHT_ACCOUNT"); fi
cpu_partition=()
if [[ -n "${PUSHT_CPU_PARTITION:-}" ]]; then cpu_partition=(--partition="$PUSHT_CPU_PARTITION"); fi
cpu_job="$(sbatch "${common[@]}" "${cpu_partition[@]}" \
  --cpus-per-task="$cpu_cores" --mem="$cpu_mem" --time="$cpu_time" \
  --output="$PUSHT_RUN_DIR/logs/cpu-%j.out" "$repo/jobs/01_prepare_cpu.sbatch")"
cpu_job="${cpu_job%%;*}"
[[ "$cpu_job" =~ ^[0-9]+$ ]] || { echo "Invalid CPU job id" >&2; exit 2; }
printf 'cpu_job=%s\n' "$cpu_job" > "$PUSHT_RUN_DIR/status/submit_stage1.txt"
echo "CPU job submitted: $cpu_job"
gpu_job="$(sbatch "${common[@]}" --partition="$PUSHT_GPU_PARTITION" \
  --gres="$gpu_request" --cpus-per-task="$gpu_cores" --mem="$gpu_mem" --time="$gpu_time" \
  --dependency="afterok:$cpu_job" --output="$PUSHT_RUN_DIR/logs/gpu-official-%j.out" \
  "$repo/jobs/02_official_eval_gpu.sbatch")" || {
   echo "GPU sbatch failed after CPU job $cpu_job was submitted; investigate/cancel CPU job if needed" >&2; exit 2;
}
gpu_job="${gpu_job%%;*}"
[[ "$gpu_job" =~ ^[0-9]+$ ]] || exit 2
printf 'gpu_job=%s\nrun_id=%s\ncommit=%s\n' "$gpu_job" "$PUSHT_RUN_ID" "$PUSHT_CODE_SHA" \
  >> "$PUSHT_RUN_DIR/status/submit_stage1.txt"
echo "GPU job submitted with afterok:$cpu_job dependency: $gpu_job"
echo "SAVE THE RUN ID: $PUSHT_RUN_ID"
echo "Monitor: squeue -j $cpu_job,$gpu_job"
