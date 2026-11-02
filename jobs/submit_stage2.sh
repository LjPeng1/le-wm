#!/usr/bin/env bash
# Submit the approved ONE GPU allocation: real resume qualification, paper10, eval.
set -Eeuo pipefail
mode="${1:---plan}"
[[ "$mode" = --plan || "$mode" = --submit ]] || { echo "Usage: bash jobs/submit_stage2.sh [--plan|--submit]" >&2; exit 2; }
repo="$(cd "$(dirname "$0")/.." && pwd)"
export PUSHT_REPO="$repo"
: "${STABLEWM_HOME:?Set the approved absolute project storage root}"
: "${PUSHT_RUN_ID:?Set the stage1 PUSHT_RUN_ID}"
: "${PUSHT_GPU_PARTITION:?Set the approved GPU Slurm partition}"
[[ "$STABLEWM_HOME" = /* && "$PUSHT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
export PUSHT_RUN_DIR="$STABLEWM_HOME/workflows/$PUSHT_RUN_ID"
[[ -d "$PUSHT_RUN_DIR" ]] || { echo "Stage1 run directory missing" >&2; exit 2; }
export PUSHT_CODE_SHA="$(git -C "$repo" rev-parse HEAD)"
python3 - "$PUSHT_RUN_DIR" "$PUSHT_CODE_SHA" <<'PY'
import json,sys,pathlib
root,commit=pathlib.Path(sys.argv[1]),sys.argv[2]
# Gate3 is deliberately NOT required: the user chose to run Gate5 in parallel
# with the Gate3 reference evaluation on 2026-11-01. Gate1/Gate2 still gate
# this submission because they bind the data and model bytes.
for name in ("gate1","gate2"):
  with (root/"gates"/f"{name}.json").open() as f:d=json.load(f)
  assert d["status"]=="passed" and d["commit_sha"]==commit
print("Gate1/Gate2 receipts verified; Gate3 runs in parallel by explicit user decision")
PY
[[ ! -e "$PUSHT_RUN_DIR/status/submit_stage2.txt" ]] || { echo "Stage2 already submitted; refuse duplicate" >&2; exit 2; }
gpu_time="${PUSHT_GPU2_TIME:-12:00:00}"
gpu_mem="${PUSHT_GPU_MEM:-64G}"
gpu_cores="${PUSHT_GPU_CORES:-8}"
gpu_request="${PUSHT_GPU_GRES:-gpu:1}"
echo "PLAN stage2: run=$PUSHT_RUN_ID HEAD=$PUSHT_CODE_SHA"
echo "GPU partition=$PUSHT_GPU_PARTITION GRES=$gpu_request cpus=$gpu_cores mem=$gpu_mem time=$gpu_time"
echo "Single Slurm GPU allocation: real resume A+B1+B2 -> compare -> paper10 -> final 50 eval."
if [[ "$mode" = --plan ]]; then echo "PLAN ONLY: no sbatch calls made."; exit 0; fi
[[ "${PUSHT_CONFIRM_GPU_TRAIN:-}" = YES && "${PUSHT_CONFIRM_10EPOCH:-}" = YES ]] || {
  echo "Separate explicit approvals required: PUSHT_CONFIRM_GPU_TRAIN=YES PUSHT_CONFIRM_10EPOCH=YES" >&2; exit 2;
}
command -v sbatch >/dev/null
common=(--parsable --nodes=1 --ntasks=1 --chdir="$repo" --export=ALL)
if [[ -n "${PUSHT_ACCOUNT:-}" ]]; then common+=(--account="$PUSHT_ACCOUNT"); fi
gpu_job="$(sbatch "${common[@]}" --partition="$PUSHT_GPU_PARTITION" --gres="$gpu_request" \
  --cpus-per-task="$gpu_cores" --mem="$gpu_mem" --time="$gpu_time" \
  --output="$PUSHT_RUN_DIR/logs/gpu-paper10-%j.out" \
  "$repo/jobs/03_resume_train_eval_gpu.sbatch")"
gpu_job="${gpu_job%%;*}"
[[ "$gpu_job" =~ ^[0-9]+$ ]] || exit 2
printf 'gpu_job=%s\nrun_id=%s\ncommit=%s\n' "$gpu_job" "$PUSHT_RUN_ID" "$PUSHT_CODE_SHA" \
  > "$PUSHT_RUN_DIR/status/submit_stage2.txt"
echo "Stage2 GPU job submitted: $gpu_job"
echo "No extra seed, decoder, or 100-epoch training scheduled."
