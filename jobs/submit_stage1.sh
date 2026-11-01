#!/usr/bin/env bash
# Stage 1 on CHPC:
#   --prep    install/download/convert/verify interactively on chpc-sandbox
#   --submit  submit the SINGLE gated GPU official-checkpoint evaluation
#   --plan    print both steps; never install, download or call sbatch
#
# Why preparation is not a Slurm job here: on this cluster a CPU-only sbatch job
# never leaves QOS=init, and compute nodes have no external network. The
# preparation needs pip, git and a 13.14 GB Hugging Face download, so it runs on
# chpc-sandbox (inside tmux) where those are allowed. Only the GPU stage is
# submitted; it re-checks the Gate1/Gate2 receipts before touching the GPU.
set -Eeuo pipefail
mode="${1:---plan}"
case "$mode" in
  --plan|--prep|--submit) ;;
  *) echo "Usage: bash jobs/submit_stage1.sh [--plan|--prep|--submit]" >&2; exit 2 ;;
esac

repo="$(cd "$(dirname "$0")/.." && pwd)"
export PUSHT_REPO="$repo"
[[ -n "${STABLEWM_HOME:-}" && "$STABLEWM_HOME" = /* ]] || {
  echo "Set STABLEWM_HOME to approved absolute persistent project storage (CHPC campus quota is 1T TOTAL across all projects)" >&2; exit 2;
}
[[ -d "$STABLEWM_HOME" ]] || { echo "Create/confirm the approved STABLEWM_HOME first" >&2; exit 2; }
export PUSHT_CODE_SHA="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" diff --quiet 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac HEAD -- train.py eval.py jepa.py module.py utils.py config || {
  echo "Author scientific files have changed; refuse submission" >&2; exit 2;
}

run_id_file="$STABLEWM_HOME/workflows/.stage1_run_id"
if [[ -z "${PUSHT_RUN_ID:-}" && -f "$run_id_file" ]]; then
  PUSHT_RUN_ID="$(cat "$run_id_file")"
fi
if [[ -z "${PUSHT_RUN_ID:-}" ]]; then
  PUSHT_RUN_ID="pusht_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
export PUSHT_RUN_ID
[[ "$PUSHT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid PUSHT_RUN_ID" >&2; exit 2; }
export PUSHT_RUN_DIR="$STABLEWM_HOME/workflows/$PUSHT_RUN_ID"

gpu_time="${PUSHT_GPU1_TIME:-12:00:00}"
gpu_mem="${PUSHT_GPU_MEM:-64G}"
gpu_cores="${PUSHT_GPU_CORES:-8}"
gpu_request="${PUSHT_GPU_GRES:-gpu:1}"

echo "PLAN stage1: run=$PUSHT_RUN_ID"
echo "HEAD=$PUSHT_CODE_SHA"
echo "Step 1 (chpc-sandbox, interactive, NOT sbatch): bash jobs/submit_stage1.sh --prep"
echo "Step 2 (GPU): partition=${PUSHT_GPU_PARTITION:-<set PUSHT_GPU_PARTITION>} GRES=$gpu_request cpus=$gpu_cores mem=$gpu_mem time=$gpu_time"
echo "Persistent run=$PUSHT_RUN_DIR"
if [[ "$mode" = --plan ]]; then echo "PLAN ONLY: no installs, downloads or sbatch calls made."; exit 0; fi

if [[ "$mode" = --prep ]]; then
  [[ "${PUSHT_CONFIRM_STORAGE:-}" = YES && "${PUSHT_CONFIRM_DATA_DOWNLOAD:-}" = YES ]] || {
    echo "Required: PUSHT_CONFIRM_STORAGE=YES PUSHT_CONFIRM_DATA_DOWNLOAD=YES" >&2; exit 2;
  }
  case "$(hostname -s)" in
    *login*) [[ "${PUSHT_ALLOW_LOGIN_NODE:-}" = YES ]] || {
      echo "Refuse heavy preparation on the login node (its processes are killed). Run on chpc-sandbox; override with PUSHT_ALLOW_LOGIN_NODE=YES only on a node with network and enough CPU." >&2; exit 2;
    } ;;
  esac
  for gate in gate1 gate2; do
    [[ ! -e "$PUSHT_RUN_DIR/gates/$gate.json" ]] || {
      echo "$gate is already recorded for run $PUSHT_RUN_ID; use --prep only for an incomplete run, otherwise pass a new PUSHT_RUN_ID" >&2; exit 2;
    }
  done
  command -v uv >/dev/null || { echo "uv not found; enable it before preparation" >&2; exit 2; }
  mkdir -p "$STABLEWM_HOME/workflows"
  printf '%s\n' "$PUSHT_RUN_ID" > "$run_id_file"
  echo "Preparing on host=$(hostname -s) run=$PUSHT_RUN_ID"
  df -h "$STABLEWM_HOME" || true
  bash "$repo/jobs/01_prepare_cpu.sbatch"
  echo "Preparation finished. Review $PUSHT_RUN_DIR, then submit the GPU evaluation:"
  echo "  PUSHT_CONFIRM_GPU_EVAL=YES bash jobs/submit_stage1.sh --submit"
  exit 0
fi

# --submit: exactly one GPU job, and only after Gate1/Gate2 receipts exist.
[[ "${PUSHT_CONFIRM_GPU_EVAL:-}" = YES ]] || {
  echo "Required: PUSHT_CONFIRM_GPU_EVAL=YES (set only after reviewing Gate1/Gate2)" >&2; exit 2;
}
[[ -n "${PUSHT_GPU_PARTITION:-}" ]] || { echo "Set PUSHT_GPU_PARTITION (CHPC: cse_24h)" >&2; exit 2; }
[[ -d "$PUSHT_RUN_DIR" ]] || { echo "Stage1 run directory missing; run --prep first" >&2; exit 2; }
[[ ! -e "$PUSHT_RUN_DIR/status/submit_stage1.txt" ]] || {
  echo "Stage1 GPU evaluation already submitted for run $PUSHT_RUN_ID; refuse duplicate" >&2; exit 2;
}
python3 - "$PUSHT_RUN_DIR" "$PUSHT_CODE_SHA" <<'PY'
import json,sys,pathlib
root,commit=pathlib.Path(sys.argv[1]),sys.argv[2]
for name in ("gate1","gate2"):
    with (root/"gates"/f"{name}.json").open() as f:d=json.load(f)
    assert d["status"]=="passed" and d["commit_sha"]==commit
print("Gate1/Gate2 receipts verified for",commit)
PY
command -v sbatch >/dev/null
mkdir -p "$PUSHT_RUN_DIR/logs" "$PUSHT_RUN_DIR/status"
common=(--parsable --nodes=1 --ntasks=1 --chdir="$repo" --export=ALL)
if [[ -n "${PUSHT_ACCOUNT:-}" ]]; then common+=(--account="$PUSHT_ACCOUNT"); fi
gpu_job="$(sbatch "${common[@]}" --partition="$PUSHT_GPU_PARTITION" \
  --gres="$gpu_request" --cpus-per-task="$gpu_cores" --mem="$gpu_mem" --time="$gpu_time" \
  --output="$PUSHT_RUN_DIR/logs/gpu-official-%j.out" \
  "$repo/jobs/02_official_eval_gpu.sbatch")"
gpu_job="${gpu_job%%;*}"
[[ "$gpu_job" =~ ^[0-9]+$ ]] || { echo "Invalid GPU job id" >&2; exit 2; }
printf 'gpu_job=%s\nrun_id=%s\ncommit=%s\n' "$gpu_job" "$PUSHT_RUN_ID" "$PUSHT_CODE_SHA" \
  > "$PUSHT_RUN_DIR/status/submit_stage1.txt"
echo "GPU evaluation submitted: $gpu_job"
echo "SAVE THE RUN ID: $PUSHT_RUN_ID"
echo "Monitor: squeue -j $gpu_job"
