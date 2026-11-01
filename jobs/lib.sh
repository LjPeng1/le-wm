#!/usr/bin/env bash
# Shared, fail-closed helpers for the opt-in Slurm jobs. Never submit or allocate here.
set -Eeuo pipefail
export SLURM_EXPORT_ENV=ALL

need_env() {
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then echo "ERROR: export $name first" >&2; return 2; fi
  done
}
resolve_paths() {
  need_env STABLEWM_HOME PUSHT_RUN_ID PUSHT_CODE_SHA PUSHT_REPO
  [[ "$STABLEWM_HOME" = /* && "$PUSHT_REPO" = /* ]] || { echo "Use absolute persistent paths" >&2; return 2; }
  [[ "$PUSHT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid PUSHT_RUN_ID" >&2; return 2; }
  export PUSHT_RUN_DIR="$STABLEWM_HOME/workflows/$PUSHT_RUN_ID"
  export PUSHT_VENV="$STABLEWM_HOME/venvs/lewm_pusht_py310"
}
check_source() {
  local observed
  observed="$(git -C "$PUSHT_REPO" rev-parse HEAD)"
  [[ "$observed" = "$PUSHT_CODE_SHA" ]] || { echo "Git checkout moved: $observed != $PUSHT_CODE_SHA" >&2; return 2; }
  git -C "$PUSHT_REPO" diff --quiet HEAD -- train.py eval.py jepa.py module.py utils.py config || {
    echo "Tracked reference scientific files have local modifications" >&2; return 2;
  }
  git -C "$PUSHT_REPO" diff --quiet 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac HEAD -- train.py eval.py jepa.py module.py utils.py config || {
    echo "Author reference training/evaluation/config files changed" >&2; return 2;
  }
}
job_start() {
  resolve_paths
  need_env SLURM_JOB_ID
  export PUSHT_STAGE="$1"
  export PUSHT_STEP="initializing"
  mkdir -p "$PUSHT_RUN_DIR/status" "$PUSHT_RUN_DIR/gates" "$PUSHT_RUN_DIR/audits"
  export PUSHT_STATUS="$PUSHT_RUN_DIR/status/$PUSHT_STAGE.$SLURM_JOB_ID.json"
  [[ ! -e "$PUSHT_STATUS" ]] || { echo "Receipt already exists: $PUSHT_STATUS" >&2; return 2; }
  trap 'job_finish $?' EXIT
  echo "stage=$PUSHT_STAGE slurm_job=$SLURM_JOB_ID run=$PUSHT_RUN_ID date=$(date -u +%FT%TZ)"
  check_source
  cd "$PUSHT_REPO"
}
job_finish() {
  local rc="$1"
  trap - EXIT
  python3 - "$PUSHT_STATUS" "$rc" "$PUSHT_STEP" <<'PY' || true
import json,os,sys,time
path,rc,step=sys.argv[1:]
payload={"stage":os.environ.get("PUSHT_STAGE"),"run_id":os.environ.get("PUSHT_RUN_ID"),
 "job_id":os.environ.get("SLURM_JOB_ID"),"git_commit":os.environ.get("PUSHT_CODE_SHA"),
 "status":"passed" if int(rc)==0 else "failed","exit_code":int(rc),
 "last_step":step,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
with open(path,"x",encoding="utf-8") as f: json.dump(payload,f,indent=2);f.write("\n")
PY
  echo "stage=$PUSHT_STAGE exit=$rc last_step=$PUSHT_STEP"
}
gate_pass() {
  local name="$1"
  python3 - "$PUSHT_RUN_DIR/gates/$name.json" "$name" <<'PY'
import json,os,sys,time
path,name=sys.argv[1:]
value={"gate":name,"status":"passed","run_id":os.environ["PUSHT_RUN_ID"],
 "commit_sha":os.environ["PUSHT_CODE_SHA"],"slurm_job_id":os.environ["SLURM_JOB_ID"],
 "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
with open(path,"x",encoding="utf-8") as f:json.dump(value,f,indent=2);f.write("\n")
PY
}
gate_require() {
  python3 - "$PUSHT_RUN_DIR/gates/$1.json" "$1" <<'PY'
import json,os,sys
path,gate=sys.argv[1:]
with open(path,encoding="utf-8") as f: d=json.load(f)
assert d.get("gate")==gate and d.get("status")=="passed"
assert d.get("run_id")==os.environ["PUSHT_RUN_ID"]
assert d.get("commit_sha")==os.environ["PUSHT_CODE_SHA"]
print("PASS predecessor:",gate)
PY
}
activate_env() {
  [[ -x "$PUSHT_VENV/bin/python" ]] || { echo "Missing venv created by CPU gate: $PUSHT_VENV" >&2; return 2; }
  source "$PUSHT_VENV/bin/activate"
  export WANDB_MODE=offline
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
  export STABLEWM_HOME
  cd "$PUSHT_REPO"
}
check_gpu() {
  command -v nvidia-smi >/dev/null
  nvidia-smi
  python - <<'PY'
import torch
assert torch.cuda.is_available(),"Slurm GPU allocation has no usable CUDA device"
print("CUDA GPU:",torch.cuda.get_device_name(0),"Torch:",torch.__version__)
PY
}
