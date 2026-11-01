#!/usr/bin/env bash
#SBATCH --job-name=pusht-official-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPU GRES, GPU partition, memory, time and output are set in submit_stage1.sh.
# Submit only after the sandbox preparation wrote Gate1/Gate2 receipts.
set -Eeuo pipefail
PUSHT_REPO="${PUSHT_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PUSHT_REPO
source "$PUSHT_REPO/jobs/lib.sh"
job_start official_eval
[[ "${PUSHT_CONFIRM_GPU_EVAL:-}" = YES ]] || { echo "GPU evaluation not explicitly approved" >&2; exit 2; }
gate_require gate1
gate_require gate2
PUSHT_STEP="gpu_cuda_smoke"
activate_env
check_gpu

PUSHT_STEP="two_trial_cem_smoke"
python -m scripts.repro_pusht.evaluate \
  --audit-dir "$PUSHT_RUN_DIR/audits/official_smoke" \
  --execute "policy=$PUSHT_RUN_DIR/official_pusht/weights.pt" eval.num_eval=2
test -f "$PUSHT_RUN_DIR/audits/official_smoke/completed.json"

PUSHT_STEP="official_50_trial_cem"
python -m scripts.repro_pusht.evaluate \
  --audit-dir "$PUSHT_RUN_DIR/audits/official_50" \
  --execute "policy=$PUSHT_RUN_DIR/official_pusht/weights.pt"
python - <<'PY'
import json,os,pathlib
root=pathlib.Path(os.environ["PUSHT_RUN_DIR"])/"audits/official_50"
selection=json.loads((root/"selection.json").read_text())
result=json.loads((root/"completed.json").read_text())
assert selection["episode_count"]==50 and len(selection["rows"])==50
assert len(selection["episodes"])==len(selection["start_steps"])==50
assert result["phase"]=="world_evaluate_returned" and "metrics" in result
print("OFFICIAL 50-TRIAL METRICS:",json.dumps(result["metrics"],default=str))
PY
gate_pass gate3
echo "Official checkpoint evaluated. Stop here for review; do not start training automatically."
