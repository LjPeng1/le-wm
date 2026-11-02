#!/usr/bin/env bash
#SBATCH --job-name=pusht-recover-paper10
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPU/time/memory/partition supplied by submit_stage2.sh. Requires separate approval.
set -Eeuo pipefail
PUSHT_REPO="${PUSHT_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PUSHT_REPO
source "$PUSHT_REPO/jobs/lib.sh"
job_start resume_and_paper10
[[ "${PUSHT_CONFIRM_GPU_TRAIN:-}" = YES && "${PUSHT_CONFIRM_10EPOCH:-}" = YES ]] || {
  echo "Recovery and 10-epoch GPU training require explicit approval" >&2; exit 2;
}
gate_require gate1
gate_require gate2
# Gate3 (the released-checkpoint reference evaluation) is NOT required here:
# on 2026-11-01 the user explicitly chose to run Gate5 in parallel with Gate3
# to save wall-clock. Gate5 training does not depend on Gate3's result; its
# final evaluation uses the same fixed eval path as Gate3. See
# $STABLEWM_HOME/ops/STAGE2_GATE4_SKIPPED.md.
activate_env
check_gpu
DATA_REPORT="$PUSHT_RUN_DIR/audits/data_check/data_verification.json"
# Speed mode (2026-11-01, user direction "把能跳过的全跳过"): reuse the hashes
# already audited in Gate1/Gate2 instead of re-reading 58 GB at training entry
# and 44 GB again at evaluation. Both code paths record the observed size/mtime
# and the audited digest; they never claim a hash they did not compute. Default
# (env unset) remains a full rehash, so Gate3 is unaffected.
export PUSHT_DATA_REHASH=skip
export PUSHT_EXPECTED_H5_SHA256="$(cat "$PUSHT_RUN_DIR/gates/unpacked_h5.sha256")"

# Gate4 (resume qualification) deliberately omitted on 2026-11-01 at the
# user's explicit direction ("把能跳过的全跳过"). It produces no scientific
# result; it only qualifies interrupt/resume recovery. Consequence: if this
# job is preempted or times out, the run cannot be safely resumed. See
# $STABLEWM_HOME/ops/STAGE2_GATE4_SKIPPED.md for the recorded divergence.

PUSHT_STEP="gate5_one_seed_paper10"
MODEL_NAME="paper10_$PUSHT_RUN_ID"
python -m scripts.repro_pusht.train \
  --audit-dir "$PUSHT_RUN_DIR/audits/paper10" \
  --data-verification "$DATA_REPORT" --global-seed 3072 \
  --execute data=pusht trainer.max_epochs=10 trainer.devices=1 \
  subdir="$MODEL_NAME" output_model_name="$MODEL_NAME"
WEIGHTS="$STABLEWM_HOME/checkpoints/$MODEL_NAME/weights_epoch_10.pt"
test -f "$WEIGHTS"

PUSHT_STEP="gate5_eval_trained_epoch10"
python -m scripts.repro_pusht.evaluate \
  --audit-dir "$PUSHT_RUN_DIR/audits/paper10_eval" \
  --execute "policy=$WEIGHTS"
python - <<'PY'
import json,os,pathlib
root=pathlib.Path(os.environ["PUSHT_RUN_DIR"])/"audits/paper10_eval"
s=json.loads((root/"selection.json").read_text())
r=json.loads((root/"completed.json").read_text())
assert s["episode_count"]==50 and len(s["rows"])==50 and r["phase"]=="world_evaluate_returned"
print("OUR PAPER10 METRICS:",json.dumps(r["metrics"],default=str))
PY
gate_pass gate5
echo "Paper-10 candidate completed. No extra seed, decoder or 100-epoch continuation is scheduled."
