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
gate_require gate3
activate_env
check_gpu
DATA_REPORT="$PUSHT_RUN_DIR/audits/data_check/data_verification.json"

PUSHT_STEP="gate4_continuous_two_epoch_smoke"
python -m scripts.repro_pusht.train \
  --audit-dir "$PUSHT_RUN_DIR/audits/resume_A" \
  --data-verification "$DATA_REPORT" --trace-inputs --global-seed 3072 \
  --execute data=pusht trainer.max_epochs=2 trainer.devices=1 \
  +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 \
  subdir="resume_A_$PUSHT_RUN_ID" output_model_name="resume_A_$PUSHT_RUN_ID"

PUSHT_STEP="gate4_interrupted_first_epoch"
python -m scripts.repro_pusht.train \
  --audit-dir "$PUSHT_RUN_DIR/audits/resume_B1" \
  --data-verification "$DATA_REPORT" --trace-inputs --global-seed 3072 \
  --stop-after-epoch 1 --execute data=pusht trainer.max_epochs=2 trainer.devices=1 \
  +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 \
  subdir="resume_B1_$PUSHT_RUN_ID" output_model_name="resume_B1_$PUSHT_RUN_ID"

PUSHT_STEP="gate4_restore_second_epoch"
python -m scripts.repro_pusht.train \
  --audit-dir "$PUSHT_RUN_DIR/audits/resume_B2" \
  --data-verification "$DATA_REPORT" --trace-inputs --global-seed 3072 \
  --resume "$PUSHT_RUN_DIR/audits/resume_B1/checkpoints/last.ckpt" \
  --execute data=pusht trainer.max_epochs=2 trainer.devices=1 \
  +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 \
  subdir="resume_B2_$PUSHT_RUN_ID" output_model_name="resume_B2_$PUSHT_RUN_ID"

PUSHT_STEP="gate4_compare_real_resume"
python -m scripts.repro_pusht.compare_resume \
  --continuous "$PUSHT_RUN_DIR/audits/resume_A" \
  --interrupted "$PUSHT_RUN_DIR/audits/resume_B1" \
  --resumed "$PUSHT_RUN_DIR/audits/resume_B2" \
  --output "$PUSHT_RUN_DIR/audits/recovery_comparison.json" --execute
python - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ["PUSHT_RUN_DIR"])/"audits/recovery_comparison.json"
d=json.loads(p.read_text())
assert d["status"]=="passed" and d["input_bytes_observed"]
PY
gate_pass gate4

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
