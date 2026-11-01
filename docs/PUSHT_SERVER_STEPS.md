# Push-T server handoff: gated preparation

These commands are proposed, not allocation authorization. No server GPU evaluation/training has been run. First evaluate the released checkpoint, then qualify real-model recovery, then request **one** `paper10_candidate`. No sweep, extra seed or 100-epoch continuation is authorized.

Contents: [source](#gate-0-source-and-artifact-protection), [environment/data](#gate-1-linux-environment-and-data-cpustorage), [model](#gate-2-actual-model-loading-and-function-checks-cpu-default), [evaluation](#gate-3-official-checkpoint-evaluation--proposed-not-authorized), [recovery](#gate-4-real-data-recovery-qualification--proposed-not-authorized), [single training](#gate-5-one-paper-10-candidate--separate-approval-required).

## Gate 0: source and artifact protection

Checkout the reviewed commit reported in the PR, not an unverified moving branch. Record `git rev-parse HEAD`. The scientific-file diff must be empty:

```bash
git diff 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac HEAD -- train.py eval.py jepa.py module.py utils.py config
```

The wrapper claims audit/run/export directories before original config/export writes, stages explicit resume inputs into the new checkpoint directory, and isolates evaluation exports with a hash-verified policy copy. Use new `subdir`, `output_model_name` and audit paths for every invocation, including resumed invocations. Version-1 preparation checkpoints are rejected under the new full-state identity contract. Only trusted local Lightning checkpoints may be supplied for full-state deserialization.

## Gate 1: Linux environment and data (CPU/storage)

```bash
uname -a
python3 --version
command -v nvidia-smi && nvidia-smi
command -v sbatch || command -v qsub || true
df -h
```

Confirm actual GPU memory/driver, CPU/RAM, account/scheduler/partition, storage quota, network and preemption policy. Set `STABLEWM_HOME` to approved persistent storage. Do not run compute on a login node. Historical server descriptions are not current inventory; no cluster-specific submission script is supplied before inventory.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/repro-pusht.txt
uv pip install -r requirements/repro-tests.txt
uv pip freeze > "$STABLEWM_HOME/pusht-environment.freeze.txt"
python -m scripts.repro_pusht.preflight --cache-root "$STABLEWM_HOME"
python -m pytest tests -q
```

Use a new freeze destination and retain package direct-url/commit identities. Stop on install failure rather than upgrading silently. This is a compatibility candidate, not a recovered historical author environment. Download pinned assets and verify release sizes/hashes against `docs/PUSHT_SOURCE_MANIFEST.json`:

```bash
hf download quentinll/lewm-pusht --revision 22b330c28c27ead4bfd1888615af1340e3fe9052 --local-dir "$STABLEWM_HOME/hf_pusht"
hf download quentinll/lewm-pusht --repo-type dataset --revision 655cd446b9929369d7d406001da85c15d1457850 --local-dir "$STABLEWM_HOME/pusht_release"
sha256sum "$STABLEWM_HOME/hf_pusht/weights.pt" "$STABLEWM_HOME/pusht_release/pusht_expert_train.h5.zst"
```

The 13,136,247,974-byte archive is a single zstandard-compressed HDF5, **not tar**. Check space for compressed H5, unpacked H5, Lance, dependencies, full checkpoints and videos; actual expansion and throughput are unknown. Preserve the archive and write the unpacked H5 exclusively:

```bash
python - <<'PY'
import os,pathlib,zstandard
root=pathlib.Path(os.environ['STABLEWM_HOME'])
(root/'datasets').mkdir(parents=True,exist_ok=True)
with (root/'pusht_release/pusht_expert_train.h5.zst').open('rb') as src:
    with (root/'datasets/pusht_expert_train.h5').open('xb') as dst:
        zstandard.ZstdDecompressor().copy_stream(src,dst)
PY
python - <<'PY'
import os,pathlib
from stable_worldmodel.data import convert
root=pathlib.Path(os.environ['STABLEWM_HOME'])/'datasets'
convert(str(root/'pusht_expert_train.h5'),str(root/'pusht_expert_train.lance'),
        source_format='hdf5',dest_format='lance',mode='error',jpeg_quality=95)
PY
python -m scripts.repro_pusht.verify_data --hdf5 "$STABLEWM_HOME/datasets/pusht_expert_train.h5" --lance "$STABLEWM_HOME/datasets/pusht_expert_train.lance" --output-dir "$STABLEWM_HOME/audits/data_check" --execute
```

An interrupted conversion is not a valid cache; diagnose without automatic overwrite. The verifier fully hashes artifacts and checks row/episode/numeric mappings; decoded image and reader-window checks are bounded deterministic samples. Preserve `data_verification.json` and `data_content_manifest.json` together. Sampled JPEG checks explicitly report coverage/lossy differences, not universal pixel identity or exhaustive image-decode health. Training rehashes content before accepting the receipt.

## Gate 2: actual model loading and function checks (CPU default)

```bash
python -m scripts.repro_pusht.convert_hf --source "$STABLEWM_HOME/hf_pusht" --output "$STABLEWM_HOME/checkpoints/official_pusht" --execute
python -m scripts.repro_pusht.verify_model --source "$STABLEWM_HOME/hf_pusht" --converted "$STABLEWM_HOME/checkpoints/official_pusht" --hdf5 "$STABLEWM_HOME/datasets/pusht_expert_train.h5" --expected-hdf5-sha256 "$VERIFIED_H5_SHA256" --output "$STABLEWM_HOME/audits/model_check" --execute
```

Set `VERIFIED_H5_SHA256` from the verified **unpacked** H5 manifest entry; it differs from the release archive hash. The model checker defaults to CPU/float32/eval, performs no downloads and checks official release bytes, strict source loads, native local reload tensor identity, original real-data preprocessing/normalizers, history1/3, encode/actions/predict/each rollout frame/final cost/first and repeated cache calls. Fixed initial tolerances are engineering criteria. A mismatch saves diagnostics and stops the gate; do not loosen tolerances to force a pass. Any GPU inference requires a later resource decision.

## Gate 3: official checkpoint evaluation — PROPOSED, NOT AUTHORIZED

After Gates1–2 pass and GPU approval, run a bounded simulator/CEM smoke, then the unchanged 50-start evaluation. A two-trial smoke is not the first two of the final 50 or a final score. Use new audit directories:

```bash
python -m scripts.repro_pusht.evaluate --audit-dir "$STABLEWM_HOME/audits/official_pusht_seed42" --execute policy="$STABLEWM_HOME/checkpoints/official_pusht/weights.pt"
```

The exact policy/config copy isolates original fixed-name videos and text logs inside this audit. Preserve selection.json before rollout, all 50 row/episode/start tuples, raw per-trial outcomes, stdout/stderr, videos, wall time and peak memory. Early entry failures also get receipts. Original sampling/off-by-one, normalizers and CEM remain unchanged. Repeated episode IDs may share an original video filename within one run; raw per-trial outcomes remain authoritative.

## Gate 4: real-data recovery qualification — PROPOSED, NOT AUTHORIZED

Use real model/data and original six workers/prefetch/bf16; one allocated visible GPU. Continuous A and interrupted B share a two-epoch horizon, bounded to four training/two validation batches per epoch. These qualify recovery, not reproduction performance:

```bash
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/resume_A" --data-verification "$STABLEWM_HOME/audits/data_check/data_verification.json" --trace-inputs --global-seed 3072 --execute data=pusht trainer.max_epochs=2 trainer.devices=1 +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 subdir=resume_A output_model_name=resume_A
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/resume_B1" --data-verification "$STABLEWM_HOME/audits/data_check/data_verification.json" --trace-inputs --global-seed 3072 --stop-after-epoch 1 --execute data=pusht trainer.max_epochs=2 trainer.devices=1 +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 subdir=resume_B1 output_model_name=resume_B1
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/resume_B2" --data-verification "$STABLEWM_HOME/audits/data_check/data_verification.json" --trace-inputs --global-seed 3072 --resume "$STABLEWM_HOME/audits/resume_B1/checkpoints/last.ckpt" --execute data=pusht trainer.max_epochs=2 trainer.devices=1 +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 subdir=resume_B2 output_model_name=resume_B2
python -m scripts.repro_pusht.compare_resume --continuous "$STABLEWM_HOME/audits/resume_A" --interrupted "$STABLEWM_HOME/audits/resume_B1" --resumed "$STABLEWM_HOME/audits/resume_B2" --output "$STABLEWM_HOME/audits/recovery_comparison.json" --execute
```

Compare step/LR/loss traces, model/optimizer/scheduler/loop/RNG state, split/window identities and callback paths; verify B1 source hashes unchanged. The bounded qualification explicitly enables full input-byte fingerprints **before** forward mutates NaN actions; this incurs CPU transfer/hash cost. The comparator refuses qualification if fingerprints are absent. Sample IDs may be unavailable in real batches; split hashes alone do not certify worker ordering. Do not enable expensive batch-byte tracing by default on the later full training. Persistent-worker generator behavior remains a real Gate4 question, not a local zero-worker guarantee. Cache=None disables Manager auto-requeue discovery; recovery is explicit, never automatic from-scratch restart.

## Gate 5: one Paper-10 candidate — separate approval required

After Gates3–4 pass, request actual hardware-based time/storage/budget. Use one visible GPU and immutable max_epochs10 from initialization:

```bash
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/paper10_seed3072" --data-verification "$STABLEWM_HOME/audits/data_check/data_verification.json" --global-seed 3072 --execute data=pusht trainer.max_epochs=10 subdir=paper10_seed3072 output_model_name=paper10_seed3072
```

Global3072 is the declared reproducibility patch. Original loss/architecture/optimizer/data settings/normalizers/planner remain unchanged. Evaluate exact epoch10 weights using a fresh audit. Report one-seed raw outcomes/cost before any aggregate; do not tune against the final 50 starts. Published statistics, released-checkpoint replay and our candidate are separate. A 100-epoch schedule remains future exploration with no first-round command here.
