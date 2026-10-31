# Server handoff: preparation before allocation

Current authorization ends at code preparation and PR. Run the read-only section first; evaluation and training require the later allocation decision. No cluster-specific submit command is provided because account, partition, wall-time limits and persistent paths have not been verified.

## Read-only server inventory

```bash
uname -a
python3 --version
command -v nvidia-smi && nvidia-smi
command -v sbatch || command -v qsub || true
df -h
```

Record available GPU memory, scheduler/account/partition, preemption behavior, quota and approved storage locations. Do not run model work on a login node. Set `STABLEWM_HOME` to an existing approved persistent project/cache directory, and retain its value with the run manifest.

## Checkout and environment

```bash
git clone https://github.com/LjPeng1/le-wm.git
cd le-wm
git checkout experiment/pusht-baseline-reproduction
git rev-parse HEAD
git diff 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac -- train.py eval.py jepa.py module.py utils.py config
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/repro-pusht.txt
uv pip install -r requirements/repro-tests.txt
python -m pip freeze > "$STABLEWM_HOME/pusht-environment.freeze.txt"
python -m scripts.repro_pusht.preflight --cache-root "$STABLEWM_HOME"
python -m pytest tests -q
```

The scientific-file diff must be empty. Requirements pin the audited candidate sources; the complete resolved environment must be frozen and accepted on Linux before use. An installation failure is a supported blocker, not permission to upgrade dependencies silently. Use `uv pip freeze` instead of `python -m pip freeze` if the isolated environment has no pip module.

## Model and dataset identity

The official model is `quentinll/lewm-pusht` at revision `22b330c28c27ead4bfd1888615af1340e3fe9052`; download only that revision and verify its raw identities against the dependency audit manifest. The matching HDF5 data release is `quentinll/lewm-pusht` (dataset) at `655cd446b9929369d7d406001da85c15d1457850`. Its compressed `pusht_expert_train.h5.zst` is 13,136,247,974 bytes, SHA256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`. Allow additional space for decompressed HDF5, Lance, checkpoints and videos; compression ratio/storage and throughput have not been measured here.

```bash
hf download quentinll/lewm-pusht --revision 22b330c28c27ead4bfd1888615af1340e3fe9052 --local-dir "$STABLEWM_HOME/hf_pusht"
hf download quentinll/lewm-pusht --repo-type dataset --revision 655cd446b9929369d7d406001da85c15d1457850 --local-dir "$STABLEWM_HOME/pusht_release"
sha256sum "$STABLEWM_HOME/pusht_release/pusht_expert_train.h5.zst"
```

Use the audited release's decompression/conversion instructions, preserving the archive. Hash the decompressed HDF5 and any Lance conversion. For conversion, record writer revision, JPEG settings, schema, row/episode/start-step correspondence, action/state values and measured pixel differences. The official writer's JPEG conversion is lossy; training equivalence is unresolved until those differences are accepted.

## First allocation request

Return the inventory, resolved environment, strict model-load result, data/backend checks and proposed runtime/storage budget before requesting allocation. Propose a bounded environment/model/data smoke test and official-checkpoint evaluation first. Save the exact 50 selected (row, episode, start-step) tuples before rollout, then preserve original raw metrics and videos. A failed smoke test stops the phase; do not launch a sweep to work around it.

After the official checkpoint has passed Gate A and the framework-specific resume test has passed Gate B, request one baseline training run with one immutable schedule identity. The 100-epoch schedule's additional 90 epochs and later independent seeds are separate decisions. The following commands are prepared for review, not authorized for execution by this document.

## Prepared evaluation and training commands (after approval)

Strict-convert the pinned local HF model to the native weights/config layout expected by the audited loader. Preserve the original downloaded directory and audit both versions:

```bash
python -m scripts.repro_pusht.convert_hf --source "$STABLEWM_HOME/hf_pusht" --output "$STABLEWM_HOME/checkpoints/official_pusht" --execute
python -m scripts.repro_pusht.evaluate --audit-dir "$STABLEWM_HOME/audits/official_pusht_seed42" --execute policy="$STABLEWM_HOME/checkpoints/official_pusht/weights.pt"
```

After Gate A/B and an explicit training decision, choose one schedule, record its immutable identifier, and use a fresh audit directory:

```bash
# Paper-10 candidate. Global seeding is an explicit documented reproducibility patch.
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/paper10_seed3072" --global-seed 3072 --execute data=pusht trainer.max_epochs=10 subdir=paper10_seed3072 output_model_name=paper10_seed3072

# Separate run: 100-epoch schedule from update zero, stop at the epoch-10 boundary.
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/schedule100_seed3072_first10" --global-seed 3072 --stop-after-epoch 10 --execute data=pusht trainer.max_epochs=100 subdir=schedule100_seed3072 output_model_name=schedule100_seed3072

# Separate approval required. Restore the full state from the preceding run.
python -m scripts.repro_pusht.train --audit-dir "$STABLEWM_HOME/audits/schedule100_seed3072_continue" --resume "$STABLEWM_HOME/audits/schedule100_seed3072_first10/checkpoints/last.ckpt" --execute data=pusht trainer.max_epochs=100 subdir=schedule100_seed3072 output_model_name=schedule100_seed3072
```

Do not run both schedule alternatives automatically. The resume wrapper rejects a changed scheduler horizon and weights-only checkpoints. Evaluate any trained checkpoint by naming the exact `weights_epoch_N.pt` file and a new audit directory; retain the original fixed 50-point population for final reporting. The full worker/GPU resume qualification remains separate from CPU callback tests.
