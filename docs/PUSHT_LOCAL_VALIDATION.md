# Local validation and acceptance boundary

Date: 2026-11-01. This record covers code preparation on a CPU development machine. No official Push-T control evaluation, server GPU allocation or training reproduction result is reported.

## Latest review-fix validation

The final combined suite passed **56 tests**, skipped **1**, and reported **29 warnings** in **31.45 seconds**, exit 0. Python 3.12.14/torch 2.8 CPU and isolated pinned dependency cores are the local environment; the target remains the separately resolved Linux/Python3.10 candidate. All 16 helper/test Python files also parsed, and the original scientific files/configuration still have an empty upstream diff.

Coverage includes exclusive audit/run/export claims, policy export isolation without resampling, old resume SHA preservation, new-directory-only callbacks, fixed final `last.ckpt`, data-manifest binding/mutation rejection, actual parent-URI/table-path resolution, immutable schedule/dependency/split identities, stepwise LR/loss/input-byte traces, full optimizer/scheduler/loop/RNG state comparison and missing-input-identity rejection. The tiny zero-worker continuous and restored runs agree exactly in both live and **on-disk** final states; deliberate LR corruption is detected. Tests also exercise actual SWM/JEPA/native-loader methods on tiny fixtures, strict state/config rejection, first/repeated caches, and full-row numeric/JPEG/window audit logic on simulated readers.

The skipped test is the real tiny HDF5-to-Lance official-converter/reader integration: its optional local dependencies are unavailable. No official weights or dataset were downloaded or executed. Data-reader fixtures do not prove the real backend API/codec/data passes; the model fixtures do not prove official full-size forward equivalence. Those are explicit Gate1/2 runtime checks.

Warnings include legacy plotting parser deprecations, small fixture worker/logging settings and checkpoint-path metadata restoration. The run-directory metadata warning is expected after staging into a new directory; regression tests verify the old file remains unchanged and the new `last.ckpt` contains the terminal state. During development the new disk comparator caught Lightning's default `last-v1.ckpt` behavior; the owned new callback now disables version counters so it updates the new directory's fixed `last.ckpt` only. An earlier test assertion also used the wrong Lightning callback namespace and was corrected before final acceptance.

Run the local fixture suite with the optional source-isolation dependencies on PYTHONPATH:

```bash
PUSHT_SOURCE_ISOLATION=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -p no:cacheprovider --basetemp /path/to/new/workspace/test-output
```

Use an unused test-output directory. Source isolation is test-only and is never enabled by the production CLI. Actual Linux integration should run without it, after installing the candidate environment. Further coverage and server commands are in [review fixes](PUSHT_REVIEW_FIXES.md) and [server handoff](PUSHT_SERVER_STEPS.md).

## Verified source inspection

The pinned author source is `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; unchanged reference files are train.py, eval.py, jepa.py, module.py, utils.py and all config YAML. New scripts observe the original evaluation population and provide opt-in full-state checkpoint support. Strict weight loading is required before accepting a converted official model.

Dependency metadata and source contracts were inspected, and a Linux/Python 3.10 candidate lock was resolved without installing a CUDA training environment. That proves dependency resolution, not successful simulator execution. See the dependency audit for exact upstream revisions, release identities and unresolved compatibility.

## Unresolved runtime gates

- Gates1–3 remain pending: actual Linux install, release bytes, data/backend verification, official model function checks, simulator smoke and original 50-trial CEM outcomes.
- Gate4 remains pending: real multi-worker/bf16/GPU uninterrupted equivalence. In particular, persistent-worker iterator recreation can consume generator state differently at resume; only the real qualification with input-byte traces can accept that scope. Do not treat a zero-worker exact fixture as production equivalence.
- Cluster account/partition, actual GPU availability/VRAM, persistent paths, quotas and preemption policy remain unverified. GPU wall time and peak memory are unmeasured.
- The candidate environment is not a recovered author environment. JPEG conversion and the paper/code epoch difference remain explicit comparability limits.

## Earlier preparation check (historical)

Root independently executed the final suite: **12 passed, 15 warnings in 29.33 seconds**, exit 0. Interpreter: Python 3.12.14, torch 2.8.0+cpu, Lightning 2.6.1, stable-pretraining 0.1.7 core, pytest 8.4.2. This is distinct from the resolved Linux/Python 3.10 production candidate.

The source-isolated fixture uses the real Lightning loops, SPT Module/Manager/optimizer/scheduler, JEPA/ARPredictor and original loss function. It uses a tiny linear encoder, four-dimensional embeddings, eight SIGReg projections, zero-worker fixture DataModule, and 32-bit CPU arithmetic. Optional package initializers, external logging/monitoring callbacks and a POSIX-signal diagnostic unavailable on Windows are excluded in this test mode. It does not certify simulator, full DataModule, production dimensions or GPU execution.

The three-epoch continuous run and one-epoch-stop-plus-resume run match exactly in per-step loss/LR/sample order, final model tensors, optimizer/scheduler state, global step and estimated step horizon (12). Full checkpoint fields are inspected; weights-only input and changed 10-to-100 horizons are rejected. Output routing uses the production wrapper, with SPT cache routing temporarily disabled and restored. The initial test iterations exposed missing optional dataset imports, Windows signal diagnostics, forward-wrapper introspection and checkpoint cache routing; these were corrected or explicitly scoped before the passing final run.

Other tests cover the unchanged source loss and target gradients, pinned YAML, original 50-row sampling including its off-by-one exclusion, pre-rollout audit persistence, published HF configuration target mapping, strict-load rejection and no-overwrite behavior. Three additional root CLI checks confirm training/evaluation/conversion refuse to execute without `--execute`. A read-only preflight generated source hashes and correctly reported missing local assets.

For the isolated CPU test environment use `PUSHT_SOURCE_ISOLATION=1 python -m pytest tests -q`. The environment variable changes only fixture import/diagnostic handling, not production entry points. A clean Linux environment must also run the tests and the actual data/model smoke checks. The warnings concern tiny fixture worker/logging configuration and checkpoint-directory metadata; they do not represent measured Push-T performance.
