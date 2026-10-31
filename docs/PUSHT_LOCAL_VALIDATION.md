# Local validation and acceptance boundary

Date: 2026-11-01. This record covers code preparation on a CPU development machine. No official Push-T control evaluation, server GPU allocation or training reproduction result is reported.

## Verified source inspection

The pinned author source is `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; unchanged reference files are train.py, eval.py, jepa.py, module.py, utils.py and all config YAML. New scripts observe the original evaluation population and provide opt-in full-state checkpoint support. Strict weight loading is required before accepting a converted official model.

Dependency metadata and source contracts were inspected, and a Linux/Python 3.10 candidate lock was resolved without installing a CUDA training environment. That proves dependency resolution, not successful simulator execution. See the dependency audit for exact upstream revisions, release identities and unresolved compatibility.

## Unresolved runtime gates

- Gate A remains pending: official weights strict-load, full data/backend identity, simulator smoke test, original CEM 50-trial run and saved raw outcomes.
- Gate B's paper schedule identity remains uncertain. A CPU checkpoint mechanism test cannot establish full multi-worker/bf16/GPU uninterrupted equivalence. The server must qualify that scope before a resumed run is described as equivalent.
- Cluster account/partition, actual GPU availability/VRAM, persistent paths, quotas and preemption policy remain unverified. GPU wall time and peak memory are unmeasured.
- The candidate environment is not a recovered author environment. JPEG conversion and the paper/code epoch difference remain explicit comparability limits.

## Executed local checks

Root independently executed the final suite: **12 passed, 15 warnings in 29.33 seconds**, exit 0. Interpreter: Python 3.12.14, torch 2.8.0+cpu, Lightning 2.6.1, stable-pretraining 0.1.7 core, pytest 8.4.2. This is distinct from the resolved Linux/Python 3.10 production candidate.

The source-isolated fixture uses the real Lightning loops, SPT Module/Manager/optimizer/scheduler, JEPA/ARPredictor and original loss function. It uses a tiny linear encoder, four-dimensional embeddings, eight SIGReg projections, zero-worker fixture DataModule, and 32-bit CPU arithmetic. Optional package initializers, external logging/monitoring callbacks and a POSIX-signal diagnostic unavailable on Windows are excluded in this test mode. It does not certify simulator, full DataModule, production dimensions or GPU execution.

The three-epoch continuous run and one-epoch-stop-plus-resume run match exactly in per-step loss/LR/sample order, final model tensors, optimizer/scheduler state, global step and estimated step horizon (12). Full checkpoint fields are inspected; weights-only input and changed 10-to-100 horizons are rejected. Output routing uses the production wrapper, with SPT cache routing temporarily disabled and restored. The initial test iterations exposed missing optional dataset imports, Windows signal diagnostics, forward-wrapper introspection and checkpoint cache routing; these were corrected or explicitly scoped before the passing final run.

Other tests cover the unchanged source loss and target gradients, pinned YAML, original 50-row sampling including its off-by-one exclusion, pre-rollout audit persistence, published HF configuration target mapping, strict-load rejection and no-overwrite behavior. Three additional root CLI checks confirm training/evaluation/conversion refuse to execute without `--execute`. A read-only preflight generated source hashes and correctly reported missing local assets.

For the isolated CPU test environment use `PUSHT_SOURCE_ISOLATION=1 python -m pytest tests -q`. The environment variable changes only fixture import/diagnostic handling, not production entry points. A clean Linux environment must also run the tests and the actual data/model smoke checks. The warnings concern tiny fixture worker/logging configuration and checkpoint-directory metadata; they do not represent measured Push-T performance.
