# Push-T baseline reproduction protocol

Status: code preparation only, 2026-11-01. No official-checkpoint evaluation or server training has been executed. The paper's 96.0 ± 2.83% is an external reference, not a result obtained here.

## Scope and source identity

The reference is [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac), commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The fork retains the original training, evaluation, architecture, objective and YAML files. Added scripts provide explicit execution gates, provenance and CPU checks. No reconstruction/Decoder objective, state-targeted training or altered planner is included.

## Paper and code discrepancies

| Topic | Evidence | Handling |
| --- | --- | --- |
| Epochs | [Paper v3 Appendix E](https://arxiv.org/html/2603.19312v3#A5) says Push-T uses 10 epochs; `config/train/lewm.yaml` sets 100 | First round: only paper10_candidate, max_epochs10 from initialization. A 100-epoch horizon is future exploration, not an extra first-round run. |
| Data backend | `config/train/data/pusht.yaml` names `pusht_expert_train.lance`; README documents HDF5 and `eval.py:get_dataset` directly instantiates HDF5Dataset | Require matching trajectory/content evidence and separate backend identities. Do not rename or convert a dataset and assume equivalence. |
| Global seed | Upstream seeds split/DataLoader generator with 3072, but does not call a global seed initializer | The original run is not globally deterministic merely because cfg.seed exists. The audited wrapper records its explicit initialization policy. |
| Evaluation population | `eval.py` samples sorted valid row indices with `choice(len(valid_indices)-1, 50, replace=False)` | Preserve the excluded final valid row and the exact algorithm; record row/episode/start-step identities before evaluation. Episode IDs may repeat. |
| Checkpoints | Callback exports model weights; Lightning training state and portable model artifacts serve different purposes | Keep full resume state separate from the model used by evaluation. Never resume from weights alone. |
| Output routing | Manager cache mode redirects checkpoint paths; legacy resume can add a callback saving to its input directory | The wrapper explicitly uses legacy mode, claims all output paths, and stages a hash-verified resume copy into its new callback directory. Cache auto-requeue discovery is disabled; resume is explicit. Evaluation policy copies isolate video/text writes. |
| Scheduler interval | Audited stable-pretraining v0.1.7 manually steps the scheduler per optimizer step despite upstream's `interval: epoch` declaration | Its horizon uses `trainer.estimated_stepping_batches` and default warmup is 1% of total steps; see dependency audit. A 10-epoch versus 100-epoch run therefore differs from the first update. |
| Data conversion | The audited official Lance writer defaults to JPEG quality 95 | Record conversion settings and image differences; matching episode/row identities does not establish pixel equality. This remains a paper-comparability limitation. |
| Evaluation seed | Seed 42 enters start-point selection and CEM config | Record framework, environment and device RNG separately; do not infer full determinism from this single field. |

The paper sections read for this task are Appendices D–G and associated training/evaluation passages. This is targeted full-text reading, not a complete-paper review. Its reported variation is over three trained models on the same 50 evaluation trajectories; reproduce and report the aggregation convention explicitly.

## Frozen reference settings

Training reference: AdamW lr `5e-5`, weight decay `1e-3`, batch 128, image 224, history 3, latent 192, frameskip 5, prediction length 1, bf16, gradient clip 1.0, SIGReg weight .09 / knots 17 / projections 1024. Full gradients reach both encoder branches. State/proprioception may be present for normalization/evaluation but must not become physical-state training targets.

Evaluation reference: `swm/PushT-v1`, 50 trials, goal offset 25, action budget 50, seed 42; CEM 300 candidates, 30 iterations, top 30, variance 1.0, horizon/receding horizon/action block all 5. Original image preprocessing and dataset-derived normalization remain part of the reference.

## Acceptance gates and schedule policy

**Gate 0:** unchanged author source/configuration and reviewed wrapper/output protection. **Gate 1:** actual Linux candidate install and release HDF5/Lance verification. The new `verify_data` fully checks content manifests, row/index/numeric mapping and reports bounded JPEG/window sampling. It does not claim exhaustive image decoding or source-pixel equivalence.

**Gate 2:** actual official strict-load/native reload and function consistency using `verify_model`, real HDF5 inputs and original preprocessing/normalizers; CPU/float32 default. Compare encode/action/predict/all rollout frames/cost/first and repeated cache calls. Failure stops evaluation; fixed engineering tolerance is not an equivalence proof for all inputs.

**Gate 3:** separately approved released-checkpoint original 50-start CEM evaluation, saving selected rows/episodes/start steps before rollout, raw per-trial results, hardware/wall time/peak memory and persistent artifacts. A command or fixture pass does not pass this gate. **Gate 4:** separately approved real-model/data epoch-boundary recovery qualification with fixed horizon and matched steps. Full checkpoint v2 includes original scientific sources, helper/installed-dependency/data/split identities, optimizer/scheduler/loops and RNG/loader state. Stepwise LR/loss traces and actual scheduler horizon are recorded; source resume bytes remain untouched. Production worker ordering is not certified by split hashes or zero-worker CPU fixtures. **Gate 5:** separately approved one-seed paper10_candidate, only after Gates3–4 pass. Changing max_epochs10 to100 is not equivalent recovery.

README's `_object.ckpt` conversion example is also older than the pinned `eval.py` call to `swm.wm.utils.load_pretrained`. Use the audited dependency loader contract for evaluation and retain strict state-dict validation; do not assume the README's object artifact is the eval entry point's current input.

**Server execution:** Gate1–4 evidence, actual cluster configuration, persistent paths and explicit allocation decisions are prerequisites for the one approved baseline seed. Additional seeds or another schedule are future decisions. No cluster submission template is supplied before account/partition/GPU/time-limit/storage checks.

## Evaluation selection

Choose the epoch/protocol before opening the 50 final outcomes. Validation uses the training split's held-out loss; it does not reuse final control episodes to tune epochs, weights or CEM. If a separate planning-validation set is introduced, predeclare its identities and disjointness. Record any subsequent test-informed decision as exploratory. Use one fixed final evaluation population across training seeds and report each seed before aggregate mean/variation.

## Server handoff and unresolved gates

Use `docs/PUSHT_DEPENDENCY_AUDIT.md` for pinned dependency candidates and compatibility limitations. Use `docs/PUSHT_LOCAL_VALIDATION.md` for actual local outcomes and `scripts/repro_pusht/` for gated commands. Current preparation does not assert runtime Gates1–5 passed. Large data, weights, videos and logs belong under a server-provided persistent path and are excluded from git.

Before requesting the first server run, collect: Linux/Python/CUDA/driver versions; GPU type/VRAM/count; scheduler/account/partition/time and preemption limits; persistent project/cache paths and quota; download policy; exact data/model revisions/checksums; installed environment lock; expected run wall time/memory measured by a bounded smoke test. Historical server descriptions are not evidence of current availability. No GPU hours are estimated as measured until that smoke test exists.
