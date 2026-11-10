# Codex task: TwoRoom LeWorldModel reconstruction transfer (CODE ONLY)

## Goal and scope
Implement and locally validate the **TwoRoom** control benchmark for two paired model arms:
- **A** = official LeWM: `L_pred + lambda_SIGReg * L_SIGReg`.
- **C** = A + self-reconstruction on the **current** observation: `L_A + alpha * L_rec`.
Use the official LeWM baseline config **unchanged** except for the extra auxiliary loss in C. Do NOT launch GPU training, CHPC/Slurm jobs, heavy evaluations, or change any Push-T experiment branch/results. Submit your changes in this draft PR for review; no merge.

## Existing code and fidelity
Repo: `LjPeng1/le-wm`. Original entries on `main`: `train.py`, `jepa.py`, `config/train/lewm.yaml`, `config/train/data/tworoom.yaml`, `config/train/model/lewm.yaml`, `config/eval/tworoom.yaml`, `config/eval/solver/cem.yaml`.

The production Push-T r010 self-reconstruction code lives in a separate CHPC workspace (`experiments/pusht_decoder_production`) and is not in the currently published GitHub branches. Search for an accessible original implementation first; if accessible, port the same decoder construction, reconstruction target, pixel normalization, and calibration exactly. If inaccessible, create an isolated, faithful implementation of the *specified* scheme and state explicitly that exact production-code parity remains unverified. Do not invent parity.

## Official TwoRoom baseline
- Use `data=tworoom`, HDF5 `tworoom.h5`, official dataset preprocessing and action normalization, 5-frame skip, input image 224, 3 history frames + 1 future target, ViT-tiny + latent dimension 192.
- Match official model/optimizer/trainer/scheduler/seed settings. `config/train/lewm.yaml` currently sets `loss.sigreg.weight=0.09`, AdamW lr 5e-5, weight decay 1e-3, max_epochs=100: confirm *whether TwoRoom has a paper-specific override* and use the correct published protocol. Do NOT copy `alpha` as a fixed numeric Push-T weight.
- Build A and C from identical initial world-model weights and same train split, minibatch order, schedule and training duration. Avoid RNG perturbation when constructing a decoder, e.g. initialize both arms deterministically and compare world-model tensor SHA. Model A must be the original loss mathematically; optionally instantiate but do not optimize a dummy decoder if required for paired initialization.
- Decoder receives 192-D *encoder/projector current latent* and reconstructs the corresponding current pixels only. Pixel-level reconstruction with gradient flowing to Encoder. Decoder MUST NOT participate in downstream planning; keep original latent MSE / CEM objective, no RGB cost.

## r010 reconstruction calibration
Transfer the Push-T **gradient target ratio=0.1** rule. On three fixed deterministic minibatches compute
`g_base = grad_encoder(L_pred + lambda*L_SIGReg)` and `g_rec = grad_encoder(L_rec)`;
freeze `alpha = 0.1 * median(||g_base|| / ||g_rec||)` for the full training.
If canonical Push-T code is found, use its exact statistical order, norm aggregation and calibration reset semantics. Save calibration samples, norm pairs, chosen alpha, encoder param set, initialization fingerprint, RNG state and strict failure behavior for NaN or near-zero gradients. Alpha calibration must not train the model, alter optimizer state, or change subsequent RNG/data order.
Each task keeps its own official `lambda_SIGReg`; for TwoRoom verify and retain the baseline value.

## Large paired evaluation, *without running now*
- Use official `swm/TwoRoom-v1`, official environment reset/goal-setting, action-block=5, horizon=5, receding=5, CEM default (300 candidates, 30 iterations, topk=30), goal_offset_steps=25, eval_budget=50, seed identity and success criterion. Resolve `max_episode_steps: ???` explicitly in an override or CLI, without changing the task definition.
- Default official evaluation only covers 50 trials, too little for this research. Create a deterministic **chunked and resumable** eval plan: e.g. 500 development trials and target >=2000 **independent** final trials, *if supported*; audit true unique episodes/start-goal pairs, duplicates and any replay overlap. If 2000 independent trials are not supported, accurately report max feasible and effective sample count; never count re-evaluations of identical starts as independent trials.
- Use paired A/C episodes and random planning seeds; split development vs final before model tuning, do not access final to tune. Implement manifest SHA, per-trial success, paired deltas and confidence intervals (cluster bootstrap by source episode if episodes repeat). Do not mix planning-seed repetitions with training-seed replicates.
- Code should support pilot with one paired **training seed** and later two additional independent train seeds, but **do not launch any training** in this assignment.
- Holdout independence is a hypothesis until verified; check whether simulator states and training corpus overlap, and label benchmark claims accordingly.

## Expected deliverables
1. Clean experiment entry points and configs (prefer `experiments/tworoom_reconstruction/`); unit/synthetic smoke tests for paired losses, gradients, alpha calibration, baseline unchanged, determinism, masks/normalization, dataset environment binding, resume, no test leakage, and simulator protocol if installed.
2. Document precise commands for preflight, train A/C, dev evaluation, final locked evaluation, repeated training seeds, reports. Show all provenance/outputs (model SHA, config digest, dataset/split, sample order, package versions, effective episode counts).
3. Run feasible **lightweight CPU checks** in Codex environment; distinguish these from actual simulator/GPU validation, and report any blocked dependency/data.
4. Do not remove or modify Push-T results, run heavy jobs, expose secrets, or merge. Final PR summary should include tested files, test results, and deviations from the production Push-T r010 implementation.
