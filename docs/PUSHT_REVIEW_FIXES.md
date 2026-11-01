# Push-T review fixes and readiness

Status: code preparation complete, 2026-11-01. The reviewed output/resume issues are corrected and local fixture tests pass. Runtime acceptance on the candidate Linux environment, official data/weights and GPU remains pending. This change does not run or authorize a server allocation.

## What changed

The original `train.py`, `eval.py`, `jepa.py`, `module.py`, `utils.py` and all author YAML remain unchanged relative to `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The following changes are confined to explicit preparation wrappers and checks:

| Review issue | Correction | Evidence |
| --- | --- | --- |
| Manager writes resume state back to its input directory | Exclusive, SHA-verified copy to the new callback directory; record original/staged input identities | Real Lightning/SPT-core regression checks old source SHA and every checkpoint callback directory |
| A copied `last.ckpt` remains stale when Lightning versions its successor | Disable version counter only on the wrapper-owned callback in the new directory | Disk-level final checkpoint comparison, not just live tensors |
| Fresh audit path does not protect other exports | Claim audit, original config directory and model export directory before original Module/config writes; explicit policy copy isolates original eval exports | Collision and original-content preservation tests; unchanged sampled rows |
| Data/helper/runtime/split identity is not enforced on recovery | Bind checkpoint v2 to full data manifest/report, actual split hashes, helper/scientific hashes, candidate lock and installed dependency identities; rehash data first | Tampered-data/report and runtime-change rejection tests |
| No executable backend verification | `verify_data`: full file/row/episode/numeric audit plus bounded actual-reader window and JPEG95 replay checks | Fixture coverage; optional real-converter test skips locally |
| Strict state keys alone do not prove model behavior | `verify_model`: strict SWM/local JEPA loads, native reload tensor identity, original real H5 preprocessing and history1/3 encode/action/predict/rollout/cost comparisons | Actual model methods on tiny source-isolated fixtures; official assets still pending |
| Recovery evidence lacks observable batches/loop state | Optional `--trace-inputs` for bounded qualification, before forward mutation; step LR/loss traces and loop/RNG comparison; fail qualification when fingerprints are absent | Exact tiny continuous-vs-resumed bytes/state; deliberate mismatch rejection |
| First-round docs still offer two training schedules | Single paper10 first-round path and Gates0–5; 100-epoch schedule is future exploration | [Protocol](PUSHT_REPRO_PROTOCOL.md), [server commands](PUSHT_SERVER_STEPS.md) |

Cache=None is an explicit Manager legacy-mode choice. It disables automatic cache-based requeue discovery, not only output routing. Recovery therefore requires an explicit checkpoint and new directories; the wrapper rejects old version-1 state and changed schedule/data/runtime identities. The unmodified author entry point retains its original behavior.

## What the checks mean

`verify_data` does not implement a new converter. It reads the existing official HDF5 and Lance backends, accounts for regenerated episode/step IDs and float32 tabular conversion, and distinguishes full numeric/index coverage from sampled image/window coverage. JPEG95 replay is checked with the pinned codec path; original-vs-JPEG image error is descriptive. Cross-codec discrepancies fail visibly rather than silently relaxing equality. The input archive's official release hash remains a separate download/decompression provenance check.

`verify_model` is CPU/float32/eval by default, even on GPU hosts. It uses pinned local release bytes, unchanged data-derived normalizers/image processing, independent mutable dictionaries and first/repeated cached calls. Fixed `atol=1e-5, rtol=1e-4` are initial engineering thresholds, not a general equivalence proof. The real input row is explicit; a missing/nonfinite/insufficient real window fails with a receipt rather than silently changing data. It produces function diagnostics, not CEM scores.

`compare_resume` loads only trusted local full checkpoints. It compares observed input fingerprints, traces, model/optimizer/scheduler/loop/RNG state and source checkpoint preservation. Qualification fingerprints transfer full tensor bytes to CPU and hash them, so they are enabled for bounded recovery tests rather than silently imposed on full training. No input identity means no qualification pass. Persistent-worker state and bf16 still require the real Gate4 run; matched tiny fixtures do not prove arbitrary future trajectories.

## Acceptance boundary

The final local suite reports **56 passed, 1 skipped**; see [local validation](PUSHT_LOCAL_VALIDATION.md) for environment, timing, limitations and the skipped real-backend integration. No official data/weight acceptance, 50-start CEM run or independent training result is claimed.

The next action is bounded Gate1/2 environment/data/model preflight after actual resource/storage inventory. Gate3 official evaluation and Gate4 real recovery require explicit GPU approval. Gate5 remains exactly one separately approved 10-epoch-from-update-zero candidate. No decoder, additional objective, altered planner, changed loss weight or new seed is introduced.
