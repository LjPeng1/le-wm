# Push-T reproduction on CHPC: two-visit workflow (PREPARATION ONLY)

These scripts run **only the existing audited LeWorldModel baseline** in the fork. They do not alter the author model, loss, CEM solver, data split or configuration files. Creating this commit **does not start any GPU job**.

Reference: upstream `lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
Review branch: `experiment/pusht-baseline-reproduction`. Use the **reviewed commit SHA** printed by `git rev-parse HEAD`, not a moving branch. See `docs/PUSHT_SERVER_STEPS.md` for the protocol.

## CHPC execution model (why stage 1 is split this way)

- **CPU-only `sbatch` jobs never schedule** on this account: they land in `QOS=init` (`MaxJobsPU=0`) and stay pending forever. Do not submit `01_prepare_cpu.sbatch` as a Slurm job.
- **Compute nodes have no external network**, so `uv pip install`, the pinned git dependency and the 13.14 GB Hugging Face download cannot run inside a Slurm job.
- Therefore **stage-1 preparation runs interactively on `chpc-sandbox`** (inside `tmux`; that node has network and is not subject to the login-node process killer), and **only the GPU stage is submitted via `sbatch`**. `chpc-sandbox` is where this repository and the FYP line already run heavy I/O.
- All download/cache directories are forced **under `$STABLEWM_HOME/cache`** by `jobs/lib.sh` (`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `WANDB_DIR`). This keeps the shared ~20 GB NFS home quota out of the picture.
- `STABLEWM_HOME` should sit in the FYP AUX area, e.g. `/auxstore/cse/d11/data/s1155211002/FYP/lewm`. **The 1 TB campus quota is shared by every project, not per project** — check the current headroom before the 13.14 GB download and the conversions. It must hold the compressed archive, the unpacked HDF5, the Lance copy, the venv, full checkpoints and videos.

## First login: prepare on sandbox, submit one GPU evaluation

1. On the Slurm **login node**, run the read-only inventory: `bash jobs/00_inventory.sh`. It does not allocate a GPU or submit anything. Confirm partitions (`cse_24h`), your association, storage, and GitHub connectivity (`command -v uv`, `command -v sbatch`, `sinfo -G`).
2. Clone the public fork and checkout the **fixed reviewed commit**. Do not run `main`. The repo must be on persistent storage visible to compute nodes (`~/FYP_Project/le-wm` is fine).
3. Set `STABLEWM_HOME` to approved **persistent absolute project storage** (`mkdir -p` it first; `--plan` also checks it exists). Set `PUSHT_GPU_PARTITION=cse_24h`. Optional: `PUSHT_ACCOUNT`, `PUSHT_GPU1_TIME` (default `12:00:00`), `PUSHT_GPU_MEM` (default `64G`), `PUSHT_GPU_GRES` (default `gpu:1`). Values are proposals, not confirmed cluster capabilities.
4. Preview both steps: `bash jobs/submit_stage1.sh --plan`. It never installs, downloads or calls `sbatch`.
5. Once storage and the large download are explicitly approved, run **on `chpc-sandbox`, inside `tmux`**:
   ```bash
   export STABLEWM_HOME=/auxstore/cse/d11/data/s1155211002/FYP/lewm
   export PUSHT_GPU_PARTITION=cse_24h
   export PUSHT_CONFIRM_STORAGE=YES PUSHT_CONFIRM_DATA_DOWNLOAD=YES
   bash jobs/submit_stage1.sh --prep
   ```
   This installs the pinned Python 3.10 environment, runs the local tests and `preflight`, downloads and hash-checks the pinned HF assets, decompresses the H5, converts it to Lance with the original converter, then runs `verify_data` and `verify_model`. It writes `status/prepare_cpu.<token>.json` and `gates/gate1.json` / `gates/gate2.json` under `$STABLEWM_HOME/workflows/<RUN_ID>/`.
6. Review Gate 1/Gate 2 output. Then, after separately approving **one** GPU evaluation:
   ```bash
   export PUSHT_CONFIRM_GPU_EVAL=YES
   bash jobs/submit_stage1.sh --submit
   ```
   This submits exactly one GPU job (`02_official_eval_gpu.sbatch`, `--gres=gpu:1`) with no `afterok` dependency; the job itself re-checks both gate receipts and refuses to run if they are missing or bound to another commit. It runs a two-trial smoke and then the unmodified **50-start CEM evaluation** of the released checkpoint. Record the printed `PUSHT_RUN_ID` and job ID.

If preparation is interrupted after `gate1` is written, rerun with a **new** `PUSHT_RUN_ID` (the downloaded assets under `STABLEWM_HOME` are reused; the audits are regenerated). Do not overwrite an existing gate receipt.

## Second login (only after reviewing stage1)

Inspect the stage status, official `completed.json`, selection and GPU logs. Decide whether the released-checkpoint evaluation is scientifically acceptable; **Gate3 passing is a completion check, not proof that the result matches the paper 96%**. If serious mismatch exists, stop and investigate rather than approving a training job.

Set the same `STABLEWM_HOME` and stage-1 `PUSHT_RUN_ID`; checkout the same fixed commit. Preview `bash jobs/submit_stage2.sh --plan`. Only after a *new* explicit GPU-training approval set `PUSHT_CONFIRM_GPU_TRAIN=YES` and `PUSHT_CONFIRM_10EPOCH=YES`, then submit.

```bash
export STABLEWM_HOME=/auxstore/cse/d11/data/s1155211002/FYP/lewm
export PUSHT_RUN_ID=PASTE_STAGE1_RUN_ID
export PUSHT_GPU_PARTITION=cse_24h
bash jobs/submit_stage2.sh --plan
# ONLY after reviewing stage1 and explicitly approving the actual GPU allocation:
export PUSHT_CONFIRM_GPU_TRAIN=YES
export PUSHT_CONFIRM_10EPOCH=YES
bash jobs/submit_stage2.sh --submit
```

Stage2 allocates **one GPU job** and runs real-model 2-epoch-limited A/B1/B2 continuous-versus-resumed verification with six workers and bf16. The `compare_resume` gate must pass **before** paper10. Then exactly one seed 3072, `max_epochs=10` from update zero with a 10-epoch scheduler horizon, followed by 50-start CEM evaluation of exported epoch10 weights. **No 100-epoch continuation, additional seeds or Decoder**. The 10-epoch horizon is a paper-aligned candidate, not proof of the authors' historical LR schedule.

If the cluster's max GPU wall time cannot cover resume qualification + training + evaluation, **do not submit this combined Stage2 job**. Split into dependent GPU jobs in a separate reviewed change while preserving one VPN login. The default 12 h is a configurable placeholder, not a measured runtime; `cse_24h` allows at most 24 h.

## Results, failures and notifications

All jobs log to `$STABLEWM_HOME/workflows/<RUN_ID>/logs/`; each stage leaves `status/<stage>.<SlurmJobId>.json`; individual completed gates leave `gates/gate{1..5}.json`. The raw audit subdirectories contain the official 50 selection IDs, metrics, complete training states, model hashes and traces. This format is intended for machine reading; **no process silently converts a failed check into a pass**.

Monitor with `squeue -j <ID>`, after completion `sacct -j <ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES`, and read `logs/*.out`. A failed GPU job leaves its own receipt and gate files untouched; fix the cause and resubmit only with explicit approval.

Only small JSON/text logs should be copied or synchronized to Google Drive **after verifying campus network policy and connector permissions**. No Google Drive credentials, GitHub tokens or SSH private keys belong in code, Slurm scripts, or logs. Large datasets, model weights and videos stay in project storage. Scripts do not auto-upload to Drive or start work at a later time outside Slurm.

**Important:** `PUSHT_CONFIRM_*=YES` must be set by the user at execution time; no automatic budget grant is implied by committing scripts. Gate1 hashes the complete 13 GB archive and the unpacked data; with slower networked project storage its duration can exceed an ordinary 30-minute CPU check. Stage2's resume verification rehashes full data for each training invocation; allow enough walltime and I/O budget. After smoke, use actual observed batch/episode runtimes instead of assuming the default 12 h will suffice.

If `uv` or `hf` is not available, install or load it in the cluster's permitted environment **before submission** or stop and collect the failure receipt; do not blindly upgrade pinned scientific dependencies. `source jobs/lib.sh` is only for jobs and is not a command the server assistant should run alone.
