# Slurm Push-T reproduction: two-visit workflow (PREPARATION ONLY)

These scripts schedule **only the existing audited LeWorldModel baseline** in the fork. They do not alter the author model, loss, CEM solver, data split or configuration files. Creating this commit **does not start any GPU job**.

Reference: upstream \`lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac\`.
Review branch: \`experiment/pusht-baseline-reproduction\`. Use the **new reviewed commit SHA** printed by \`git rev-parse HEAD\`, not a moving branch. See \`docs/PUSHT_SERVER_STEPS.md\` for exact protocol.

## First VPN login (submit 1 CPU + 1 dependent GPU job)

1. Connect to the university network. On the Slurm **login node**, verify GitHub and read-only cluster facts: \`bash jobs/00_inventory.sh\`. This does **not** allocate a GPU. Check \`sinfo\`, account, allowed partitions, limits, user/project storage quota, download access, and \`uv\` availability. A generic Linux login shell is insufficient evidence of correct cluster resources.
2. Clone the public fork and checkout the **fixed, newly reviewed commit**. Do not run \`main\`. Ensure the repository is visible to all Slurm compute nodes; do not clone only to volatile login-local storage.
3. Set \`STABLEWM_HOME\` to approved **persistent absolute project storage**. This storage contains ~13.14GB of compressed official dataset plus unpacked HDF5, Lance, environments, full checkpoints and videos. Determine actual capacity before submission.
4. Choose \`PUSHT_GPU_PARTITION\`. Optionally set \`PUSHT_CPU_PARTITION\`, \`PUSHT_ACCOUNT\`, \`PUSHT_CPU_TIME\` (default \`08:00:00\`), \`PUSHT_GPU1_TIME\` (default \`12:00:00\`), \`PUSHT_CPU_MEM\` and \`PUSHT_GPU_MEM\` (default 64G), \`PUSHT_GPU_GRES\` (default \`gpu:1\`). Values are proposals, not confirmed CUHK cluster capabilities. Adjust to actual Slurm limits.
5. Preview: \`bash jobs/submit_stage1.sh --plan\`. It never calls sbatch.
6. Once you separately approve storage, the large dataset download and **one official-checkpoint GPU evaluation**, export \`PUSHT_CONFIRM_STORAGE=YES\`, \`PUSHT_CONFIRM_DATA_DOWNLOAD=YES\`, \`PUSHT_CONFIRM_GPU_EVAL=YES\`, then run \`bash jobs/submit_stage1.sh --submit\`. Record the printed \`PUSHT_RUN_ID\` and two job IDs.

Example after supplying real cluster resource details:

\`\`\`bash
export STABLEWM_HOME=/approved/persistent/project/area/lewm
export PUSHT_GPU_PARTITION=YOUR_GPU_PARTITION
# Optional: export PUSHT_CPU_PARTITION=YOUR_CPU_PARTITION
# Optional: export PUSHT_ACCOUNT=YOUR_ALLOCATION_ACCOUNT
bash jobs/00_inventory.sh
bash jobs/submit_stage1.sh --plan

# ONLY after you explicitly approve the data/GPU operations:
export PUSHT_CONFIRM_STORAGE=YES
export PUSHT_CONFIRM_DATA_DOWNLOAD=YES
export PUSHT_CONFIRM_GPU_EVAL=YES
bash jobs/submit_stage1.sh --submit
\`\`\`

Stage1 CPU: install Python3.10 candidate env from the pinned lock, run local tests, download and hash pinned HF assets, decompress H5, convert H5 to Lance using the original converter, run \`verify_data\` and \`verify_model\`. Stage1 GPU: \`sbatch --dependency=afterok:<CPU_JOB_ID>\` executes **two-trial smoke** then unmodified **50-start CEM evaluation** of the released checkpoint; failure to pass CPU gates means no GPU evaluation. Submit IDs are saved under \`$STABLEWM_HOME/workflows/<RUN_ID>/status/\`.

## Second VPN login (only after reviewing stage1)

Inspect the first-stage status, official \`completed.json\`, selection and GPU logs. Decide whether the released checkpoint evaluation is scientifically acceptable; **Gate3 passing is a completion check, not proof that the result matches the paper 96%**. If serious mismatch exists, stop and investigate rather than approving a training job.

Set the same \`STABLEWM_HOME\` and stage1 \`PUSHT_RUN_ID\`; checkout the same fixed commit. Preview \`bash jobs/submit_stage2.sh --plan\`. Only after a *new* explicit GPU-training approval set \`PUSHT_CONFIRM_GPU_TRAIN=YES\` and \`PUSHT_CONFIRM_10EPOCH=YES\`, then submit.

\`\`\`bash
export STABLEWM_HOME=/approved/persistent/project/area/lewm
export PUSHT_RUN_ID=PASTE_STAGE1_RUN_ID
export PUSHT_GPU_PARTITION=YOUR_GPU_PARTITION
bash jobs/submit_stage2.sh --plan
# ONLY after reviewing stage1 and explicitly approving the actual GPU allocation:
export PUSHT_CONFIRM_GPU_TRAIN=YES
export PUSHT_CONFIRM_10EPOCH=YES
bash jobs/submit_stage2.sh --submit
\`\`\`

Stage2 allocates **one GPU job** and runs real-model 2-epoch-limited A/B1/B2 continuous-versus-resumed verification with six workers and bf16. The \`compare_resume\` gate must pass **before** paper10. Then exactly one seed 3072, \`max_epochs=10\` from update zero with a 10-epoch scheduler horizon, followed by 50-start CEM evaluation of exported epoch10 weights. **No 100-epoch continuation, additional seeds or Decoder**. The 10-epoch horizon is a paper-aligned candidate, not proof of the authors' historical LR schedule.

If the cluster's max GPU wall time cannot cover resume qualification + training + evaluation, **do not submit this combined Stage2 job**. Split into dependent GPU jobs in a separate reviewed change while preserving one VPN login. The default 12h is a configurable placeholder, not a measured runtime.

## Results, failures and notifications

All jobs log to \`$STABLEWM_HOME/workflows/<RUN_ID>/logs/\`; each stage leaves \`status/<stage>.<SlurmJobId>.json\`; individual completed gates leave \`gates/gate{1..5}.json\`. The raw audit subdirectories contain the official 50 selection IDs, metrics, complete training states, model hashes and traces. This format is intended for machine reading; **no process silently converts a failed check into a pass**.

Monitor with \`squeue -j <ID1>,<ID2>\`, after completion \`sacct -j <ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES\`, and read \`logs/*.out\`. An unsuccessful \`afterok\` dependency keeps the GPU job from running; it may remain pending with \`DependencyNeverSatisfied\` and can be cancelled with \`scancel <ID>\`.

Only small JSON/text logs should be copied or synchronized to Google Drive **after verifying campus network policy and connector permissions**. No Google Drive credentials, GitHub tokens or SSH private keys belong in code, Slurm scripts, or logs. Large datasets, model weights and videos stay in project storage. Scripts do not auto-upload to Drive or start work at a later time outside Slurm.

**Important:** \`PUSHT_CONFIRM_*=YES\` must be set by the user at execution time; no automatic budget grant is implied by committing scripts. Gate1 performs complete hash/data audits; with a 13GB compressed archive and possible slower networked project storage, its duration might exceed an ordinary 30-minute CPU check. Stage2's resume verification rehashes full data for each training invocation; allow enough walltime and I/O budget. After smoke, use actual observed batch/episode runtimes instead of assuming the default 12h will suffice.

If \`uv\` or \`hf\` is not available, install or load it in the cluster's permitted environment **before submission** or stop and collect the failure receipt; do not blindly upgrade pinned scientific dependencies. \`source jobs/lib.sh\` is only for jobs and is not a command the server assistant should run alone.
