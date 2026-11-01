#!/usr/bin/env bash
# Safe on the Slurm login node: read-only; never start a GPU process.
set -euo pipefail
echo "=== Host / identity ==="
date -u
hostname
id
uname -a
command -v git || true
command -v sbatch || true
command -v uv || true
command -v python3 || true
echo "=== Partitions / limits / GRES ==="
sinfo -o '%P %a %l %c %m %G' || true
echo "=== Queue and account information ==="
squeue -u "$USER" || true
command -v sacctmgr >/dev/null && sacctmgr -n show associations user="$USER" format=Account,Partition,QOS 2>/dev/null || true
echo "=== Storage ==="
if [[ -n "${STABLEWM_HOME:-}" ]]; then
  df -h "$STABLEWM_HOME" || true
  command -v quota >/dev/null && quota -s || true
else
  echo "STABLEWM_HOME not yet configured; choose approved project storage (not scratch/login home by default)."
fi
echo "=== GitHub public repository connectivity ==="
git ls-remote https://github.com/LjPeng1/le-wm.git refs/heads/experiment/pusht-baseline-reproduction || true
echo "Inventory only. Do not execute training or sbatch from this script."
