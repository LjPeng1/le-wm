"""Read-only inspection. Does not import a simulator, download, or train."""

import argparse
import json
import os
from pathlib import Path

from .common import environment, source_manifest, write_json


def inspect(cache_root):
    cache_root = Path(cache_root).expanduser().resolve()
    names = ("datasets/pusht_expert_train.lance", "datasets/pusht_expert_train.h5",
             "hf_pusht/config.json", "hf_pusht/weights.pt")
    return {
        "mode": "read_only_preflight", "training_started": False,
        "environment": environment(), "sources": source_manifest(),
        "cache_root": str(cache_root),
        "assets": {name: (cache_root / name).exists() for name in names},
        "upstream_defaults": {
            "train_dataset": "pusht_expert_train.lance",
            "eval_dataset": "pusht_expert_train (HDF5Dataset)",
            "max_epochs": 100, "batch_size": 128, "split_loader_seed": 3072,
            "global_seed_set_by_upstream": False, "eval_seed": 42,
            "num_eval": 50, "frameskip": 5, "history_size": 3,
            "num_preds": 1, "sigreg_weight": 0.09,
            "eval_sampling": "choice(len(valid_indices)-1, replace=False); sorted row IDs",
        },
        "first_round": {"label": "paper10_candidate", "max_epochs_from_update_zero": 10,
                        "global_seed_declared_patch": 3072, "extra_training_arms": 0,
                        "training_authorized": False},
        "pending": [
            "Gate 1: actual Linux installation, release bytes, HDF5/Lance data verification",
            "Gate 2: actual official weights strict-load and function consistency verification",
            "Gate 3: released-checkpoint unchanged 50-start CEM; resource approval required",
            "Gate 4: real-data multi-worker/bf16 resume qualification; resource approval required",
            "Gate 5: one paper10 candidate; separate training approval required",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=os.environ.get("STABLEWM_HOME", "~/.stable-wm"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.cache_root)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
