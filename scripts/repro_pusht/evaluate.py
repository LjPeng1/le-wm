"""Run unchanged eval.py with a persisted pre-rollout selection audit."""

import argparse
import os
import sys
import time
from pathlib import Path

from .common import environment, helper_manifest, json_value, require_execute, sha256, source_manifest, write_json
from .identity import stage_policy


def install_audit(upstream, audit_dir, isolate_outputs=False):
    """Observe actual row reads and World.evaluate arguments, never resample."""
    import numpy as np
    from omegaconf import OmegaConf

    audit_dir = Path(audit_dir)
    original_get_dataset = upstream.get_dataset
    original_evaluate = upstream.swm.World.evaluate
    original_resolve = upstream.swm.wm.utils._resolve
    captured = {}

    def resolve(*args, **kwargs):
        path, config = original_resolve(*args, **kwargs)
        captured["checkpoint"] = Path(path)
        captured["checkpoint_config"] = config
        return path, config

    upstream.swm.wm.utils._resolve = resolve

    def get_dataset(cfg, dataset_name):
        policy = cfg.get("policy", "random")
        if isolate_outputs:
            if policy == "random" or not Path(policy).is_absolute() or not Path(policy).is_file():
                raise ValueError("Use an explicit local absolute policy weights path for baseline evaluation")
            captured["policy_source"] = {"path": str(Path(policy).resolve()), "sha256": sha256(policy)}
            # eval.py derives videos and its text log from cfg.policy.parent.
            # The copied tensor/config bytes are unchanged; original artifacts are read-only.
            cfg.policy = str(stage_policy(policy, audit_dir))
        dataset = original_get_dataset(cfg, dataset_name)
        captured.update(cfg=cfg, dataset=dataset, row_reads=[])
        dataset_path = Path(dataset.h5_path)
        dataset_stat = dataset_path.stat()
        expected_h5 = os.environ.get("PUSHT_EXPECTED_H5_SHA256")
        if expected_h5:
            if len(expected_h5) != 64 or any(c not in "0123456789abcdef" for c in expected_h5):
                raise ValueError("PUSHT_EXPECTED_H5_SHA256 must be a lowercase SHA256")
            captured["dataset_identity"] = {"path": str(dataset_path.resolve()),
                                             "sha256": expected_h5, "bytes": dataset_stat.st_size,
                                             "stat": {"size": dataset_stat.st_size,
                                                      "mtime_ns": dataset_stat.st_mtime_ns},
                                             "rehash": "skipped_stat_bound"}
        else:
            captured["dataset_identity"] = {"path": str(dataset_path.resolve()),
                                             "sha256": sha256(dataset_path),
                                             "bytes": dataset_stat.st_size}
        original_get_rows = dataset.get_row_data

        def get_rows(indices, *args, **kwargs):
            captured["row_reads"].append(np.asarray(indices).copy())
            return original_get_rows(indices, *args, **kwargs)

        dataset.get_row_data = get_rows
        return dataset

    def evaluate(world, *args, **kwargs):
        if args:
            raise RuntimeError("Pinned eval.py uses keyword arguments; inspect upstream drift")
        cfg, dataset = captured["cfg"], captured["dataset"]
        if kwargs["dataset"] is not dataset or len(captured["row_reads"]) != 2:
            raise RuntimeError("Unexpected dataset row-read path; refusing unaudited rollout")
        rows = captured["row_reads"][0]
        if not np.array_equal(rows, captured["row_reads"][1]):
            raise RuntimeError("Episode/start-step row selections differ")
        col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episodes = np.asarray(dataset.get_col_data(col))[rows]
        steps = np.asarray(dataset.get_col_data("step_idx"))[rows]
        if not np.array_equal(episodes, kwargs["episodes_idx"]) or not np.array_equal(steps, kwargs["start_steps"]):
            raise RuntimeError("Selected rows do not match actual World.evaluate arguments")
        record = {
            "phase": "selected_before_world_evaluate", "sources": source_manifest(),
            "helpers": helper_manifest(), "dataset": captured["dataset_identity"],
            "environment": environment(),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "sampling": "unchanged eval.py: choice(len(valid_indices)-1, replace=False)",
            "rows": rows.tolist(), "episodes": episodes.tolist(), "start_steps": steps.tolist(),
            "goal_offset": kwargs["goal_offset"], "eval_budget": kwargs["eval_budget"],
            "episode_count": len(episodes),
            "policy_source": captured.get("policy_source"),
            "output_isolation": isolate_outputs,
        }
        policy = cfg.get("policy", "random")
        if policy != "random":
            checkpoint = captured["checkpoint"]
            record["policy_checkpoint"] = {"path": str(checkpoint.resolve()),
                                             "sha256": sha256(checkpoint),
                                             "config": captured["checkpoint_config"]}
        import torch
        record["device"] = {"cuda_available": torch.cuda.is_available(),
                              "cuda_version": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        write_json(audit_dir / "selection.json", record)
        start = time.monotonic()
        try:
            metrics = original_evaluate(world, **kwargs)
        except BaseException as error:
            write_json(audit_dir / "failed.json", {"phase": "world_evaluate_failed",
                       "error_type": type(error).__name__, "error": str(error),
                       "wall_seconds": time.monotonic() - start})
            raise
        write_json(audit_dir / "completed.json", {"phase": "world_evaluate_returned",
                   "metrics": json_value(metrics), "wall_seconds": time.monotonic() - start,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                   "video_directory": str(kwargs["video"]),
                   "selection_sha256": sha256(audit_dir / "selection.json")})
        return metrics

    upstream.get_dataset = get_dataset
    upstream.swm.World.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args, overrides = parser.parse_known_args()
    require_execute(parser, args)
    args.audit_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.audit_dir / "entry_started.json", {"phase": "evaluation_entry", "overrides": overrides})
    try:
        import eval as upstream
        install_audit(upstream, args.audit_dir.resolve(), isolate_outputs=True)
        sys.argv = ["eval.py", *overrides]
        # Hydra resolves a relative @hydra.main(config_path="./config/eval") as
        # a package path ("pkg://..config.eval") unless the task function looks
        # like __main__; when eval.py is imported, run.__module__ == "eval" and
        # the config module cannot be found. Present it as a __main__ function
        # so Hydra resolves config_path relative to eval.py's file instead.
        upstream.run.__module__ = "__main__"
        if hasattr(upstream.run, "__wrapped__"):
            upstream.run.__wrapped__.__module__ = "__main__"
        upstream.run()
    except BaseException as error:
        if not (args.audit_dir / "failed.json").exists():
            write_json(args.audit_dir / "failed.json", {"phase": "evaluation_entry_failed",
                       "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
