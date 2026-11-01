"""Compare trusted local continuous/interrupted/resumed runs on CPU, no training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import require_execute, sha256, write_json


def compare_runs(continuous, interrupted, resumed, *, atol=1e-5, rtol=1e-4):
    paths = [Path(p).resolve() for p in (continuous, interrupted, resumed)]
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("Finite nonnegative tolerances must be declared before comparison")
    failures, tensor_errors = [], []
    leaves = 0

    def equal(left, right, name):
        nonlocal leaves
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                failures.append(name + ": keys differ")
                return
            for key in left:
                equal(left[key], right[key], f"{name}/{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                failures.append(name + ": lengths differ")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                equal(a, b, f"{name}/{index}")
        elif torch.is_tensor(left) and torch.is_tensor(right):
            leaves += 1
            if left.shape != right.shape or left.dtype != right.dtype:
                failures.append(name + ": tensor shape/dtype differs")
            elif left.is_floating_point():
                a, b = left.cpu().double(), right.cpu().double()
                maximum = float((a - b).abs().max()) if a.numel() else 0.
                tensor_errors.append({"field": name, "max_abs_error": maximum if np.isfinite(maximum) else None})
                if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a, b, atol=atol, rtol=rtol):
                    failures.append(name + ": tensor values differ")
            elif not torch.equal(left.cpu(), right.cpu()):
                failures.append(name + ": exact tensor state differs")
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            leaves += 1
            if left.dtype != right.dtype or not np.array_equal(left, right):
                failures.append(name + ": exact numpy RNG state differs")
        else:
            leaves += 1
            if isinstance(left, float) and isinstance(right, float):
                good = bool(np.isfinite([left, right]).all() and np.isclose(left, right, atol=atol, rtol=rtol))
            else:
                good = left == right
            if not good:
                failures.append(name + ": values differ")

    traces = [[json.loads(line) for line in (p / "step_trace.jsonl").read_text().splitlines()] for p in paths]
    if not traces[0] or not traces[1] or not traces[2]:
        raise ValueError("All three runs must have nonempty step traces")
    equal(traces[0], traces[1] + traces[2], "step_trace")
    inputs_observed = all(row.get("input_sha256_before_forward") for trace in traces for row in trace)
    if not inputs_observed:
        failures.append("Input identity incomplete: qualify with --trace-inputs on all three runs")
    # These are trusted user-owned full Lightning checkpoints, not downloaded objects.
    checkpoints = [torch.load(p / "checkpoints/last.ckpt", map_location="cpu", weights_only=False) for p in paths]
    a, first, b = checkpoints
    if not first["global_step"] < a["global_step"] or first["global_step"] >= b["global_step"]:
        failures.append("Interrupted checkpoint did not precede both final states")
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "epoch", "global_step"):
        equal(a[key], b[key], key)
    for key in ("version", "signature", "execution_identity", "max_epochs", "sources", "rng_python",
                "rng_numpy", "rng_torch", "rng_cuda", "train_generator", "epoch_boundary_only"):
        equal(a["pusht_repro"][key], b["pusht_repro"][key], "pusht_repro/" + key)
    receipt = json.loads((paths[2] / "resume_input.json").read_text())
    origin = paths[1] / "checkpoints/last.ckpt"
    if Path(receipt["origin"]).resolve() != origin or receipt["origin_sha256"] != sha256(origin):
        failures.append("Original interrupted checkpoint changed or wrong input was restored")
    if receipt["origin_sha256"] != receipt["staged_sha256"]:
        failures.append("Initial staged input SHA differs from source")
    return {"status": "passed" if not failures else "failed", "passed": not failures,
            "atol": atol, "rtol": rtol, "compared_leaves": leaves,
            "continuous_steps": len(traces[0]), "interrupted_steps": len(traces[1]),
            "resumed_steps": len(traces[2]), "failures": failures, "tensor_errors": tensor_errors,
            "sample_ids_observed": all(row.get("sample_ids_available", False) for row in traces[0]),
            "input_bytes_observed": bool(inputs_observed),
            "qualification_scope": "observed traces/full checkpoint state; not proof for all future worker trajectories",
            "checkpoint_sha256": {str(p): sha256(p / "checkpoints/last.ckpt") for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--interrupted", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(parser, args)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_runs(args.continuous, args.interrupted, args.resumed, atol=args.atol, rtol=args.rtol)
    write_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("Recovery comparison failed; see saved report")


if __name__ == "__main__":
    main()
