"""Opt-in, local-only official LeWM/JEPA function checks; never a control score.

The CLI requires real release weights and real HDF5 windows. ``verify_models``
also supports small fixtures, but its result alone is not official-asset acceptance.
No assets are downloaded. CPU/float32/eval are the defaults, including on CUDA hosts.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .common import ROOT, environment, require_execute, sha256, source_manifest, write_json
from .convert_hf import build_model, native_config


class ModelMismatch(RuntimeError):
    def __init__(self, report):
        self.report = report
        failed = [item["stage"] for item in report["comparisons"] if not item["passed"]]
        super().__init__("Model function mismatch: " + ", ".join(failed))


def clone_input(value, device="cpu"):
    """Each implementation owns its mutable dictionary and cached tensors."""
    import torch

    if torch.is_tensor(value):
        return value.detach().clone().to(device=device,
            dtype=torch.float32 if value.is_floating_point() else value.dtype)
    if isinstance(value, dict):
        return {key: clone_input(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_input(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_input(item, device) for item in value)
    return copy.deepcopy(value)


def tensor_summary(value):
    import torch

    value = value.detach()
    finite = bool(torch.isfinite(value).all())
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "finite": finite}


def compare_tensor(stage, reference, candidate, atol=1e-5, rtol=1e-4):
    """Relative error uses abs(reference).clamp_min(1e-12); allclose is asymmetric."""
    import torch

    result = {"stage": stage, "reference": tensor_summary(reference),
              "candidate": tensor_summary(candidate), "max_abs_error": None,
              "max_relative_error": None, "passed": False}
    if reference.shape != candidate.shape or not (result["reference"]["finite"]
                                                  and result["candidate"]["finite"]):
        return result
    left, right = reference.detach().cpu().double(), candidate.detach().cpu().double()
    difference = (left - right).abs()
    result.update(max_abs_error=float(difference.max()) if difference.numel() else 0.,
        max_relative_error=float((difference / left.abs().clamp_min(1e-12)).max())
            if difference.numel() else 0.,
        passed=bool(torch.allclose(left, right, atol=atol, rtol=rtol)))
    return result


def verify_models(models, info, action_candidates, *, device="cpu", atol=1e-5, rtol=1e-4):
    """Compare actual model methods, including repeated calls on each own cache.

    ``info`` has solver-expanded (B,S,H,...) observations and goals; candidates
    have (B,S,horizon,action_dim). Cached input fields are rejected. No planner,
    sampler, loss or model method is replaced. A mismatch raises with ``report``.
    """
    import torch

    if len(models) < 2 or any(key in info for key in ("emb", "goal_emb", "predicted_emb", "act_emb")):
        raise ValueError("Need two models and fresh observations without model caches")
    if atol != 1e-5 or rtol != 1e-4:
        raise ValueError("The initial engineering tolerance is fixed at atol=1e-5, rtol=1e-4")
    if action_candidates.ndim != 4 or info["pixels"].ndim < 4:
        raise ValueError("Expected solver-expanded observations and 4D action candidates")
    history = info["pixels"].shape[2]
    if not 1 <= history <= action_candidates.shape[2]:
        raise ValueError("Observation history must fit the action horizon")
    if tuple(info["pixels"].shape[:2]) != tuple(action_candidates.shape[:2]):
        raise ValueError("Observation/candidate batch and sample shapes differ")
    actions = clone_input(action_candidates, device)
    outputs, model_details = {}, {}
    with torch.inference_mode():
        for name, model in models.items():
            model.to(device=device).float().eval().requires_grad_(False)
            model_details[name] = {"class": type(model).__module__ + "." + type(model).__name__,
                "training": model.training, "parameters": sorted({(str(p.device), str(p.dtype))
                    for p in model.parameters()})}
            own = clone_input(info, device)
            encoded_info = {key: value[:, 0] for key, value in own.items() if torch.is_tensor(value)}
            encoded_info["action"] = actions[:, 0, :history].clone()
            encoded = model.encode(encoded_info)
            result = {"encode": encoded["emb"].clone(),
                "action_encoder": model.action_encoder(actions.flatten(0, 1)).clone(),
                "encode_action": encoded["act_emb"].clone(),
                "predict": model.predict(encoded["emb"], encoded["act_emb"]).clone()}
            rolled = model.rollout(clone_input(info, device), actions.clone())
            result["rollout"] = rolled["predicted_emb"].clone()
            for index in range(result["rollout"].shape[2]):
                result[f"rollout_step_{index:02d}"] = result["rollout"][:, :, index].clone()
            goal = model.encode({"pixels": own["goal"][:, 0].clone()})["emb"]
            rolled["goal_emb"] = goal
            result["final_cost"] = model.criterion(rolled).clone()
            # Preserve each model's own mutated cache between first and repeated calls.
            cached = clone_input(info, device)
            result["cost_first"] = model.get_cost(cached, actions.clone()).clone()
            result["cost_repeated"] = model.get_cost(cached, actions.clone()).clone()
            outputs[name] = result
    reference_name = next(iter(outputs))
    comparisons = []
    for name, result in outputs.items():
        for first, second in (("final_cost", "cost_first"), ("cost_first", "cost_repeated")):
            comparisons.append(compare_tensor(f"{name}/{first}_vs_{second}",
                result[first], result[second], atol, rtol))
        if name == reference_name:
            continue
        for stage, reference in outputs[reference_name].items():
            comparisons.append(compare_tensor(f"{reference_name}_vs_{name}/{stage}",
                reference, result[stage], atol, rtol))
    report = {"scope": "function_checks_only; caller must establish real asset provenance",
        "device": str(device), "dtype": "torch.float32", "eval": True,
        "atol": atol, "rtol": rtol, "relative_error_denominator": "abs(reference).clamp_min(1e-12)",
        "history_length": history, "models": model_details,
        "inputs": {key: tensor_summary(value) for key, value in info.items() if torch.is_tensor(value)},
        "actions": tensor_summary(action_candidates), "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons)}
    if not report["passed"]:
        raise ModelMismatch(report)
    return report


def check_asset_identity(source, hdf5, expected_hdf5_sha256, source_revision=None):
    """Release byte identity, not a floating HF name; raw H5 hash must be supplied."""
    manifest = json.loads((ROOT / "docs/PUSHT_SOURCE_MANIFEST.json").read_text())
    release = manifest["official_model"]
    source, hdf5 = Path(source).resolve(), Path(hdf5).resolve()
    revision = source_revision or release["revision"]
    if revision != release["revision"]:
        raise ValueError("Source revision differs from pinned official model release")
    if len(expected_hdf5_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_hdf5_sha256):
        raise ValueError("An expected lowercase SHA256 of the real HDF5 is required")
    weight = next(item for item in release["files"] if item["rfilename"] == "weights.pt")
    checks = [(source / "weights.pt", weight["lfs"]["sha256"]),
              (source / "config.json", release["config_sha256"]),
              (hdf5, expected_hdf5_sha256)]
    assets = {}
    for path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(f"Local asset missing; no download attempted: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Asset SHA256 mismatch: {path}")
        assets[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
    return {"model_revision": revision, "dataset_revision": manifest["official_dataset"]["revision"],
            "hdf5_identity": "expected unpacked hash supplied by caller; archive provenance is a separate data gate",
            "assets": assets}


def load_models(source, converted, *, loader_cache=None):
    """Strict same-state source implementations and actual native local loader."""
    import torch
    from hydra.utils import instantiate
    from stable_worldmodel.wm.utils import load_pretrained

    source, converted = Path(source).resolve(), Path(converted).resolve()
    cfg = json.loads((source / "config.json").read_text())
    native = json.loads((converted / "config.json").read_text())
    if native != native_config(cfg):
        raise ValueError("Converted native config differs from native_config(source)")
    if not (converted / "weights.pt").is_file():
        raise FileNotFoundError("Converted local weights.pt missing; no download attempted")
    state = torch.load(source / "weights.pt", map_location="cpu", weights_only=True)
    models = {"swm": instantiate(cfg), "jepa": build_model(cfg)}
    for model in models.values():
        model.load_state_dict(state, strict=True)
    # Absolute existing .pt path forces the pinned loader's local-file path.
    cache = Path(loader_cache).resolve() if loader_cache is not None else converted.parent
    models["native_reload"] = load_pretrained(str(converted / "weights.pt"), cache_dir=str(cache))
    for name, model in models.items():
        loaded = model.state_dict()
        if loaded.keys() != state.keys() or any(loaded[key].shape != state[key].shape
                or not torch.equal(loaded[key], state[key]) for key in state):
            raise ValueError(f"{name} state differs from official strict-loaded tensors")
    return models, {"strict_source_load": ["swm", "jepa"], "native_loader": "stable_worldmodel.wm.utils.load_pretrained",
        "same_state_exact": True, "tensor_count": len(state), "source_config": cfg,
        "native_config": native, "converted_sha256": {name: sha256(converted / name)
            for name in ("config.json", "weights.pt")}}


def real_hdf5_inputs(hdf5, row=0):
    """Use original eval transforms/scalers on explicit, real expert windows.

    Two expert action sequences are probes, not CEM samples or a control score.
    H=1 is the original eval default; H=3 additionally checks the model context.
    """
    import numpy as np
    import torch
    import eval as upstream
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.data import HDF5Dataset
    from stable_worldmodel.policy import BasePolicy

    cfg = OmegaConf.load(ROOT / "config/eval/pusht.yaml")
    dataset = HDF5Dataset(path=str(Path(hdf5).resolve()), keys_to_cache=list(cfg.dataset.keys_to_cache))
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes, steps = dataset.get_col_data(col), dataset.get_col_data("step_idx")
    if not 0 <= row < len(episodes):
        raise ValueError("Requested HDF5 row out of range")
    episode, start = int(episodes[row]), int(steps[row])
    rows_by_step = {int(steps[index]): int(index) for index in np.flatnonzero(episodes == episode)}
    block, horizon, goal_offset = int(cfg.plan_config.action_block), int(cfg.plan_config.horizon), int(cfg.eval.goal_offset_steps)
    goal_row = rows_by_step.get(start + goal_offset)
    required = list(range(start, start + horizon * block + 1))
    if goal_row is None or any(step not in rows_by_step for step in required):
        raise ValueError("Requested row lacks contiguous same-episode action/goal windows")
    processors, normalizers = {}, {}
    for key in cfg.dataset.keys_to_cache:
        if key == "pixels":
            continue
        data = dataset.get_col_data(key)
        scaler = StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        processors[key] = scaler
        if key != "action":
            processors["goal_" + key] = scaler
        normalizers[key] = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist(),
                            "scale": scaler.scale_.tolist(), "samples": int(scaler.n_samples_seen_)}
    policy = BasePolicy(process=processors,
        transform={"pixels": upstream.img_transform(cfg), "goal": upstream.img_transform(cfg)})
    candidate_rows = [[rows_by_step[step] for step in range(start + offset, start + offset + horizon * block)]
                      for offset in (0, 1)]
    raw_actions = np.stack([dataset.get_row_data(rows)["action"] for rows in candidate_rows])
    if not np.isfinite(raw_actions).all():
        raise ValueError("Nonfinite real action window")
    normalized = processors["action"].transform(raw_actions.reshape(-1, raw_actions.shape[-1])).reshape(raw_actions.shape)
    actions = torch.from_numpy(normalized.reshape(1, 2, horizon, -1)).float()
    cases, windows = {}, {}
    for history in (1, 3):
        history_rows = [rows_by_step[start + index * block] for index in range(history)]
        observed, goal = dataset.get_row_data(history_rows), dataset.get_row_data([goal_row])
        raw = {"pixels": observed["pixels"][None], "goal": goal["pixels"][None]}
        for key in cfg.dataset.keys_to_cache:
            if key != "pixels":
                raw[key] = observed[key][None]
                if key != "action":
                    raw["goal_" + key] = goal[key][None]
        prepared = policy._prepare_info(raw)
        # Solver expands each original observation across its action candidates.
        info = {key: value.unsqueeze(1).expand(1, 2, *value.shape[1:]).clone()
                for key, value in prepared.items() if torch.is_tensor(value)}
        cases[f"history_{history}"] = (info, actions.clone())
        windows[f"history_{history}"] = {"observation_rows": history_rows,
            "goal_row": goal_row, "raw_numeric": {key: value.tolist() for key, value in raw.items()
                if key not in ("pixels", "goal")},
            "normalized_numeric": {key: value.tolist() for key, value in prepared.items()
                if key not in ("pixels", "goal")},
            "raw_pixels_sha256": hashlib.sha256(raw["pixels"].tobytes()).hexdigest(),
            "transformed_pixels_sha256": hashlib.sha256(prepared["pixels"].contiguous().numpy().tobytes()).hexdigest()}
    return cases, {"input_kind": "real_HDF5_expert_windows", "episode": episode, "start_step": start,
        "row": row, "action_block": block, "horizon": horizon, "goal_offset": goal_offset,
        "candidate_rows": candidate_rows, "raw_actions": raw_actions.tolist(),
        "normalized_action_blocks": actions.tolist(), "normalizers": normalizers, "windows": windows,
        "preprocessing": "original eval.img_transform and SWM BasePolicy._prepare_info; all-dataset eval StandardScaler",
        "not_a_control_evaluation": True}


def run(source, converted, hdf5, output, expected_hdf5_sha256, *, row=0, device="cpu", source_revision=None):
    output = Path(output).resolve()
    # Claim before work; an existing directory always rejects, even if empty.
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "official_asset_run": False, "training_started": False,
              "control_evaluation_started": False, "device": device, "dtype": "torch.float32"}
    write_json(output / "started.json", report)
    try:
        report["identity"] = check_asset_identity(source, hdf5, expected_hdf5_sha256, source_revision)
        report["sources"] = source_manifest()
        report["environment"] = environment()
        models, report["loading"] = load_models(source, converted, loader_cache=output / "loader_cache")
        cases, report["inputs"] = real_hdf5_inputs(hdf5, row)
        report["official_asset_run"] = True
        report["cases"] = {}
        for name, (info, actions) in cases.items():
            try:
                report["cases"][name] = verify_models(models, info, actions, device=device)
            except ModelMismatch as error:
                report["cases"][name] = error.report
                raise
        report["status"] = "passed_function_checks"
        write_json(output / "completed.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "failed.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing pinned HF release directory")
    parser.add_argument("--converted", type=Path, required=True, help="Existing native conversion directory")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Exclusive new audit directory")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Defaults to CPU; any other device must be explicitly authorized")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(parser, args)
    record = run(args.source, args.converted, args.hdf5, args.output, args.expected_hdf5_sha256,
                 row=args.row, device=args.device, source_revision=args.source_revision)
    print(json.dumps({"status": record["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
