"""Source-level contracts; these do not certify a GPU/environment reproduction."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.repro_pusht.common import ROOT, json_value, write_json
from scripts.repro_pusht.convert_hf import native_config


def source_symbol(file, name, globals_):
    """Execute the actual upstream symbol without importing optional simulator packages."""
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / file), "exec"), globals_)
    return globals_[name]


def test_pinned_config_contract():
    train = yaml.safe_load((ROOT / "config/train/lewm.yaml").read_text())
    data = yaml.safe_load((ROOT / "config/train/data/pusht.yaml").read_text())
    evaluation = yaml.safe_load((ROOT / "config/eval/pusht.yaml").read_text())
    solver = yaml.safe_load((ROOT / "config/eval/solver/cem.yaml").read_text())
    assert (train["seed"], train["trainer"]["max_epochs"], train["loader"]["batch_size"]) == (3072, 100, 128)
    assert train["loss"]["sigreg"] == {"weight": .09, "kwargs": {"knots": 17, "num_proj": 1024}}
    assert data["dataset"]["name"] == "pusht_expert_train.lance"
    assert data["dataset"]["frameskip"] == 5
    assert evaluation["eval"]["dataset_name"] == "pusht_expert_train"
    assert (evaluation["seed"], evaluation["eval"]["num_eval"]) == (42, 50)
    assert (solver["num_samples"], solver["n_steps"], solver["topk"]) == (300, 30, 30)


def test_actual_forward_loss_and_target_gradient():
    forward = source_symbol("train.py", "lejepa_forward", {"torch": torch})
    sigreg_type = source_symbol("module.py", "SIGReg", {"torch": torch})
    emb = torch.linspace(-.7, 1.3, 2 * 4 * 3).reshape(2, 4, 3).requires_grad_()
    pred = torch.zeros(2, 3, 3, requires_grad=True)
    acts = torch.ones(2, 4, 2)
    batch = {"action": torch.full((2, 4, 2), float("nan"))}
    model = SimpleNamespace(encode=lambda batch: {"emb": emb, "act_emb": acts},
                            predict=lambda ctx, act: pred)
    cfg = SimpleNamespace(history_size=3, num_preds=1,
                          loss=SimpleNamespace(sigreg=SimpleNamespace(weight=.09)))
    runner = SimpleNamespace(model=model, sigreg=sigreg_type(knots=17, num_proj=8),
                             log_dict=lambda *args, **kwargs: None)
    result = forward(runner, batch, "train", cfg)
    torch.testing.assert_close(result["pred_loss"], emb[:, 1:].square().mean())
    torch.testing.assert_close(result["loss"], result["pred_loss"] + .09 * result["sigreg_loss"])
    assert torch.isfinite(batch["action"]).all() and batch["action"].count_nonzero() == 0
    target_grad = torch.autograd.grad(result["pred_loss"], emb, retain_graph=True)[0]
    assert target_grad[:, -1].abs().sum() > 0  # No hidden target stop-gradient.
    result["loss"].backward()
    assert emb.grad is not None and pred.grad is not None


def actual_sampling(dataset, cfg):
    tree = ast.parse((ROOT / "eval.py").read_text())
    run = next(node for node in tree.body if getattr(node, "name", None) == "run")
    start = next(i for i, node in enumerate(run.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "episode_len" for t in node.targets))
    end = next(i for i, node in enumerate(run.body[start:], start)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "set_policy")
    namespace = {"np": np, "dataset": dataset, "cfg": cfg,
                 "ep_indices": np.unique(dataset.get_col_data("episode_idx"))}
    namespace["get_episodes_length"] = source_symbol("eval.py", "get_episodes_length", {"np": np})
    exec(compile(ast.Module(body=run.body[start:end], type_ignores=[]), str(ROOT / "eval.py"), "exec"), namespace)
    return namespace


class ArrayDataset:
    column_names = ("episode_idx", "step_idx")

    def __init__(self, h5_path=None):
        self.cols = {"episode_idx": np.repeat([3, 8, 11], 80),
                     "step_idx": np.tile(np.arange(80), 3)}
        self.h5_path = h5_path

    def get_col_data(self, key):
        return self.cols[key]

    def get_row_data(self, indices):
        return {key: value[indices] for key, value in self.cols.items()}


def test_actual_sampling_preserves_50_rows_and_off_by_one():
    cfg = SimpleNamespace(seed=42, eval=SimpleNamespace(num_eval=50, goal_offset_steps=25))
    first, second = actual_sampling(ArrayDataset(), cfg), actual_sampling(ArrayDataset(), cfg)
    assert np.array_equal(first["random_episode_indices"], second["random_episode_indices"])
    assert len(first["random_episode_indices"]) == 50
    assert first["valid_indices"][-1] not in first["random_episode_indices"]
    assert (np.diff(first["random_episode_indices"]) > 0).all()
    assert (first["eval_start_idx"] <= 54).all()


def test_hf_config_maps_old_hydra_targets_to_local_classes():
    # Cached public HF config at revision 22b330...; no synthetic schema assumption.
    cfg = json.loads((Path(__file__).parent / "fixtures/hf-pusht-config.json").read_text())
    mapped = native_config(cfg)
    assert mapped["_target_"] == "jepa.JEPA"
    assert mapped["predictor"]["_target_"] == "module.ARPredictor"
    assert mapped["action_encoder"]["_target_"] == "module.Embedder"
    assert mapped["action_encoder"]["input_dim"] == 10
    assert mapped["predictor"]["num_frames"] == 3
    assert mapped["projector"]["norm_fn"] == {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}


def test_audits_preserve_arrays_and_refuse_overwrite(tmp_path):
    record = {"trials": np.array([0., 1.]), "tensor": torch.tensor([1, 2])}
    path = tmp_path / "receipt.json"
    write_json(path, json_value(record))
    assert json.loads(path.read_text()) == {"trials": [0., 1.], "tensor": [1, 2]}
    with pytest.raises(FileExistsError):
        write_json(path, {})


def test_selection_is_persisted_before_evaluation(tmp_path):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from scripts.repro_pusht.evaluate import install_audit

    source = tmp_path / "dataset.h5"
    source.write_bytes(b"fixture identity, not a real HDF5 file")
    dataset = ArrayDataset(source)
    audit_dir = tmp_path / "audit"

    class World:
        def evaluate(self, **kwargs):
            assert (audit_dir / "selection.json").is_file()
            return {"per_trial": np.ones(50), "success_rate": np.float64(1.)}

    upstream = SimpleNamespace(get_dataset=lambda cfg, name: dataset,
        swm=SimpleNamespace(World=World, wm=SimpleNamespace(utils=SimpleNamespace(_resolve=lambda *a: None))))
    install_audit(upstream, audit_dir)
    cfg = OmegaConf.create({"seed": 42, "policy": "random",
                           "eval": {"num_eval": 50, "goal_offset_steps": 25}})
    observed = upstream.get_dataset(cfg, "fixture")
    # Execute the original selection block against the instrumented dataset.
    selected = actual_sampling(observed, cfg)
    result = World().evaluate(dataset=dataset, episodes_idx=selected["eval_episodes"].tolist(),
        start_steps=selected["eval_start_idx"].tolist(), goal_offset=25, eval_budget=50,
        callables=None, video=tmp_path / "video")
    record = json.loads((audit_dir / "selection.json").read_text())
    assert record["rows"] == selected["random_episode_indices"].tolist()
    assert record["episode_count"] == 50
    assert len(json.loads((audit_dir / "completed.json").read_text())["metrics"]["per_trial"]) == 50
    assert result["success_rate"] == 1.


def test_evaluation_isolates_policy_outputs_without_resampling(tmp_path):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from scripts.repro_pusht.evaluate import install_audit
    from scripts.repro_pusht.common import sha256
    source = tmp_path / "old"
    source.mkdir()
    weights = source / "weights.pt"
    weights.write_bytes(b"policy bytes fixture")
    (source / "config.json").write_text('{}')
    old_sha = sha256(weights)
    data = tmp_path / "fixture.h5"
    data.write_bytes(b"dataset identity fixture")
    dataset = ArrayDataset(data)
    audit = tmp_path / "audit"
    class World:
        def evaluate(self, **kwargs):
            assert Path(kwargs["video"]) == audit / "policy"
            return {"per_trial": np.ones(50)}
    upstream = SimpleNamespace(get_dataset=lambda cfg, name: dataset,
        swm=SimpleNamespace(World=World, wm=SimpleNamespace(utils=SimpleNamespace(
            _resolve=lambda path: (Path(path), {})))))
    install_audit(upstream, audit, isolate_outputs=True)
    cfg = OmegaConf.create({"seed": 42, "policy": str(weights.resolve()),
                           "eval": {"num_eval": 50, "goal_offset_steps": 25}})
    observed = upstream.get_dataset(cfg, "fixture")
    assert Path(cfg.policy) == (audit / "policy/weights.pt").resolve()
    upstream.swm.wm.utils._resolve(cfg.policy)
    selected = actual_sampling(observed, cfg)
    World().evaluate(dataset=dataset, episodes_idx=selected["eval_episodes"].tolist(),
        start_steps=selected["eval_start_idx"].tolist(), goal_offset=25, eval_budget=50,
        callables=None, video=Path(cfg.policy).parent)
    record = json.loads((audit / "selection.json").read_text())
    assert record["rows"] == selected["random_episode_indices"].tolist()
    assert record["policy_source"]["sha256"] == record["policy_checkpoint"]["sha256"] == old_sha
    assert sha256(weights) == old_sha
    assert sorted(p.name for p in source.iterdir()) == ["config.json", "weights.pt"]


@pytest.mark.parametrize("command,arguments", [
    ("train", ["--audit-dir", "unused"]),
    ("evaluate", ["--audit-dir", "unused"]),
    ("convert_hf", ["--source", "unused", "--output", "unused"]),
    ("verify_data", ["--hdf5", "unused", "--lance", "unused", "--output-dir", "unused"]),
    ("verify_model", ["--source", "unused", "--converted", "unused", "--hdf5", "unused",
                      "--expected-hdf5-sha256", "0" * 64, "--output", "unused"]),
    ("compare_resume", ["--continuous", "unused", "--interrupted", "unused",
                        "--resumed", "unused", "--output", "unused"]),
])
def test_execution_requires_explicit_flag(command, arguments):
    result = subprocess.run([sys.executable, "-m", "scripts.repro_pusht." + command, *arguments],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "No run was started" in result.stderr


def test_converter_rejects_missing_tensor_before_writing(monkeypatch, tmp_path):
    # Unit contract for strict loading, not a substitute for the real HF gate.
    from scripts.repro_pusht import convert_hf
    cfg = json.loads((Path(__file__).parent / "fixtures/hf-pusht-config.json").read_text())
    source = tmp_path / "hf"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(cfg))
    model = torch.nn.Linear(2, 3)
    monkeypatch.setattr(convert_hf, "build_model", lambda cfg: model)
    torch.save({"weight": model.weight.detach()}, source / "weights.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        convert_hf.convert(source, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()
