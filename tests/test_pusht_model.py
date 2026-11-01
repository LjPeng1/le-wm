"""Small real JEPA/SWM methods, with a tiny encoder; no official-asset claim.

Only optional package initializers and the encoder factory are isolated. The
actual SWM LeWM/Predictor/Embedder/MLP and local JEPA methods are executed.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.repro_pusht.common import ROOT
from scripts.repro_pusht import convert_hf, verify_model


class TinyEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, pixels, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.linear(pixels).unsqueeze(1))


def load_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_source(monkeypatch):
    """Use installed pinned SWM, or actual cached source in explicit isolation mode."""
    if os.environ.get("PUSHT_SOURCE_ISOLATION") != "1":
        pytest.importorskip("stable_worldmodel")
        import stable_worldmodel.wm.lewm as lewm
        import stable_pretraining.backbone.utils as backbone
        monkeypatch.setattr(backbone, "vit_hf", TinyEncoder)
        return lewm
    source = Path(os.environ.get("PUSHT_SWM_SOURCE", ROOT.parent /
        "tmp/pusht-dependency-audit/stable-worldmodel/stable_worldmodel"))
    if not (source / "wm/lewm/lewm.py").is_file():
        pytest.fail("Actual SWM source required for isolation; no substitute implementation")
    packages = {}
    for name in ("stable_worldmodel", "stable_worldmodel.wm", "stable_worldmodel.data",
                 "stable_worldmodel.utils", "stable_pretraining", "stable_pretraining.backbone",
                 "stable_pretraining.backbone.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        packages[name] = package
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(packages[parent], attribute, package)
    packages["stable_worldmodel.utils"].HF_BASE_URL = "https://huggingface.co"
    packages["stable_worldmodel.data"].get_cache_dir = lambda cache_dir, sub_folder: Path(cache_dir) / sub_folder
    packages["stable_worldmodel.data"].ensure_dir_exists = lambda path: path.mkdir(parents=True, exist_ok=True)
    packages["stable_pretraining.backbone.utils"].vit_hf = TinyEncoder
    lewm = load_file("stable_worldmodel.wm.lewm", source / "wm/lewm/lewm.py", monkeypatch)
    lewm.__path__ = []
    packages["stable_worldmodel.wm"].lewm = lewm
    lewm.module = load_file("stable_worldmodel.wm.lewm.module", source / "wm/lewm/module.py", monkeypatch)
    utils = load_file("stable_worldmodel.wm.utils", source / "wm/utils.py", monkeypatch)
    packages["stable_worldmodel.wm"].utils = utils
    return lewm


@pytest.fixture
def tiny_assets(tmp_path, model_source):
    cfg = json.loads((ROOT / "tests/fixtures/hf-pusht-config.json").read_text())
    cfg["predictor"].update(input_dim=4, hidden_dim=4, output_dim=4, depth=1,
                            heads=1, dim_head=4, mlp_dim=8)
    cfg["action_encoder"]["emb_dim"] = 4
    for key in ("projector", "pred_proj"):
        cfg[key].update(input_dim=4, output_dim=4, hidden_dim=8)
    source, converted = tmp_path / "hf_fixture", tmp_path / "converted_fixture"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(71)
    torch.save(convert_hf.build_model(cfg).state_dict(), source / "weights.pt")
    convert_hf.convert(source, converted)
    return source, converted


def inputs(history=3):
    # These synthetic tensors belong only to this labelled fixture test.
    pixels = torch.linspace(-1, 1, history * 4).reshape(1, 1, history, 4).expand(1, 2, history, 4).clone()
    goal = torch.tensor([.2, -.1, .4, .7]).reshape(1, 1, 1, 4).expand(1, 2, 1, 4).clone()
    actions = torch.linspace(-.8, .9, 2 * 5 * 10).reshape(1, 2, 5, 10)
    return {"pixels": pixels, "goal": goal, "action": actions[:, :, :history].clone(),
            "metadata": {"kind": "tiny_synthetic_fixture_only", "mutable": [1, 2]}}, actions


@pytest.mark.parametrize("history", [1, 3])
def test_actual_swm_jepa_native_methods_and_repeated_cache(tiny_assets, history, monkeypatch):
    torch.set_num_threads(1)
    # A CPU default must not probe or initialize CUDA, including on CUDA hosts.
    def forbidden(*args, **kwargs):
        raise AssertionError("Default function checks must not touch CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    models, loading = verify_model.load_models(*tiny_assets)
    assert loading["same_state_exact"] and loading["strict_source_load"] == ["swm", "jepa"]
    info, actions = inputs(history)
    original = copy.deepcopy(info)
    report = verify_model.verify_models(models, info, actions)
    assert report["passed"] and report["device"] == "cpu" and report["dtype"] == "torch.float32"
    assert "official_asset_run" not in report
    assert len([item for item in report["comparisons"]
                if item["stage"].startswith("swm_vs_jepa/rollout_step_")]) == 6
    assert all(item["reference"]["finite"] and item["candidate"]["finite"]
               for item in report["comparisons"])
    assert all(item["max_abs_error"] <= 1e-5 for item in report["comparisons"])
    assert all(not model.training for model in models.values())
    assert info.keys() == original.keys() and info["metadata"] == original["metadata"]
    for key in ("pixels", "goal", "action"):
        assert torch.equal(info[key], original[key])


def test_mismatch_fails_without_relaxing_tolerance(tiny_assets, monkeypatch):
    models, _ = verify_model.load_models(*tiny_assets)
    original = models["jepa"].predict
    monkeypatch.setattr(models["jepa"], "predict", lambda *args: original(*args) + .02)
    with pytest.raises(verify_model.ModelMismatch) as failure:
        verify_model.verify_models(models, *inputs())
    report = failure.value.report
    assert not report["passed"] and (report["atol"], report["rtol"]) == (1e-5, 1e-4)
    assert any(not row["passed"] and row["stage"] == "swm_vs_jepa/predict" for row in report["comparisons"])
    with pytest.raises(ValueError, match="fixed"):
        verify_model.verify_models(models, *inputs(), atol=.1)


def test_repeated_cost_error_is_observed(tiny_assets, monkeypatch):
    models, _ = verify_model.load_models(*tiny_assets)
    model = models["swm"]
    original = model.get_cost
    calls = []
    def corrupt_repeated(info, actions):
        calls.append((id(info), "goal_emb" in info, "emb" in info))
        value = original(info, actions)
        return value if len(calls) == 1 else value + .1
    monkeypatch.setattr(model, "get_cost", corrupt_repeated)
    with pytest.raises(verify_model.ModelMismatch) as failure:
        verify_model.verify_models(models, *inputs())
    assert calls[0][0] == calls[1][0]
    assert calls[0][1:] == (False, False) and calls[1][1:] == (True, True)
    assert any(row["stage"] == "swm/cost_first_vs_cost_repeated" and not row["passed"]
               for row in failure.value.report["comparisons"])


def test_native_reload_rejects_corrupted_state_and_config(tiny_assets):
    source, converted = tiny_assets
    config_path = converted / "config.json"
    config = json.loads(config_path.read_text())
    changed = copy.deepcopy(config)
    changed["predictor"]["num_frames"] = 2
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="native config"):
        verify_model.load_models(source, converted)
    config_path.write_text(json.dumps(config))
    state = torch.load(converted / "weights.pt", weights_only=True)
    key = next(key for key, value in state.items() if value.is_floating_point())
    state[key] = state[key] + 1.
    torch.save(state, converted / "weights.pt")
    with pytest.raises(ValueError, match="native_reload state"):
        verify_model.load_models(source, converted)


def test_source_strict_load_rejects_missing_tensor(tiny_assets):
    source, converted = tiny_assets
    state = torch.load(source / "weights.pt", weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, source / "weights.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        verify_model.load_models(source, converted)


def test_comparison_rejects_shape_and_nonfinite():
    for candidate in (torch.ones(3), torch.tensor([float("nan"), 1.])):
        result = verify_model.compare_tensor("bad", torch.ones(2), candidate)
        assert not result["passed"] and result["max_abs_error"] is None


def test_real_asset_guards_do_not_accept_tiny_fixture(tiny_assets, tmp_path):
    source, converted = tiny_assets
    h5 = tmp_path / "not_real.h5"
    h5.write_bytes(b"synthetic fixture, not a released HDF5")
    with pytest.raises(ValueError, match="revision"):
        verify_model.check_asset_identity(source, h5, "0" * 64, "floating-main")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_model.check_asset_identity(source, h5, "0" * 64)
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_model.run(source, converted, h5, output, "0" * 64)
    failed = json.loads((output / "failed.json").read_text())
    assert failed["status"] == "failed" and failed["training_started"] is False
    with pytest.raises(FileExistsError):
        verify_model.run(source, converted, h5, output, "0" * 64)


def test_cli_requires_execute_before_any_work(tmp_path):
    result = subprocess.run([sys.executable, "-m", "scripts.repro_pusht.verify_model",
        "--source", "absent", "--converted", "absent", "--hdf5", "absent",
        "--expected-hdf5-sha256", "0" * 64, "--output", str(tmp_path / "audit")],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "No run was started" in result.stderr
    assert not (tmp_path / "audit").exists()
