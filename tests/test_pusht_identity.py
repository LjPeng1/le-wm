"""Regression tests for old-artifact preservation and resume identity rejection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.repro_pusht.common import sha256
from scripts.repro_pusht.identity import checked_data_report, lance_table_path, reserve_training_paths, stage_resume, stage_policy, split_identity


def test_claim_checks_all_paths_before_upstream_write(tmp_path):
    root = tmp_path / "checkpoints"
    old = root / "old_export"
    old.mkdir(parents=True)
    original = old / "config.json"
    original.write_text('"original"')
    before = sha256(original)
    with pytest.raises(FileExistsError):
        reserve_training_paths(tmp_path / "new_audit", root, "new_run", "old_export")
    assert not (root / "new_run").exists()
    assert not (tmp_path / "new_audit").exists()
    assert sha256(original) == before
    paths = reserve_training_paths(tmp_path / "audit", root, "new", "new")
    assert paths["run"] == paths["export"]
    assert (Path(paths["run"]) / "pusht_output_claim.json").is_file()
    with pytest.raises(FileExistsError):
        reserve_training_paths(tmp_path / "audit2", root, "new", "new")


@pytest.mark.parametrize("name", ["../outside", ".", "/absolute"])
def test_claim_rejects_escaping_or_implicit_names(tmp_path, name):
    with pytest.raises(ValueError):
        reserve_training_paths(tmp_path / "audit", tmp_path / "root", name, "export")


def test_stage_resume_and_policy_never_replace_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weights = source / "epoch-1.pt"
    weights.write_bytes(b"checkpoint fixture; not a real model")
    (source / "config.json").write_text('{}')
    before = {p.name: sha256(p) for p in source.iterdir()}
    staged = stage_resume(weights, tmp_path / "audit")
    assert sha256(staged) == before[weights.name]
    with pytest.raises(FileExistsError):
        stage_resume(weights, tmp_path / "audit")
    policy = stage_policy(weights, tmp_path / "eval")
    assert sha256(policy) == before[weights.name]
    with pytest.raises(FileExistsError):
        stage_policy(weights, tmp_path / "eval")
    assert {p.name: sha256(p) for p in source.iterdir()} == before


def test_identity_rejects_changed_data_or_dependency(tmp_path):
    from scripts.repro_pusht.resume import ReproState
    from scripts.repro_pusht.common import source_manifest
    loader = torch.utils.data.DataLoader([1, 2, 3], generator=torch.Generator())
    current = {"data_digest": "new", "runtime": {"torch": "2.8"}}
    callback = ReproState(loader, {"epochs": 2}, tmp_path, execution_identity=current)
    state = {"version": 2, "signature": {"epochs": 2}, "max_epochs": 2,
             "sources": source_manifest(), "execution_identity": {**current, "data_digest": "old"}}
    with pytest.raises(ValueError, match="Execution identity"):
        callback.on_load_checkpoint(SimpleNamespace(max_epochs=2), None, {"pusht_repro": state})
    state["execution_identity"] = {**current, "runtime": {"torch": "changed"}}
    with pytest.raises(ValueError, match="Execution identity"):
        callback.on_load_checkpoint(SimpleNamespace(max_epochs=2), None, {"pusht_repro": state})


def test_split_identity_detects_order_change():
    data = torch.utils.data.TensorDataset(torch.arange(8))
    def loader(indices):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(data, indices), batch_size=2)
    a = split_identity(loader([0, 1, 2, 3]), loader([4, 5, 6, 7]))
    b = split_identity(loader([1, 0, 2, 3]), loader([4, 5, 6, 7]))
    assert a["train"]["indices_sha256"] != b["train"]["indices_sha256"]
    assert a["val"] == b["val"]


def test_lance_identity_resolves_parent_database_and_table(tmp_path):
    dataset = SimpleNamespace(uri=str(tmp_path), table_name="pusht_expert_train")
    assert lance_table_path(dataset) == (tmp_path / "pusht_expert_train.lance").resolve()


def test_passed_report_must_bind_current_content(tmp_path):
    from scripts.repro_pusht.common import write_json
    from scripts.repro_pusht.verify_data import content_manifest
    hdf5, lance = tmp_path / "fake.h5", tmp_path / "fake.lance"
    hdf5.write_bytes(b"identity-only fixture, not a real HDF5")
    lance.mkdir()
    (lance / "part").write_bytes(b"identity-only fixture")
    manifest = content_manifest(hdf5, lance)
    write_json(tmp_path / "data_content_manifest.json", manifest)
    report = tmp_path / "data_verification.json"
    write_json(report, {"status": "passed", "data_content_manifest_digest": manifest["digest"]})
    assert checked_data_report(report)["content_digest"] == manifest["digest"]
    report.write_text(json.dumps({"status": "passed", "data_content_manifest_digest": "wrong"}))
    with pytest.raises(ValueError, match="not bound"):
        checked_data_report(report)
    report.write_text(json.dumps({"status": "passed", "data_content_manifest_digest": manifest["digest"]}))
    hdf5.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checked_data_report(report)
