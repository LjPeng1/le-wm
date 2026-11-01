"""Content identities and exclusive output claims; no training or downloads."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

from .common import ROOT, PACKAGES, helper_manifest, sha256, source_manifest, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def runtime_identity():
    """Compare installed versions and install origins, not machine-specific paths."""
    packages = {}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
            direct = dist.read_text("direct_url.json")
            packages[name] = {"version": dist.version,
                              "origin_sha256": digest(json.loads(direct)) if direct else None}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    audited = json.loads((ROOT / "docs/PUSHT_SOURCE_MANIFEST.json").read_text())["dependency_candidates"]
    for dependency in audited:
        name = dependency["name"]
        if packages.get(name) is None:
            continue
        dist = importlib.metadata.distribution(name)
        files = {}
        for relative in dependency["files"]:
            path = Path(dist.locate_file(relative))
            files[relative] = (hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                               if path.is_file() else None)
        packages[name]["audited_source_files_lf"] = files
    return {"python": platform.python_version(), "packages": packages}


def code_identity():
    try:
        head = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                               "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None  # Source hashes still identify an exported source tree.
    return {"git_head": head, "scientific": source_manifest(), "helpers": helper_manifest(),
            "runtime": runtime_identity(),
            "lock_sha256": sha256(ROOT / "requirements/repro-pusht.txt")}


def checked_data_report(path):
    """Rehash frozen inputs; a successful old report alone is not a cache check."""
    from .verify_data import validate_content_manifest
    path = Path(path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("Data verification report must have status=passed")
    manifest_path = path.parent / "data_content_manifest.json"
    manifest = validate_content_manifest(manifest_path)
    if report.get("data_content_manifest_digest") != manifest["digest"]:
        raise ValueError("Data verification report is not bound to this content manifest")
    return {"report_sha256": sha256(path), "manifest_sha256": sha256(manifest_path),
            "content_digest": manifest["digest"], "roots": manifest["roots"]}


def split_identity(train_loader, val_loader):
    import numpy as np
    result = {}
    for name, loader in (("train", train_loader), ("val", val_loader)):
        subset = loader.dataset
        if not hasattr(subset, "indices"):
            raise ValueError("Pinned train.py must supply explicit split indices")
        indices = np.asarray(subset.indices, dtype="<i8")
        if indices.ndim != 1:
            raise ValueError("Split indices must be one-dimensional")
        result[name] = {"count": len(indices), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        "batch_size": loader.batch_size, "drop_last": loader.drop_last,
                        "num_workers": loader.num_workers}
    return result


def lance_table_path(dataset):
    # Pinned SWM separates the parent database URI from the table name.
    return (Path(dataset.uri) / (str(dataset.table_name) + ".lance")).resolve()


def _under(root, name):
    name = Path(str(name))
    if name.is_absolute() or not name.parts or ".." in name.parts or name == Path("."):
        raise ValueError("Use an explicit new relative subdir/output_model_name")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Output must stay inside the checkpoint root")
    return target


def reserve_training_paths(audit_dir, checkpoint_root, subdir, output_model_name):
    """Reject collisions before upstream train.py writes config.yaml/exports."""
    root = Path(checkpoint_root).resolve()
    paths = {"audit": Path(audit_dir).resolve(), "run": _under(root, subdir),
             "export": _under(root, output_model_name)}
    unique = set(paths.values())
    for path in unique:
        if path.exists():
            raise FileExistsError(f"Refusing an existing output directory: {path}")
        if any(path != other and (path.is_relative_to(other) or other.is_relative_to(path))
               for other in unique):
            raise ValueError("Audit/run/export directories may coincide but may not nest")
    for path in sorted(unique):
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "pusht_output_claim.json", {k: str(v) for k, v in paths.items()})
    return {k: str(v) for k, v in paths.items()}


def stage_resume(source, audit_dir):
    """A copied last.ckpt matches Manager's save callback without touching source."""
    source = Path(source).resolve()
    target = Path(audit_dir).resolve() / "checkpoints/last.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    after = sha256(target)
    if before != after or sha256(source) != before:
        raise ValueError("Resume source changed or staged copy hash mismatch")
    receipt = {"origin": str(source), "origin_sha256": before,
               "staged": str(target), "staged_sha256": after}
    write_json(Path(audit_dir) / "resume_input.json", receipt)
    return target


def stage_policy(weights, audit_dir):
    """Keep original eval paths/parameters, isolate its fixed-name video exports."""
    weights = Path(weights).resolve()
    config = weights.parent / "config.json"
    target = Path(audit_dir).resolve() / "policy"
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for src, name in ((weights, "weights.pt"), (config, "config.json")):
        before = sha256(src)
        with src.open("rb") as reader, (target / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if sha256(target / name) != before or sha256(src) != before:
            raise ValueError("Policy source changed or staged copy hash mismatch")
        entries.append({"source": str(src), "name": name, "sha256": before})
    write_json(Path(audit_dir) / "policy_input.json", {"files": entries})
    return target / "weights.pt"
