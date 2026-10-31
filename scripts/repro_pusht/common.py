"""Small, dependency-light helpers shared by the explicit entry points."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
SCIENTIFIC_FILES = (
    "train.py", "eval.py", "jepa.py", "module.py", "utils.py",
    "config/train/lewm.yaml", "config/train/model/lewm.yaml",
    "config/train/data/pusht.yaml", "config/eval/pusht.yaml",
    "config/eval/solver/cem.yaml",
)
PACKAGES = (
    "torch", "torchvision", "lightning", "stable-pretraining",
    "stable-worldmodel", "hydra-core", "omegaconf", "numpy", "h5py",
    "pylance", "einops", "transformers", "timm",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def source_manifest():
    return {"upstream_revision": UPSTREAM_REVISION,
            "files": {name: sha256(ROOT / name) for name in SCIENTIFIC_FILES}}


def helper_manifest():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "scripts" / "repro_pusht").glob("*.py"))}


def json_value(value):
    """Preserve metric shapes/values without inventing aggregate reductions."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return {"nonfinite": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def write_json(path, value):
    """Create a new record; never overwrite an earlier audit by accident."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require_execute(parser, args):
    if not args.execute:
        parser.error("This command is preparation-only without --execute. No run was started.")
