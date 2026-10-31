"""Strict local HF conversion following the pinned upstream README recipe."""

import argparse
import json
from pathlib import Path

from .common import environment, require_execute, sha256, source_manifest, write_json


def build_model(cfg):
    import torch
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"], patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"], pretrained=False, use_mask_token=False,
    )

    def mlp(key):
        return MLP(input_dim=cfg[key]["input_dim"], output_dim=cfg[key]["output_dim"],
                   hidden_dim=cfg[key]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

    plain = lambda key: {k: v for k, v in cfg[key].items() if not k.startswith("_")}
    return JEPA(encoder=encoder, predictor=ARPredictor(**plain("predictor")),
                action_encoder=Embedder(**plain("action_encoder")),
                projector=mlp("projector"), pred_proj=mlp("pred_proj"))


def native_config(cfg):
    """Translate README constructor arguments to the current SWM Hydra format."""
    result = {"_target_": "jepa.JEPA"}
    plain = lambda key: {k: v for k, v in cfg[key].items() if not k.startswith("_")}
    result["encoder"] = {**plain("encoder"), "_target_": "stable_pretraining.backbone.utils.vit_hf",
                         "pretrained": False, "use_mask_token": False}
    result["predictor"] = {**plain("predictor"), "_target_": "module.ARPredictor"}
    result["action_encoder"] = {**plain("action_encoder"), "_target_": "module.Embedder"}
    for key in ("projector", "pred_proj"):
        result[key] = {**plain(key), "_target_": "module.MLP",
                       "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}}
    return result


def convert(source, output, legacy_object=False):
    import torch

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    cfg_path, weights_path = source / "config.json", source / "weights.pt"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = build_model(cfg)
    # HF contains tensors only; object deserialization is neither required nor allowed.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    if legacy_object:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            torch.save(model, stream)
        artifacts = {output.name: sha256(output)}
        audit_path = output.with_suffix(output.suffix + ".audit.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "weights.pt").open("xb") as stream:
            torch.save(model.state_dict(), stream)
        write_json(output / "config.json", native_config(cfg))
        artifacts = {name: sha256(output / name) for name in ("weights.pt", "config.json")}
        audit_path = output / "conversion.audit.json"
    record = {"conversion": "upstream_README_strict_state_dict", "strict": True,
              "sources": source_manifest(), "environment": environment(),
              "input_config_sha256": sha256(cfg_path),
              "input_weights_sha256": sha256(weights_path),
              "artifacts_sha256": artifacts, "output": str(output.resolve()),
              "format": "legacy_object" if legacy_object else "native_weights_and_hydra_config",
              "tensor_count": len(state_dict), "training_started": False}
    write_json(audit_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New destination folder under STABLEWM_HOME/checkpoints for native eval")
    parser.add_argument("--legacy-object", action="store_true",
                        help="Write the old README object format instead; current eval.py does not use it")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(parser, args)
    print(json.dumps(convert(args.source, args.output, args.legacy_object), indent=2))


if __name__ == "__main__":
    main()
