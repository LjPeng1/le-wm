"""Explicit opt-in wrapper around the unchanged upstream train.run function."""

import argparse
import contextlib
import sys
from pathlib import Path

from .common import environment, helper_manifest, require_execute, source_manifest, write_json


@contextlib.contextmanager
def instrument(upstream, audit_dir, resume_path=None, stop_after_epoch=None, global_seed=None):
    """Attach full checkpoints and pass full-resume semantics to real Manager."""
    from omegaconf import OmegaConf
    from .resume import ReproState, checkpoint_callback

    original_trainer = upstream.pl.Trainer
    original_manager = upstream.spt.Manager
    original_data_module = upstream.spt.data.DataModule
    global_config = upstream.spt.get_config()
    original_cache_dir = global_config.cache_dir
    # Manager otherwise moves every ModelCheckpoint under its own cache tree.
    # This changes output routing only; the explicit wrapper owns its audit tree.
    global_config.cache_dir = None
    captured = {}

    class DataModule(original_data_module):
        def __init__(self, *args, **kwargs):
            captured["train_loader"] = kwargs["train"]
            super().__init__(*args, **kwargs)

    class Trainer(original_trainer):
        def __init__(self, *args, **kwargs):
            kwargs["default_root_dir"] = str(Path(audit_dir).resolve())
            kwargs["callbacks"] = [*kwargs.get("callbacks", []),
                                   checkpoint_callback(Path(audit_dir) / "checkpoints")]
            if resume_path is not None:
                kwargs["num_sanity_val_steps"] = 0
            super().__init__(*args, **kwargs)
            if self.world_size != 1:
                raise ValueError("Full-state wrapper is restricted to one training process")

    def manager(*args, **kwargs):
        if args:
            raise RuntimeError("Pinned train.py constructs Manager with keywords")
        module, trainer_instance = kwargs["module"], kwargs["trainer"]
        forward = getattr(module.forward, "__func__", module.forward)
        forward = getattr(forward, "_fn", forward)  # spt._NamedForward for partials
        cfg = getattr(forward, "keywords", {}).get("cfg")
        if cfg is None:
            raise RuntimeError("Cannot inspect original forward configuration; inspect dependency drift")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        signature = {k: v for k, v in resolved.items() if k not in ("subdir", "output_model_name", "wandb")}
        state = ReproState(captured["train_loader"], signature, audit_dir, stop_after_epoch)
        trainer_instance.callbacks.insert(0, state)
        write_json(Path(audit_dir) / "run_started.json", {
            "sources": source_manifest(), "helpers": helper_manifest(),
            "environment": environment(), "resolved_config": resolved,
            "mode": "opt_in_full_checkpoint_wrapper", "resume": str(resume_path) if resume_path else None,
            "global_seed_opt_in": global_seed, "stop_after_epoch": stop_after_epoch,
            "resume_skips_repeated_sanity_validation": resume_path is not None,
            "output_routing_patch": {"spt_cache_dir": None,
                                      "trainer_default_root_dir": str(Path(audit_dir).resolve())},
            "scheduler_horizon_max_epochs": trainer_instance.max_epochs,
        })
        if kwargs.get("ckpt_path") is not None and resume_path is None:
            raise ValueError("Upstream found an implicit checkpoint; use a clean run or explicit --resume")
        kwargs["ckpt_path"] = str(Path(resume_path).resolve()) if resume_path else None
        kwargs["weights_only"] = False
        return original_manager(**kwargs)

    upstream.spt.data.DataModule = DataModule
    upstream.pl.Trainer = Trainer
    upstream.spt.Manager = manager
    try:
        if global_seed is not None:
            upstream.pl.seed_everything(global_seed, workers=True)
        yield
    finally:
        global_config.cache_dir = original_cache_dir
        upstream.spt.data.DataModule = original_data_module
        upstream.pl.Trainer = original_trainer
        upstream.spt.Manager = original_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop at an epoch boundary while preserving configured scheduler horizon")
    parser.add_argument("--global-seed", type=int,
                        help="Explicit behavioral patch; upstream seeds only split/loader")
    parser.add_argument("--execute", action="store_true")
    args, overrides = parser.parse_known_args()
    require_execute(parser, args)
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must name an existing complete .ckpt file")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    if (args.audit_dir / "run_started.json").exists():
        parser.error("Choose a new audit directory; earlier records are preserved")
    import train as upstream
    sys.argv = ["train.py", *overrides]
    with instrument(upstream, args.audit_dir.resolve(), args.resume,
                    args.stop_after_epoch, args.global_seed):
        upstream.run()


if __name__ == "__main__":
    main()
