"""Explicit opt-in wrapper around the unchanged upstream train.run function."""

import argparse
import contextlib
import sys
from pathlib import Path

from .common import environment, helper_manifest, require_execute, source_manifest, write_json
from .identity import checked_data_report, code_identity, lance_table_path, reserve_training_paths, split_identity, stage_resume


@contextlib.contextmanager
def instrument(upstream, audit_dir, resume_path=None, stop_after_epoch=None, global_seed=None,
               data_identity=None, protect_outputs=True, trace_inputs=False):
    """Attach full checkpoints and pass full-resume semantics to real Manager."""
    from omegaconf import OmegaConf
    from .resume import ReproState, checkpoint_callback

    original_trainer = upstream.pl.Trainer
    original_manager = upstream.spt.Manager
    original_data_module = upstream.spt.data.DataModule
    original_module = upstream.spt.Module
    if protect_outputs and data_identity is None:
        raise ValueError("A reverified data identity is required for production execution")
    audit_dir = Path(audit_dir).resolve()
    execution = {"schema_version": 1, "code": code_identity(), "data": data_identity,
                 "global_seed": global_seed, "trace_inputs": trace_inputs}
    global_config = upstream.spt.get_config()
    original_cache_dir = global_config.cache_dir
    # Manager otherwise moves every ModelCheckpoint under its own cache tree.
    # Legacy mode also disables automatic cache/requeue discovery. Resume is explicit.
    global_config.cache_dir = None
    captured = {}

    class DataModule(original_data_module):
        def __init__(self, *args, **kwargs):
            captured["train_loader"] = kwargs["train"]
            captured["val_loader"] = kwargs["val"]
            super().__init__(*args, **kwargs)

    def module(*args, **kwargs):
        # Original train.py creates Module BEFORE it writes config.yaml or exports.
        cfg = getattr(kwargs.get("forward"), "keywords", {}).get("cfg")
        if cfg is None:
            raise RuntimeError("Cannot protect output paths without the original forward cfg")
        execution["split"] = split_identity(captured["train_loader"], captured["val_loader"])
        if protect_outputs:
            dataset = captured["train_loader"].dataset.dataset
            if lance_table_path(dataset) != Path(data_identity["roots"]["lance"]).resolve():
                raise ValueError("Actual training dataset differs from verified Lance root")
            captured["paths"] = reserve_training_paths(audit_dir,
                upstream.swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
                cfg.subdir, cfg.output_model_name)
        else:
            # Explicitly test-only: tiny fixture has no upstream config/export callback.
            audit_dir.mkdir(parents=True, exist_ok=False)
        write_json(audit_dir / "execution_identity.json", execution)
        captured["resume_input"] = stage_resume(resume_path, audit_dir) if resume_path else None
        return original_module(*args, **kwargs)

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
        state = ReproState(captured["train_loader"], signature, audit_dir, stop_after_epoch,
                           execution_identity=execution, trace_inputs=trace_inputs)
        trainer_instance.callbacks.insert(0, state)
        write_json(Path(audit_dir) / "run_started.json", {
            "sources": source_manifest(), "helpers": helper_manifest(),
            "environment": environment(), "resolved_config": resolved,
            "mode": "opt_in_full_checkpoint_wrapper", "resume": str(resume_path) if resume_path else None,
            "global_seed_opt_in": global_seed, "stop_after_epoch": stop_after_epoch,
            "resume_skips_repeated_sanity_validation": resume_path is not None,
            "output_routing_patch": {"spt_cache_dir": None,
                                      "trainer_default_root_dir": str(audit_dir),
                                      "automatic_requeue_discovery": False,
                                      "claimed_paths": captured.get("paths")},
            "scheduler_horizon_max_epochs": trainer_instance.max_epochs,
        })
        if kwargs.get("ckpt_path") is not None and resume_path is None:
            raise ValueError("Upstream found an implicit checkpoint; use a clean run or explicit --resume")
        kwargs["ckpt_path"] = str(captured["resume_input"]) if resume_path else None
        kwargs["weights_only"] = False
        return original_manager(**kwargs)

    upstream.spt.data.DataModule = DataModule
    upstream.spt.Module = module
    upstream.pl.Trainer = Trainer
    upstream.spt.Manager = manager
    try:
        if global_seed is not None:
            upstream.pl.seed_everything(global_seed, workers=True)
        yield
    finally:
        global_config.cache_dir = original_cache_dir
        upstream.spt.data.DataModule = original_data_module
        upstream.spt.Module = original_module
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
    parser.add_argument("--data-verification", type=Path,
                        help="Passed verify_data report; all input files are rehashed before training")
    parser.add_argument("--trace-inputs", action="store_true",
                        help="Hash full batch tensor bytes for bounded recovery qualification; incurs CPU transfer")
    parser.add_argument("--execute", action="store_true")
    args, overrides = parser.parse_known_args()
    require_execute(parser, args)
    if args.data_verification is None:
        parser.error("--data-verification is required; prepare and verify real inputs first")
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must name an existing complete .ckpt file")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    if args.audit_dir.exists():
        parser.error("Choose a new audit directory; earlier records are preserved")
    import train as upstream
    data_identity = checked_data_report(args.data_verification)
    sys.argv = ["train.py", *overrides]
    try:
        # Hydra resolves a relative @hydra.main(config_path="./config/train")
        # as a package path unless the task function looks like __main__; when
        # train.py is imported, run.__module__ == "train" and the config module
        # cannot be found. Present it as a __main__ function so Hydra resolves
        # config_path relative to train.py's file instead.
        upstream.run.__module__ = "__main__"
        if hasattr(upstream.run, "__wrapped__"):
            upstream.run.__wrapped__.__module__ = "__main__"
        with instrument(upstream, args.audit_dir.resolve(), args.resume,
                        args.stop_after_epoch, args.global_seed, data_identity=data_identity,
                        trace_inputs=args.trace_inputs):
            upstream.run()
    except BaseException as error:
        if args.audit_dir.is_dir() and not (args.audit_dir / "failed.json").exists():
            write_json(args.audit_dir / "failed.json", {"phase": "training_entry_failed",
                       "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
