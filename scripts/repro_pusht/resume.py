"""Epoch-boundary full-state support for the opt-in upstream runner."""

import random
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .common import json_value, sha256, source_manifest, write_json


class ReproState(Callback):
    """Add RNG/loader state and reject a changed training schedule on resume.

    Lightning's own checkpoint still owns model, optimizer, scheduler and loops.
    This callback is restricted to epoch-boundary, single-process recovery.
    Worker-process RNG identity is not asserted by the CPU test.
    """

    def __init__(self, train_loader, signature, audit_dir, stop_after_epoch=None,
                 execution_identity=None, trace_inputs=False):
        self.train_loader = train_loader
        self.signature = signature
        self.audit_dir = Path(audit_dir)
        self.stop_after_epoch = stop_after_epoch
        self.pending_rng = None
        self.execution_identity = execution_identity
        self.trace = None
        self.before_lrs = None
        self.trace_inputs = trace_inputs
        self.input_digest = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["pusht_repro"] = {
            "version": 2, "signature": self.signature,
            "execution_identity": self.execution_identity,
            "max_epochs": trainer.max_epochs, "sources": source_manifest(),
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_generator": self.train_loader.generator.get_state(),
            "epoch_boundary_only": True,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get("pusht_repro")
        if not state or state.get("version") != 2:
            raise ValueError("Not a complete reproduction checkpoint (missing pusht_repro state)")
        if state["max_epochs"] != trainer.max_epochs or state["signature"] != self.signature:
            raise ValueError("Resume requires identical configuration and max_epochs; 10→100 is not a resume")
        if state["sources"] != source_manifest():
            raise ValueError("Scientific source hashes changed since the checkpoint")
        if state.get("execution_identity") != self.execution_identity:
            raise ValueError("Execution identity changed (data/split/helper/dependency/seed)")
        if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers") or "loops" not in checkpoint:
            raise ValueError("Optimizer, scheduler or loop state missing; weights-only is not full resume")
        self.train_loader.generator.set_state(state["train_generator"])
        self.pending_rng = state

    def on_train_start(self, trainer, pl_module):
        if self.pending_rng is not None:
            state = self.pending_rng
            random.setstate(state["rng_python"])
            np.random.set_state(state["rng_numpy"])
            torch.set_rng_state(state["rng_torch"])
            if state["rng_cuda"] is not None:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            self.pending_rng = None
        self.trace = (self.audit_dir / "step_trace.jsonl").open("x", encoding="utf-8")
        schedulers = []
        for item in trainer.lr_scheduler_configs:
            scheduler = item.scheduler
            schedulers.append({"class": type(scheduler).__qualname__,
                               "warmup_steps": getattr(scheduler, "warmup_steps", None),
                               "max_steps": getattr(scheduler, "max_steps", None)})
        write_json(self.audit_dir / "scheduler_started.json", {
            "estimated_stepping_batches": trainer.estimated_stepping_batches,
            "global_step": trainer.global_step, "max_epochs": trainer.max_epochs,
            "schedulers": schedulers})

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.before_lrs = [[group["lr"] for group in optimizer.param_groups]
                           for optimizer in trainer.optimizers]
        self.input_digest = None
        if self.trace_inputs:
            checksum = hashlib.sha256()
            for key in sorted(batch):
                value = batch[key]
                if not torch.is_tensor(value):
                    raise ValueError("Input tracing requires the pinned tensor-only batch dictionary")
                header = json.dumps([key, str(value.dtype), list(value.shape)]).encode()
                checksum.update(len(header).to_bytes(8, "little"))
                checksum.update(header)
                raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
                checksum.update(raw.numpy().tobytes())
            self.input_digest = checksum.hexdigest()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        record = {"epoch": trainer.current_epoch, "batch_idx": batch_idx,
                  "global_step": trainer.global_step, "lr_before": self.before_lrs,
                  "input_sha256_before_forward": self.input_digest,
                  "lr_after": [[g["lr"] for g in opt.param_groups] for opt in trainer.optimizers],
                  "losses": {k: json_value(v) for k, v in (outputs or {}).items() if "loss" in k},
                  "sample_ids": json_value(batch.get("sample_id")),
                  "sample_ids_available": "sample_id" in batch}
        self.trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.trace.flush()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stop_after_epoch is not None and trainer.current_epoch + 1 >= self.stop_after_epoch:
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        if self.trace is not None:
            self.trace.close()
        write_json(self.audit_dir / "fit_completed.json", {
            "global_step": trainer.global_step, "current_epoch": trainer.current_epoch,
            "max_epochs": trainer.max_epochs, "stop_after_epoch": self.stop_after_epoch,
            "estimated_stepping_batches": trainer.estimated_stepping_batches,
            "step_trace_sha256": sha256(self.audit_dir / "step_trace.jsonl"),
            "recovery_scope": "epoch_boundary_single_process; worker RNG identity unverified",
        })

    def on_exception(self, trainer, pl_module, exception):
        if self.trace is not None:
            self.trace.close()


def checkpoint_callback(directory):
    return ModelCheckpoint(dirpath=str(Path(directory).resolve()), filename="epoch-{epoch:03d}",
                           save_last=True, save_top_k=-1, every_n_epochs=1,
                           save_weights_only=False, save_on_train_epoch_end=True,
                           enable_version_counter=False)
