"""Epoch-boundary full-state support for the opt-in upstream runner."""

import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .common import source_manifest, write_json


class ReproState(Callback):
    """Add RNG/loader state and reject a changed training schedule on resume.

    Lightning's own checkpoint still owns model, optimizer, scheduler and loops.
    This callback is restricted to epoch-boundary, single-process recovery.
    Worker-process RNG identity is not asserted by the CPU test.
    """

    def __init__(self, train_loader, signature, audit_dir, stop_after_epoch=None):
        self.train_loader = train_loader
        self.signature = signature
        self.audit_dir = Path(audit_dir)
        self.stop_after_epoch = stop_after_epoch
        self.pending_rng = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["pusht_repro"] = {
            "version": 1, "signature": self.signature,
            "max_epochs": trainer.max_epochs, "sources": source_manifest(),
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_generator": self.train_loader.generator.get_state(),
            "epoch_boundary_only": True,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get("pusht_repro")
        if not state or state.get("version") != 1:
            raise ValueError("Not a complete reproduction checkpoint (missing pusht_repro state)")
        if state["max_epochs"] != trainer.max_epochs or state["signature"] != self.signature:
            raise ValueError("Resume requires identical configuration and max_epochs; 10→100 is not a resume")
        if state["sources"] != source_manifest():
            raise ValueError("Scientific source hashes changed since the checkpoint")
        if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers") or "loops" not in checkpoint:
            raise ValueError("Optimizer, scheduler or loop state missing; weights-only is not full resume")
        self.train_loader.generator.set_state(state["train_generator"])
        self.pending_rng = state

    def on_train_start(self, trainer, pl_module):
        if self.pending_rng is None:
            return
        state = self.pending_rng
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        self.pending_rng = None

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stop_after_epoch is not None and trainer.current_epoch + 1 >= self.stop_after_epoch:
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        write_json(self.audit_dir / "fit_completed.json", {
            "global_step": trainer.global_step, "current_epoch": trainer.current_epoch,
            "max_epochs": trainer.max_epochs, "stop_after_epoch": self.stop_after_epoch,
            "estimated_stepping_batches": trainer.estimated_stepping_batches,
            "recovery_scope": "epoch_boundary_single_process; worker RNG identity unverified",
        })


def checkpoint_callback(directory):
    return ModelCheckpoint(dirpath=str(Path(directory).resolve()), filename="epoch-{epoch:03d}",
                           save_last=True, save_top_k=-1, every_n_epochs=1,
                           save_weights_only=False, save_on_train_epoch_end=True)
