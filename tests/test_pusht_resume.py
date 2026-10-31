"""Real Lightning/SPT core resume identity on a tiny CPU JEPA fixture.

PUSHT_SOURCE_ISOLATION=1 bypasses only eager package __init__ files for optional
dataset/visualisation imports. Module, Manager, optimizer, scheduler and
Lightning loops are still the actual installed source. A small Lightning
DataModule returns the fixture loaders in isolation mode. This mode does not
certify the full simulator/data-worker/GPU environment.
The external logging/monitoring callback factory is disabled in isolation mode;
the real SPT manual-optimization and atomic-checkpoint patches remain enabled.
"""

import importlib
import os
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_pusht_preparation import source_symbol


def framework():
    pl = pytest.importorskip("lightning")
    spt = pytest.importorskip("stable_pretraining")
    if os.environ.get("PUSHT_SOURCE_ISOLATION") == "1":
        root = Path(spt.__file__).parent
        for name in ("callbacks", "utils", "data", "loggers"):
            qualified = "stable_pretraining." + name
            if qualified not in sys.modules:
                package = types.ModuleType(qualified)
                package.__path__ = [str(root / name)]
                sys.modules[qualified] = package
                setattr(spt, name, package)
        inspection = importlib.import_module("stable_pretraining.utils.inspection_utils")
        spt.utils.get_required_fn_parameters = inspection.get_required_fn_parameters
        class FixtureDataModule(pl.LightningDataModule):
            def __init__(self, train=None, val=None):
                super().__init__()
                self.train, self.val = train, val

            def train_dataloader(self):
                return self.train

            def val_dataloader(self):
                return self.val

            def set_pl_trainer(self, trainer):
                self.trainer = trainer

        spt.data.DataModule = FixtureDataModule
        fixture_data = types.ModuleType("stable_pretraining.data.module")
        fixture_data.DataModule = FixtureDataModule
        sys.modules["stable_pretraining.data.module"] = fixture_data
        connector = importlib.import_module("lightning.pytorch.trainer.connectors.callback_connector")
        connector._load_external_callbacks = lambda group: []
    # Force real imports now; failure is a test failure rather than a fake substitute.
    spt.Module
    spt.Manager
    if os.environ.get("PUSHT_SOURCE_ISOLATION") == "1" and os.name == "nt":
        # This POSIX-only diagnostic accesses SIGUSR1/SIGCONT on Windows.
        importlib.import_module("stable_pretraining.manager").print_signal_info = lambda *a, **k: None
    return pl, spt


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, pixels, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.linear(pixels).unsqueeze(1))


class FixtureDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {"pixels": torch.arange(16).reshape(4, 4).float() / 20 + index / 30,
                "action": torch.arange(8).reshape(4, 2).float() / 10,
                "sample_id": index}


def run_fixture(directory, stop_after_epoch=None, resume=None):
    from omegaconf import OmegaConf
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from scripts.repro_pusht.train import instrument

    pl, spt = framework()
    trace = []

    class Trace(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            trace.append({"ids": batch["sample_id"].tolist(),
                          "loss": outputs["loss"].detach().item(),
                          "lr": trainer.optimizers[0].param_groups[0]["lr"]})

    cfg = OmegaConf.create({"history_size": 3, "num_preds": 1, "seed": 3072,
        "trainer": {"max_epochs": 3}, "loss": {"sigreg": {"weight": .09}},
        "fixture": "tiny_JEPA_CPU; not production dimensions"})
    forward = source_symbol("train.py", "lejepa_forward", {"torch": torch})
    upstream = SimpleNamespace(pl=pl, spt=spt)
    with instrument(upstream, directory, resume, stop_after_epoch, global_seed=41):
        generator = torch.Generator().manual_seed(3072)
        train_set, val_set = torch.utils.data.random_split(FixtureDataset(), [16, 4], generator=generator)
        loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True,
                                           generator=generator, drop_last=True, num_workers=0)
        val = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
        data = spt.data.DataModule(train=loader, val=val)
        model = JEPA(encoder=TinyEncoder(),
            predictor=ARPredictor(num_frames=3, input_dim=4, hidden_dim=4, output_dim=4,
                                  depth=1, heads=1, dim_head=4, mlp_dim=8, dropout=.1),
            action_encoder=Embedder(input_dim=2, smoothed_dim=2, emb_dim=4),
            projector=MLP(4, 8, 4, norm_fn=torch.nn.BatchNorm1d),
            pred_proj=MLP(4, 8, 4, norm_fn=torch.nn.BatchNorm1d))
        module = spt.Module(model=model, sigreg=SIGReg(knots=17, num_proj=8),
            forward=partial(forward, cfg=cfg), optim={"model_opt": {
                "modules": "model", "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 1e-3},
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch"}})
        trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true", max_epochs=3,
            callbacks=[Trace()], logger=False, enable_progress_bar=False,
            enable_model_summary=False, gradient_clip_val=1., num_sanity_val_steps=1,
            default_root_dir=str(directory))
        manager = spt.Manager(trainer=trainer, module=module, data=data, ckpt_path=None)
        manager()
        return {"trace": trace, "state": {k: v.detach().clone() for k, v in module.state_dict().items()},
                "optimizer": trainer.optimizers[0].state_dict(),
                "scheduler": trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                "step": trainer.global_step, "estimated_steps": trainer.estimated_stepping_batches}


def assert_same(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_real_lightning_spt_epoch_resume_identity(tmp_path):
    torch.set_num_threads(1)
    continuous = run_fixture(tmp_path / "continuous")
    first = run_fixture(tmp_path / "first", stop_after_epoch=1)
    checkpoint = tmp_path / "first/checkpoints/last.ckpt"
    assert checkpoint.is_file()
    resumed = run_fixture(tmp_path / "resumed", resume=checkpoint)
    assert continuous["trace"] == first["trace"] + resumed["trace"]
    for key in ("state", "optimizer", "scheduler", "step", "estimated_steps"):
        assert_same(continuous[key], resumed[key])
    assert continuous["step"] == 12 and first["step"] == 4
    assert continuous["estimated_steps"] == first["estimated_steps"] == resumed["estimated_steps"] == 12
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert all(key in saved for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "pusht_repro"))
    assert saved["pusht_repro"]["max_epochs"] == 3


def test_resume_rejects_changed_schedule_or_weights_only(tmp_path):
    framework()
    from scripts.repro_pusht.resume import ReproState
    loader = torch.utils.data.DataLoader(FixtureDataset(), batch_size=4,
                                         generator=torch.Generator().manual_seed(3072))
    callback = ReproState(loader, {"max_epochs": 100}, tmp_path)
    trainer = SimpleNamespace(max_epochs=100)
    with pytest.raises(ValueError, match="complete"):
        callback.on_load_checkpoint(trainer, None, {"state_dict": {}})
    with pytest.raises(ValueError, match="10→100"):
        callback.on_load_checkpoint(trainer, None, {"pusht_repro": {
            "version": 1, "max_epochs": 10, "signature": {"max_epochs": 10}}})
