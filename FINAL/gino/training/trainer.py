import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import combined_loss, field_loss, state_loss


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def horizon_for_epoch(epoch, epochs, schedule):
    index = min(len(schedule) - 1, int((epoch - 1) * len(schedule) / max(epochs, 1)))
    return int(schedule[index])


class Trainer:
    def __init__(self, model, latent_grid, train_dataset, val_dataset, normalization, config, run_dir, device):
        self.model, self.latent_grid = model, latent_grid
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.normalization, self.config = normalization, config
        self.run_dir, self.device = Path(run_dir), device
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5) if config.get("scheduler") == "plateau" else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("mixed_precision")) and str(device).startswith("cuda"))
        self.history = []

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=dataset_collate)

    def _epoch(self, loader, train):
        self.model.train(train)
        losses = []
        accum = max(int(self.config.get("gradient_accumulation_steps", 1)), 1)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = move_batch(batch, self.device)
            with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=self.scaler.is_enabled()):
                pred, field = self.model(batch["input_geom"], self.latent_grid, batch["output_queries"], batch["x"], batch["global_params"], batch_dict=batch)
                state_term = state_loss(pred[..., :batch["delta_target"].shape[-1]], batch["delta_target"])
                field_term = field_loss(field, batch["field_target"], bool(self.config.get("use_relative_l2_field_loss", False)))
                total = combined_loss(state_term if self.config.get("use_state_loss", True) else None,
                                      field_term if self.config.get("use_field_loss", True) else None,
                                      weights={"state": self.config.get("state_loss_weight", 1.0), "field": self.config.get("field_loss_weight", 1.0)},
                                      log_state_var=self.model.log_delta_var, log_field_var=self.model.log_field_var,
                                      homoscedastic=bool(self.config.get("use_homoscedastic_weighting", True)))
            if train:
                self.scaler.scale(total / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip_norm", 1.0)))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            losses.append(float(total.detach().cpu()))
        if train and len(loader) % accum:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip_norm", 1.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self, resolved_config, metadata):
        set_deterministic_seed(int(resolved_config["seed"]))
        (self.run_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, default=str))
        env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(self.device)}
        (self.run_dir / "environment.json").write_text(json.dumps(env, indent=2))
        best, stale = float("inf"), 0
        train_loader, val_loader = self._loader(self.train_dataset, True), self._loader(self.val_dataset, False)
        for epoch in range(1, int(self.config["epochs"]) + 1):
            train_loss = self._epoch(train_loader, True)
            val_loss = self._epoch(val_loader, False) if len(val_loader.dataset) else train_loss
            if self.scheduler:
                self.scheduler.step(val_loss)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "rollout_horizon": horizon_for_epoch(epoch, self.config["epochs"], self.config.get("rollout_horizon_schedule", [1]))}
            self.history.append(row)
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            if val_loss < best:
                best, stale = val_loss, 0
                torch.save({"checkpoint_tag": "mt_gino", "architecture_version": 1, "config": resolved_config,
                            "model_state_dict": self.model.state_dict(), "normalization": metadata["normalization"],
                            "dataset_path": metadata["dataset_path"], "split": metadata["split"], "best_score": best,
                            "mechanisms": {k: self.config.get(k) for k in ("use_rollout_loss", "use_scheduled_sampling", "use_gns_noise", "use_pushforward", "use_homoscedastic_weighting")},
                            "history": self.history}, self.run_dir / "best_model.pt")
            else:
                stale += 1
            if stale >= int(self.config.get("early_stopping_patience", 20)):
                break
        return self.history


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def dataset_collate(items):
    from gino.data.dataset import collate_one
    return collate_one(items)
