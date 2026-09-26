import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import combined_loss, field_loss, state_loss, normalized_rollout_loss
from .noise import perturb_flow_inputs
from .scheduled_sampling import scheduled_sampling_probability
from gino.data.reconstruction import rebuild_next_batch_training
from gino.dynamics.rollout import training_rollout, pushforward_rollout


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

    def _epoch(self, loader, train, epoch=1):
        self.model.train(train)
        totals = {k: [] for k in ("total_loss", "state_loss", "field_loss", "rollout_loss", "pushforward_loss", "gradient_norm", "effective_horizon")}
        accum = max(int(self.config.get("gradient_accumulation_steps", 1)), 1)
        max_probability = float(self.config.get("scheduled_sampling_max_probability", 0.3))
        sample_probability = scheduled_sampling_probability(
            epoch, max_probability, self.config.get("scheduled_sampling_ramp_epochs", 30)
        ) if train and self.config.get("use_scheduled_sampling", False) else 0.0
        schedule = self.config.get("rollout_horizon_schedule", [1])
        active_horizon = min(int(self.config.get("rollout_horizon_max", max(schedule))), horizon_for_epoch(epoch, self.config.get("epochs", 1), schedule))
        use_rollout = bool(train and self.config.get("use_rollout_loss", False))
        use_pushforward = bool(train and self.config.get("use_pushforward", False))
        do_sequence = (use_rollout or use_pushforward) and active_horizon > 1
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch["rollout_state_targets"] = batch["rollout_state_targets"][:, :active_horizon]
            batch = move_batch(batch, self.device)
            flow_indices = [i for i, name in enumerate(batch["feature_names"]) if name in ("u_x", "u_y", "u_z") or name.startswith("gradU_")]
            use_noise = bool(train and self.config.get("use_gns_noise", False))
            noise = None
            if use_noise and (float(self.config.get("gns_initial_std", 0.0)) or float(self.config.get("gns_walk_std", 0.0))):
                batch["x"], noise = perturb_flow_inputs(batch, noise, flow_indices,
                    self.config.get("gns_initial_std", 0.0), self.config.get("gns_walk_std", 0.0))
            with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=self.scaler.is_enabled()):
                if sample_probability > 0 and flow_indices:
                    with torch.no_grad():
                        encoded = self.model.encode_process(batch["input_geom"], self.latent_grid,
                                                            batch["x"], batch["global_params"])
                        flow_norm = self.model.decode_field(encoded, batch["particle_queries"])
                        flow_phys = self.normalization.denormalize_field(flow_norm)
                        if bool(torch.rand((), device=self.device) < sample_probability):
                            means = torch.as_tensor(batch["input_mean"], dtype=batch["x"].dtype, device=self.device).reshape(-1)
                            stds = torch.as_tensor(batch["input_std"], dtype=batch["x"].dtype, device=self.device).reshape(-1).clamp_min(1e-8)
                            batch["x"] = batch["x"].clone()
                            grad_names = [f"gradU_{i}{j}" for i in "xyz" for j in "xyz"]
                            for j in flow_indices:
                                name = batch["feature_names"][j]
                                if name in ("u_x", "u_y", "u_z"):
                                    value = flow_phys[..., ("u_x", "u_y", "u_z").index(name)]
                                else:
                                    value = flow_phys[..., 3 + grad_names.index(name)]
                                batch["x"][..., j] = (value - means[j]) / stds[j]
                pred, field = self.model(batch["input_geom"], self.latent_grid, batch["output_queries"], batch["x"], batch["global_params"], batch_dict=batch)
                state_term = state_loss(pred[..., :batch["delta_target"].shape[-1]], batch["delta_target"])
                field_term = field_loss(field, batch["field_target"], bool(self.config.get("use_relative_l2_field_loss", False)))
                one_step = combined_loss(state_term if self.config.get("use_state_loss", True) else None,
                                      field_term if self.config.get("use_field_loss", True) else None,
                                      weights={"state": self.config.get("state_loss_weight", 1.0), "field": self.config.get("field_loss_weight", 1.0)},
                                      log_state_var=self.model.log_delta_var, log_field_var=self.model.log_field_var,
                                      homoscedastic=bool(self.config.get("use_homoscedastic_weighting", True)))
                rollout_term = pushforward_term = None
                if do_sequence:
                    horizon = min(active_horizon, batch["rollout_state_targets"].shape[1])
                    totals["effective_horizon"].append(float(horizon))
                    if horizon > 1:
                        target = batch["rollout_state_targets"][:, :horizon]
                        state_mean = self.normalization.state_mean if self.normalization.state_mean is not None else np.zeros(7, dtype=np.float32)
                        state_std = self.normalization.state_std if self.normalization.state_std is not None else np.ones(7, dtype=np.float32)
                        def make_rebuild(differentiable):
                            walk_noise = noise
                            def rebuild(old, predicted, field_phys):
                                nonlocal walk_noise
                                nxt = rebuild_next_batch_training(old, predicted, field_phys, self.normalization,
                                                                 pushforward=not differentiable)
                                if sample_probability > 0 and flow_indices:
                                    if bool(torch.rand((), device=self.device) >= sample_probability):
                                        nxt["x"][..., flow_indices] = old["x"][..., flow_indices]
                                if use_noise and (float(self.config.get("gns_initial_std", 0.0)) or float(self.config.get("gns_walk_std", 0.0))):
                                    nxt["x"], walk_noise = perturb_flow_inputs(nxt, walk_noise, flow_indices,
                                        self.config.get("gns_initial_std", 0.0), self.config.get("gns_walk_std", 0.0))
                                return nxt
                            return rebuild
                        if use_rollout:
                            rollout_pred, _ = training_rollout(self.model, batch, self.latent_grid,
                                self.normalization, horizon, make_rebuild(True))
                            n = min(rollout_pred.shape[2], target.shape[2])
                            rollout_term = normalized_rollout_loss(rollout_pred[:, :, :n], target[:, :, :n], state_mean, state_std)
                        if use_pushforward:
                            pushforward_pred, _ = pushforward_rollout(self.model, batch, self.latent_grid,
                                self.normalization, horizon, make_rebuild(False))
                            n = min(pushforward_pred.shape[2], target.shape[2])
                            pushforward_term = normalized_rollout_loss(pushforward_pred[:, :, :n], target[:, :, :n], state_mean, state_std)
                total = one_step
                if not do_sequence:
                    totals["effective_horizon"].append(1.0)
                if rollout_term is not None:
                    total = total + float(self.config.get("rollout_weight", 0.0)) * rollout_term
                if pushforward_term is not None:
                    total = total + float(self.config.get("pushforward_weight", 1.0)) * pushforward_term
            if train:
                self.scaler.scale(total / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip_norm", 1.0)))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = None
            else:
                grad_norm = None
            totals["total_loss"].append(float(total.detach().cpu()))
            totals["state_loss"].append(float(state_term.detach().cpu()))
            totals["field_loss"].append(float(field_term.detach().cpu()))
            totals["rollout_loss"].append(float(rollout_term.detach().cpu()) if rollout_term is not None else 0.0)
            totals["pushforward_loss"].append(float(pushforward_term.detach().cpu()) if pushforward_term is not None else 0.0)
            if grad_norm is not None:
                totals["gradient_norm"].append(float(grad_norm.detach().cpu()))
        if train and len(loader) % accum:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip_norm", 1.0)))
            totals["gradient_norm"].append(float(grad_norm.detach().cpu()))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return {key: float(np.mean(values)) if values else float("nan") for key, values in totals.items()}

    def fit(self, resolved_config, metadata):
        set_deterministic_seed(int(resolved_config["seed"]))
        (self.run_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, default=str))
        env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(self.device)}
        (self.run_dir / "environment.json").write_text(json.dumps(env, indent=2))
        best, stale = float("inf"), 0
        train_loader, val_loader = self._loader(self.train_dataset, True), self._loader(self.val_dataset, False)
        for epoch in range(1, int(self.config["epochs"]) + 1):
            train_metrics = self._epoch(train_loader, True, epoch)
            val_metrics = self._epoch(val_loader, False, epoch) if len(val_loader.dataset) else train_metrics
            train_loss, val_loss = train_metrics["total_loss"], val_metrics["total_loss"]
            schedule = self.config.get("rollout_horizon_schedule", [1])
            active_horizon = min(int(self.config.get("rollout_horizon_max", max(schedule))),
                                 horizon_for_epoch(epoch, self.config["epochs"], schedule))
            max_probability = float(self.config.get("scheduled_sampling_max_probability", 0.3))
            if self.scheduler:
                self.scheduler.step(val_loss)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()},
                   "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                   "train_effective_rollout_horizon": train_metrics["effective_horizon"],
                   "rollout_horizon": active_horizon if self.config.get("use_rollout_loss") or self.config.get("use_pushforward") else 1,
                   "scheduled_sampling_probability": scheduled_sampling_probability(
                       epoch, max_probability, self.config.get("scheduled_sampling_ramp_epochs", 30)
                   ) if self.config.get("use_scheduled_sampling") else 0.0,
                   "noise_initial_std": float(self.config.get("gns_initial_std", 0.0)) if self.config.get("use_gns_noise") else 0.0,
                   "noise_walk_std": float(self.config.get("gns_walk_std", 0.0)) if self.config.get("use_gns_noise") else 0.0,
                   "log_state_variance": float(self.model.log_delta_var.detach().cpu()),
                   "log_field_variance": float(self.model.log_field_var.detach().cpu()),
                   "state_task_weight": float(torch.exp(-self.model.log_delta_var.detach()).cpu()),
                   "field_task_weight": float(torch.exp(-self.model.log_field_var.detach()).cpu())}
            self.history.append(row)
            print(
                f"epoch {epoch:03d} | train {train_loss:.6g} | val {val_loss:.6g} | "
                f"state {train_metrics['state_loss']:.6g} | field {train_metrics['field_loss']:.6g} | "
                f"rollout {train_metrics['rollout_loss']:.6g} | pushforward {train_metrics['pushforward_loss']:.6g} | "
                f"H {row['rollout_horizon']} | sampling {row['scheduled_sampling_probability']:.3f} | "
                f"noise ({row['noise_initial_std']:.3g}, {row['noise_walk_std']:.3g}) | "
                f"grad {train_metrics['gradient_norm']:.6g} | lr {row['learning_rate']:.3g}"
            )
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            if val_loss < best:
                best, stale = val_loss, 0
                norm = metadata["normalization"]
                torch.save({"checkpoint_tag": "mt_gino", "architecture_version": 1, "config": resolved_config,
                            "model_state_dict": self.model.state_dict(), "normalization": metadata["normalization"],
                            "input_mean": norm["input_mean"], "input_std": norm["input_std"],
                            "target_mean": norm["residual_mean"], "target_std": norm["residual_std"],
                            "field_mean": norm["field_mean"], "field_std": norm["field_std"],
                            "coord_min": norm["coord_min"], "coord_span": norm["coord_span"],
                            "state_mean": norm["state_mean"], "state_std": norm["state_std"],
                            "feature_names": metadata["feature_names"], "target_names": metadata["target_names"],
                            "field_target_names": metadata["field_target_names"],
                            "global_condition_channels": metadata["global_condition_channels"],
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
