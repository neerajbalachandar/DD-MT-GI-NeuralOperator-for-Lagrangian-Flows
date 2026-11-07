import json
import os
import platform
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import (grouped_field_loss, normalized_rollout_loss, uncertainty_weighted_sum,
                     normalized_physical_state_loss)
from .noise import perturb_flow_inputs, random_walk_noise, noise_for_rollout_input
from .scheduled_sampling import scheduled_sampling_probability, choose_predicted_flow_inputs, teacher_input_at
from gino.data.reconstruction import rebuild_next_batch_training
from gino.dynamics.rollout import training_rollout, pushforward_rollout
from gino.dynamics.state_transition import predict_next_state


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUDA grid_sample backward has no deterministic implementation. Enabling
    # deterministic algorithms only produces a warning for this model's path.
    torch.use_deterministic_algorithms(False)


def horizon_for_epoch(epoch, epochs, schedule):
    index = min(len(schedule) - 1, int((epoch - 1) * len(schedule) / max(epochs, 1)))
    return int(schedule[index])


def _worker_init(worker_id):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"


class Trainer:
    def __init__(self, model, latent_grid, train_dataset, val_dataset, normalization, config, run_dir, device):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.model, self.latent_grid = model, latent_grid
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.normalization, self.config = normalization, config
        self.run_dir, self.device = Path(run_dir), device
        self.run_dir.mkdir(parents=True, exist_ok=True)
        uncertainty_names = {"log_physical_state_var", "log_residual_var", "log_velocity_var",
                             "log_gradient_var", "log_vorticity_var"}
        regularized, unregularized = [], []
        for name, parameter in model.named_parameters():
            (unregularized if name in uncertainty_names else regularized).append(parameter)
        self.optimizer = torch.optim.AdamW([
            {"params": regularized, "weight_decay": float(config["weight_decay"])},
            {"params": unregularized, "weight_decay": 0.0},
        ], lr=float(config["learning_rate"]))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5) if config.get("scheduler") == "plateau" else None
        self.mixed_precision = bool(config.get("mixed_precision")) and str(device).startswith("cuda")
        self.history = []

    def _loader(self, dataset, shuffle):
        num_workers = int(self.config.get("dataloader_num_workers", 0))
        kwargs = dict(batch_size=1, shuffle=shuffle, collate_fn=dataset_collate,
                      num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["worker_init_fn"] = _worker_init
        return DataLoader(dataset, **kwargs)

    def _epoch(self, loader, train, epoch=1):
        self.model.train(train)
        totals = {k: [] for k in ("total_loss", "state_loss", "residual_loss", "field_loss", "field_velocity_loss",
                                  "field_gradient_loss", "field_vorticity_loss", "rollout_loss",
                                  "pushforward_loss", "gradient_norm", "effective_horizon",
                                  "field_supervision_fraction")}
        accum = max(int(self.config.get("gradient_accumulation_steps", 1)), 1)
        max_probability = float(self.config.get("scheduled_sampling_max_probability", 0.3))
        use_scheduled_sampling = bool(self.config.get("use_scheduled_sampling", False))
        sample_probability = scheduled_sampling_probability(
            epoch, max_probability, self.config.get("scheduled_sampling_ramp_epochs", 30)
        ) if train and use_scheduled_sampling else 0.0
        schedule = self.config.get("rollout_horizon_schedule", [1])
        if train:
            active_horizon = min(int(self.config.get("rollout_horizon_max", max(schedule))),
                                 horizon_for_epoch(epoch, self.config.get("epochs", 1), schedule))
        else:
            # Keep checkpoint selection comparable across epochs: validation always
            # scores the same longest supported chain, independent of the curriculum.
            active_horizon = max(1, int(self.config.get("validation_rollout_horizon", 1)))
        # Evaluate the autoregressive objective on validation data as well, so
        # checkpoint selection cannot silently optimize only the one-step task.
        use_rollout = bool(self.config.get("use_rollout_loss", False))
        use_pushforward = bool(train and self.config.get("use_pushforward", False))
        do_sequence = (use_rollout or use_pushforward) and active_horizon > 1
        if train and use_scheduled_sampling and sample_probability > 0.0 and not do_sequence:
            raise ValueError(
                "Scheduled sampling is active at probability "
                f"{sample_probability:.3f} but the current epoch has "
                f"active_horizon={active_horizon} (<2). Either lower "
                "training.rollout_horizon_schedule so the first epoch is >= 2, "
                "or set scheduled_sampling_ramp_epochs so probability stays 0 "
                "until the horizon is > 1."
            )
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        if train and len(loader) == 0:
            raise ValueError("Training dataset is empty; no batches are available for an epoch.")
        t_data = t_fwd = t_bwd = 0.0
        cuda_timing = torch.device(self.device).type == "cuda"
        fwd_events, bwd_events = [], []
        prev = time.perf_counter()
        for step, batch in enumerate(loader):
            now = time.perf_counter()
            t_data += now - prev
            batch["rollout_state_targets"] = batch["rollout_state_targets"][:, :active_horizon]
            batch = move_batch(batch, self.device)
            if cuda_timing:
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_start.record()
            else:
                fwd_start = time.perf_counter()
            flow_indices = [i for i, name in enumerate(batch["feature_names"]) if name in ("u_x", "u_y", "u_z") or name.startswith("gradU_")]
            use_noise = bool(train and self.config.get("use_gns_noise", False))
            noise = None
            if use_noise and (float(self.config.get("gns_initial_std", 0.0)) or float(self.config.get("gns_walk_std", 0.0))):
                noise = random_walk_noise(batch["x"][..., flow_indices].unsqueeze(0).expand(active_horizon, -1, -1, -1),
                    self.config.get("gns_initial_std", 0.0), self.config.get("gns_walk_std", 0.0))
                batch["x"], _ = perturb_flow_inputs(batch, noise_for_rollout_input(noise, 0), flow_indices)
            with torch.set_grad_enabled(train), torch.autocast(
                    device_type="cuda", enabled=self.mixed_precision, dtype=torch.bfloat16):
                prediction = predict_next_state(self.model, batch, self.latent_grid, self.normalization)
                field = prediction["field_norm"]
                residual_target = batch["delta_target"]
                residual_pred = prediction["residual_norm"]
                if residual_pred.shape != residual_target.shape:
                    raise ValueError(f"Residual prediction/target shape mismatch: {residual_pred.shape} vs {residual_target.shape}")
                # The first seven channels correct the physical baseline; the
                # final three are supervised delta-u auxiliaries and are not propagated.
                residual_term = torch.nn.functional.mse_loss(residual_pred, residual_target)
                state_term = normalized_physical_state_loss(prediction["predicted_state"], batch["next_state_phys"],
                    self.normalization.state_mean[:7], self.normalization.state_std[:7])
                has_field_target = bool(batch.get("field_available", True))
                totals["field_supervision_fraction"].append(float(has_field_target))
                if has_field_target:
                    _, field_groups = grouped_field_loss(field, batch["field_target"])
                    field_term = sum(field_groups.values())
                else:
                    field_term, field_groups = None, {}
                task_losses = {}
                if self.config.get("use_state_loss", True):
                    task_losses["physical_state"] = state_term
                    task_losses["residual"] = residual_term
                if self.config.get("use_field_loss", True) and has_field_target:
                    task_losses.update({f"field_{key}": value for key, value in field_groups.items()})
                variances = {"physical_state": self.model.log_physical_state_var,
                             "residual": self.model.log_residual_var,
                             "field_velocity": self.model.log_velocity_var,
                             "field_gradient": self.model.log_gradient_var,
                             "field_vorticity": self.model.log_vorticity_var}
                if self.config.get("use_homoscedastic_weighting", True):
                    one_step = uncertainty_weighted_sum(task_losses, variances)
                else:
                    one_step = sum(task_losses.values())
                rollout_term = pushforward_term = None
                if do_sequence:
                    horizon = min(active_horizon, batch["rollout_state_targets"].shape[1])
                    totals["effective_horizon"].append(float(horizon))
                    if horizon > 1:
                        target = batch["rollout_state_targets"][:, :horizon]
                        state_mean = self.normalization.state_mean if self.normalization.state_mean is not None else np.zeros(7, dtype=np.float32)
                        state_std = self.normalization.state_std if self.normalization.state_std is not None else np.ones(7, dtype=np.float32)
                        def make_rebuild(differentiable):
                            transition_index = 0
                            def rebuild(old, predicted, field_phys):
                                nonlocal transition_index
                                nxt = rebuild_next_batch_training(old, predicted, field_phys, self.normalization,
                                                                 pushforward=not differentiable)
                                if train and self.config.get("use_scheduled_sampling", False) and flow_indices:
                                    teacher = teacher_input_at(old.get("rollout_teacher_inputs"), transition_index)
                                    velocity = torch.stack([field_phys[..., 0], field_phys[..., 1], field_phys[..., 2]], dim=-1)
                                    gradient = field_phys[..., 3:12].reshape(*field_phys.shape[:-1], 3, 3)
                                    nxt["x"], _ = choose_predicted_flow_inputs(nxt, velocity, gradient,
                                        sample_probability, teacher_x=teacher)
                                if noise is not None:
                                    nxt["x"], _ = perturb_flow_inputs(
                                        nxt, noise_for_rollout_input(noise, transition_index + 1), flow_indices)
                                transition_index += 1
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
            if cuda_timing:
                fwd_end = torch.cuda.Event(enable_timing=True)
                fwd_end.record()
                fwd_events.append((fwd_start, fwd_end))
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_start.record()
            else:
                t_fwd += time.perf_counter() - fwd_start
                bwd_start = time.perf_counter()
            if train:
                group_start = (step // accum) * accum
                group_size = min(accum, len(loader) - group_start)
                (total / group_size).backward()
                if (step + 1) % accum == 0 or step + 1 == len(loader):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip_norm", 1.0)))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = None
            else:
                grad_norm = None
            totals["total_loss"].append(total.detach())
            totals["state_loss"].append(state_term.detach())
            totals["residual_loss"].append(residual_term.detach())
            if field_term is not None:
                totals["field_loss"].append(field_term.detach())
                for name, metric in (("velocity", "field_velocity_loss"),
                                     ("gradient", "field_gradient_loss"),
                                     ("vorticity", "field_vorticity_loss")):
                    totals[metric].append(field_groups[name].detach())
            totals["rollout_loss"].append(rollout_term.detach() if rollout_term is not None else total.detach().new_zeros(()))
            totals["pushforward_loss"].append(pushforward_term.detach() if pushforward_term is not None else total.detach().new_zeros(()))
            if grad_norm is not None:
                totals["gradient_norm"].append(grad_norm.detach())
            if cuda_timing and train:
                bwd_end = torch.cuda.Event(enable_timing=True)
                bwd_end.record()
                bwd_events.append((bwd_start, bwd_end))
            elif not cuda_timing:
                t_bwd += time.perf_counter() - bwd_start
            prev = time.perf_counter()
        if cuda_timing and (fwd_events or bwd_events):
            torch.cuda.synchronize(self.device)
            t_fwd = sum(start.elapsed_time(end) for start, end in fwd_events) / 1000.0
            t_bwd = sum(start.elapsed_time(end) for start, end in bwd_events) / 1000.0
        metric_means = []
        for values in totals.values():
            if not values:
                metric_means.append(torch.tensor(float("nan"), device=self.device))
            else:
                tensors = [value if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)
                           for value in values]
                metric_means.append(torch.stack(tensors).mean())
        values = torch.stack(metric_means).detach().cpu().tolist()
        metrics = dict(zip(totals, map(float, values)))
        metrics.update(data_time=t_data, fwd_time=t_fwd, bwd_time=t_bwd)
        return metrics

    def fit(self, resolved_config, metadata):
        set_deterministic_seed(int(resolved_config["seed"]))
        print(f"[loader] num_workers={int(self.config.get('dataloader_num_workers', 0))}")
        (self.run_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, default=str))
        env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(self.device)}
        (self.run_dir / "environment.json").write_text(json.dumps(env, indent=2))
        best, stale = float("inf"), 0
        previous_val_metrics = None
        validation_interval = max(int(self.config.get("validation_every_n_epochs", 1)), 1)
        train_loader, val_loader = self._loader(self.train_dataset, True), self._loader(self.val_dataset, False)
        total_epochs = int(self.config["epochs"])
        for epoch in range(1, total_epochs + 1):
            train_metrics = self._epoch(train_loader, True, epoch)
            train_loss = train_metrics["total_loss"]
            near_best = (np.isfinite(best) and np.isfinite(train_loss)
                         and abs(best - train_loss) <= 0.05 * max(abs(train_loss), 1e-8))
            should_validate = (previous_val_metrics is None or epoch % validation_interval == 0
                               or epoch == total_epochs or near_best)
            did_validate = bool(len(val_loader.dataset) and should_validate)
            if did_validate:
                val_metrics = self._epoch(val_loader, False, epoch)
                previous_val_metrics = val_metrics
            elif len(val_loader.dataset):
                val_metrics = previous_val_metrics
                next_validation = min(((epoch // validation_interval) + 1) * validation_interval,
                                      total_epochs)
                print(f"epoch {epoch:03d} | val skipped (will validate at epoch {next_validation:03d})")
            else:
                val_metrics = train_metrics
            train_loss, val_loss = train_metrics["total_loss"], val_metrics["total_loss"]
            schedule = self.config.get("rollout_horizon_schedule", [1])
            active_horizon = min(int(self.config.get("rollout_horizon_max", max(schedule))),
                                 horizon_for_epoch(epoch, self.config["epochs"], schedule))
            max_probability = float(self.config.get("scheduled_sampling_max_probability", 0.3))
            if self.scheduler and did_validate:
                self.scheduler.step(val_loss)
            uncertainty_names = ("physical_state", "residual", "velocity", "gradient", "vorticity")
            uncertainty_values = {}
            for name in uncertainty_names:
                parameter = getattr(self.model, f"log_{name}_var")
                value = float(parameter.detach().cpu())
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite learned log variance for {name}: {value}")
                if abs(value) > float(self.config.get("uncertainty_log_variance_warning", 10.0)):
                    print(f"[uncertainty warning] log variance for {name}={value:.4g}; effective precision may have collapsed")
                uncertainty_values[f"log_variance_{name}"] = value
                uncertainty_values[f"precision_{name}"] = float(np.exp(np.clip(-value, -80.0, 80.0)))
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()},
                   **uncertainty_values,
                   "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                   "train_effective_rollout_horizon": train_metrics["effective_horizon"],
                   "rollout_horizon": active_horizon if self.config.get("use_rollout_loss") or self.config.get("use_pushforward") else 1,
                   "validation_rollout_horizon": int(self.config.get("validation_rollout_horizon", 1)),
                   "scheduled_sampling_probability": scheduled_sampling_probability(
                       epoch, max_probability, self.config.get("scheduled_sampling_ramp_epochs", 30)
                   ) if self.config.get("use_scheduled_sampling") else 0.0,
                   "noise_initial_std": float(self.config.get("gns_initial_std", 0.0)) if self.config.get("use_gns_noise") else 0.0,
                   "noise_walk_std": float(self.config.get("gns_walk_std", 0.0)) if self.config.get("use_gns_noise") else 0.0,
                   "checkpoint_selection_objective": (
                       "fixed validation horizon total objective: uncertainty-weighted physical-state and residual "
                       "MSE plus available Eulerian group MSEs"
                       + (f" plus rollout loss (weight={float(self.config.get('rollout_weight', 0.0))})"
                          if self.config.get("use_rollout_loss", False) and int(self.config.get("validation_rollout_horizon", 1)) > 1
                          else "")
                       + "; pushforward is training-only and does not affect checkpoint selection"
                   ),
                   "rollout_weight": float(self.config.get("rollout_weight", 0.0)),
                   "pushforward_weight": float(self.config.get("pushforward_weight", 0.0))}
            self.history.append(row)
            print(
                f"epoch {epoch:03d} | train {train_loss:.6g} | val {val_loss:.6g} | "
                f"state {train_metrics['state_loss']:.6g} | field {train_metrics['field_loss']:.6g} | "
                f"field groups (u/grad/omega) {train_metrics['field_velocity_loss']:.4g}/"
                f"{train_metrics['field_gradient_loss']:.4g}/{train_metrics['field_vorticity_loss']:.4g} | "
                f"field coverage {train_metrics['field_supervision_fraction']:.1%} | "
                f"rollout {train_metrics['rollout_loss']:.6g} | pushforward {train_metrics['pushforward_loss']:.6g} | "
                f"H {row['rollout_horizon']} (effective {train_metrics['effective_horizon']:.2f}) | "
                f"sampling {row['scheduled_sampling_probability']:.3f} | "
                f"noise ({row['noise_initial_std']:.3g}, {row['noise_walk_std']:.3g}) | "
                f"grad {train_metrics['gradient_norm']:.6g} | lr {row['learning_rate']:.3g} | "
                f"data {train_metrics['data_time']:.1f}s fwd {train_metrics['fwd_time']:.1f}s "
                f"bwd {train_metrics['bwd_time']:.1f}s"
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
                if did_validate:
                    stale += 1
            if stale >= int(self.config.get("early_stopping_patience", 20)):
                break
        return self.history


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def dataset_collate(items):
    from gino.data.dataset import collate_one
    return collate_one(items)
