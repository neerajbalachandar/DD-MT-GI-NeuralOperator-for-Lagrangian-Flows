from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _phase_binned_by_case(records, phase_key, metric_key, bins=20):
    grouped = {}
    for row in records:
        phase, value = row.get(phase_key), row.get(metric_key)
        if phase is None or value is None or not np.isfinite(value):
            continue
        index = int(np.clip(np.floor(float(phase) * bins), 0, bins - 1))
        grouped.setdefault((str(row.get("case", "unknown")), index), []).append(float(value))
    by_phase = {}
    for (_, index), values in grouped.items():
        by_phase.setdefault(index, []).append(float(np.mean(values)))
    phases = np.asarray(sorted(by_phase), dtype=float)
    means = np.asarray([np.mean(by_phase[int(i)]) for i in phases])
    stds = np.asarray([np.std(by_phase[int(i)]) for i in phases])
    return (phases + 0.5) / bins, means, stds


def plot_one_step_loss_vs_time(records, output_path, bins=20):
    """Plot case-aggregated one-step particle/field relative-L2 versus phase."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    curves = (("phase", "position_relative_l2", "Particle position", "#176b87"),
              ("field_phase", "field_relative_l2", "Eulerian velocity", "#d66b35"))
    for phase_key, metric_key, label, color in curves:
        phase, mean, std = _phase_binned_by_case(records, phase_key, metric_key, bins)
        if not len(phase):
            continue
        lower = np.maximum(mean - std, 1e-8)
        axis.plot(phase, mean, marker="o", markersize=3, color=color, label=label)
        axis.fill_between(phase, lower, mean + std, color=color, alpha=0.18)
    axis.set_yscale("log")
    axis.set_xlabel("Normalized simulation phase")
    axis.set_ylabel("Relative L2 error")
    axis.set_title("One-step prediction error versus simulation time")
    axis.grid(True, which="both", alpha=0.22)
    if axis.lines:
        axis.legend(frameon=False)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_error_by_step(records, output_path):
    """Aggregate within each test simulation, then show mean +/- std across cases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_case_horizon = {}
    for row in records:
        value = row.get("position_relative_l2")
        if value is None or not np.isfinite(value):
            continue
        key = (str(row.get("case", "unknown")), int(row["horizon"]))
        by_case_horizon.setdefault(key, []).append(float(value))
    by_horizon = {}
    for (_, horizon), values in by_case_horizon.items():
        by_horizon.setdefault(horizon, []).append(float(np.mean(values)))
    horizons = np.asarray(sorted(by_horizon), dtype=int)
    if not len(horizons):
        return
    means = np.asarray([np.mean(by_horizon[int(h)]) for h in horizons])
    stds = np.asarray([np.std(by_horizon[int(h)]) for h in horizons])
    fig, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    axis.plot(horizons, means, marker="o", color="#176b87", label="Mean across test simulations")
    axis.fill_between(horizons, np.maximum(means - stds, 0.0), means + stds,
                      color="#176b87", alpha=0.2, label=" +/- 1 std across simulations")
    axis.set_xlabel("Autoregressive rollout step k")
    axis.set_ylabel("Particle position relative L2 error")
    axis.set_title("Long-horizon rollout error")
    axis.set_xticks(horizons)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_errors(one_step, rollout, output_dir):
    """Create the two requested temporal accuracy/stability plots."""
    output_dir = Path(output_dir)
    plot_one_step_loss_vs_time(one_step, output_dir / "one_step_loss_vs_time.png")
    plot_rollout_error_by_step(rollout, output_dir / "rollout_loss_vs_step.png")


def plot_rollout_snapshots(snapshots, output_path, max_snapshots=6, title=None):
    """Plot paired true/predicted x-z particle snapshots, colored by |Gamma|/sigma^3."""
    if not snapshots:
        return
    horizons = sorted(snapshots)
    if len(horizons) > max_snapshots:
        indices = np.unique(np.linspace(0, len(horizons) - 1, max_snapshots, dtype=int))
        horizons = [horizons[i] for i in indices]
    all_values = []
    for horizon in horizons:
        for key in ("truth", "prediction"):
            state = np.asarray(snapshots[horizon][key])
            sigma = np.maximum(np.abs(state[:, 6]), 1e-8)
            all_values.append(np.linalg.norm(state[:, 3:6], axis=-1) / sigma ** 3)
    vmax = max(float(np.percentile(np.concatenate(all_values), 99)), 1e-12)
    norm = plt.Normalize(vmin=0.0, vmax=vmax, clip=True)
    fig, axes = plt.subplots(len(horizons), 2, figsize=(11, 3.0 * len(horizons)),
                             squeeze=False, constrained_layout=True)
    last = None
    for row, horizon in enumerate(horizons):
        sample = snapshots[horizon]
        for column, key, title in ((0, "truth", "True"), (1, "prediction", "Predicted")):
            state = np.asarray(sample[key])
            omega_proxy = np.linalg.norm(state[:, 3:6], axis=-1) / np.maximum(np.abs(state[:, 6]), 1e-8) ** 3
            last = axes[row, column].scatter(state[:, 0], state[:, 2], c=omega_proxy,
                                              s=8, cmap="viridis", norm=norm, rasterized=True)
            time_value = sample.get("physical_time")
            when = f", t={time_value:.4g} s" if time_value is not None else ""
            axes[row, column].set_title(f"{title}, step {horizon}{when}")
            axes[row, column].set_xlabel("x [m]")
            axes[row, column].set_ylabel("z [m]")
            axes[row, column].set_aspect("equal", adjustable="datalim")
            axes[row, column].grid(alpha=0.15)
    fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.85,
                 label=r"Circulation/core-size proxy $|\Gamma|/\sigma^3$")
    fig.suptitle(title or "Autoregressive particle rollout (x-z projection)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
