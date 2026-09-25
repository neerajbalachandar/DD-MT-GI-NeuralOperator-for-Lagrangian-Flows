from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_temporal_errors(one_step, rollout, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    cases = sorted({r.get("case", "case") for r in one_step} | {r.get("case", "case") for r in rollout})
    colors = ("#176b87", "#d66b35", "#588157", "#a23b72")
    for i, case in enumerate(cases):
        color = colors[i % len(colors)]
        one = sorted((r for r in one_step if r.get("case", "case") == case), key=lambda r: r.get("phase", r["pair_id"]))
        closed = sorted((r for r in rollout if r.get("case", "case") == case), key=lambda r: r["horizon"])
        horizons = sorted({int(r["horizon"]) for r in closed})
        rollout_error = [float(np.mean([r.get("position_rmse", np.sqrt(r["position_mse"]))
                                        for r in closed if int(r["horizon"]) == horizon]))
                         for horizon in horizons]
        axes[0].plot([r.get("phase", r["pair_id"]) for r in one],
                     [r.get("position_rmse", np.sqrt(r["position_mse"])) for r in one],
                     marker=".", color=color, label=case)
        axes[1].plot(horizons, rollout_error, marker="o", color=color, label=case)
    axes[0].set(xlabel="Phase", ylabel="Position RMSE", title="One-step")
    axes[1].set(xlabel="Prediction horizon", ylabel="Position RMSE", title="Autoregressive rollout")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(output_dir / "temporal_error_comparison.png", bbox_inches="tight")
    plt.close(fig)
