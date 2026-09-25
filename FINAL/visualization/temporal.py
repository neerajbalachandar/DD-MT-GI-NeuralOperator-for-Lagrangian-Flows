from pathlib import Path
import matplotlib.pyplot as plt


def plot_temporal_errors(one_step, rollout, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([r["pair_id"] for r in one_step], [r["mse"] for r in one_step], marker=".")
    ax.set(xlabel="Pair", ylabel="State MSE", title="One-step error")
    fig.savefig(output_dir / "one_step_error.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r["horizon"] for r in rollout], [r["mse"] for r in rollout], marker=".")
    ax.set(xlabel="Rollout step", ylabel="State MSE", title="Autoregressive rollout error")
    fig.savefig(output_dir / "rollout_error.png", bbox_inches="tight"); plt.close(fig)
