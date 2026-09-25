from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_field_comparison(x, y, true_field, predicted_field, output_path,
                          phases=(0.3, 0.5, 0.7), component_label="Velocity"):
    """Three-phase native truth/prediction comparison and matching profiles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    true_field, predicted_field = np.asarray(true_field), np.asarray(predicted_field)
    if true_field.ndim != 3 or predicted_field.shape != true_field.shape or true_field.shape[0] != len(phases):
        raise ValueError("Fields must have shape (n_phases, n_y, n_x) with one shared grid")
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    lo = float(np.nanmin([np.nanmin(true_field), np.nanmin(predicted_field)]))
    hi = float(np.nanmax([np.nanmax(true_field), np.nanmax(predicted_field)]))
    fig, axes = plt.subplots(3, len(phases), figsize=(4.2 * len(phases), 10), constrained_layout=True)
    for col, phase in enumerate(phases):
        for row, field, title in ((0, true_field[col], "True"), (1, predicted_field[col], "Predicted")):
            im = axes[row, col].imshow(field, origin="lower", extent=extent, aspect="equal", vmin=lo, vmax=hi)
            axes[row, col].set_title(f"{title}, phase={phase:.1f}")
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
        iy = true_field.shape[1] // 2
        axes[2, col].plot(x, true_field[col, iy], label="True", color="#176b87")
        axes[2, col].plot(x, predicted_field[col, iy], label="Predicted", color="#d66b35")
        axes[2, col].set_title(f"{component_label} profile, phase={phase:.1f}")
        axes[2, col].set_xlabel("x")
        axes[2, col].set_ylabel(component_label)
        axes[2, col].grid(alpha=0.2)
    fig.colorbar(im, ax=axes[:2, :].ravel().tolist(), shrink=0.82, label=component_label)
    axes[2, 0].legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)
