from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_field_comparison(x, y, true_field, predicted_field, output_path):
    """Same coordinates, extent and color limits for truth/prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    lo = float(np.nanmin([np.nanmin(true_field), np.nanmin(predicted_field)]))
    hi = float(np.nanmax([np.nanmax(true_field), np.nanmax(predicted_field)]))
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
    for ax, field, title in zip(axes, (true_field, predicted_field), ("True", "Predicted")):
        ax.imshow(field, origin="lower", extent=extent, aspect="equal", vmin=lo, vmax=hi)
        ax.set_title(title)
    fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)
