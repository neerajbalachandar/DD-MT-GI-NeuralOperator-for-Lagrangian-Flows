from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_COMPONENTS = ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
FIELD_COMPONENTS = ("u_x", "u_y", "u_z", "gradU_xx", "gradU_xy", "gradU_xz",
                    "gradU_yx", "gradU_yy", "gradU_yz", "gradU_zx", "gradU_zy", "gradU_zz")


def plot_state_parity(truth, prediction, output_path, title):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.shape[-1] != len(STATE_COMPONENTS):
        raise ValueError(f"Expected matching state arrays ending in 7 channels; got {truth.shape}, {prediction.shape}")
    truth, prediction = truth.reshape(-1, 7), prediction.reshape(-1, 7)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index >= len(STATE_COMPONENTS):
            axis.remove()
            continue
        x, y = truth[:, index], prediction[:, index]
        axis.scatter(x, y, s=5, alpha=0.35, color="#176b87", rasterized=True)
        bounds = np.nanmin(np.concatenate((x, y))), np.nanmax(np.concatenate((x, y)))
        axis.plot(bounds, bounds, color="#d66b35", linewidth=1)
        axis.set_title(STATE_COMPONENTS[index])
        axis.set_xlabel("Ground truth")
        axis.set_ylabel("Prediction")
        axis.grid(alpha=0.18)
    fig.suptitle(title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_field_error_maps(coords, truth, prediction, output_path, component_names=FIELD_COMPONENTS,
                          title="Absolute field error"):
    coords, truth, prediction = np.asarray(coords), np.asarray(truth), np.asarray(prediction)
    if coords.ndim != 2 or coords.shape[1] != 3 or truth.shape != prediction.shape or truth.shape[0] != len(coords):
        raise ValueError("Expected xyz coordinates and matching [n_points, n_components] field arrays")
    channels = min(truth.shape[1], len(component_names))
    if channels == 0:
        raise ValueError("Field arrays contain no components to plot")
    error = np.abs(prediction[:, :channels] - truth[:, :channels])
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= channels:
            axis.remove()
            continue
        points = axis.scatter(coords[:, 0], coords[:, 2], c=error[:, index], s=5,
                              cmap="magma", rasterized=True)
        axis.set_title(f"{component_names[index]} | absolute error")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.set_aspect("equal", adjustable="datalim")
        fig.colorbar(points, ax=axis, shrink=0.82)
    fig.suptitle(title + " (x-z projection)")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
