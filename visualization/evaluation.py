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


def plot_eulerian_field_comparison(coords, truth, prediction, output_path, title="Eulerian field reconstruction"):
    """Plot true, predicted, and absolute-error fields on the central x-z grid plane."""
    coords, truth, prediction = np.asarray(coords), np.asarray(truth), np.asarray(prediction)
    if coords.ndim != 2 or coords.shape[1] != 3 or truth.shape != prediction.shape or truth.shape[0] != len(coords):
        raise ValueError("Expected matching [n_points, 15] fields and [n_points, 3] Eulerian coordinates")
    if truth.shape[1] < 15:
        raise ValueError(f"Expected velocity, gradient, and vorticity channels; got {truth.shape[1]}")
    axes = [np.unique(coords[:, axis]) for axis in range(3)]
    if int(np.prod([len(axis) for axis in axes])) != len(coords):
        raise ValueError("Eulerian field coordinates do not form a complete Cartesian grid")
    lookup = {tuple(np.searchsorted(axes[a], coords[i, a]) for a in range(3)): i
              for i in range(len(coords))}
    iy = len(axes[1]) // 2
    if any((ix, iy, iz) not in lookup for ix in range(len(axes[0])) for iz in range(len(axes[2]))):
        raise ValueError("Eulerian field grid is missing nodes on the selected x-z plane")
    plane_ids = np.asarray([[lookup[(ix, iy, iz)] for iz in range(len(axes[2]))]
                            for ix in range(len(axes[0]))], dtype=np.int64)
    quantities = ([ ("u_x", 0), ("u_y", 1), ("u_z", 2), ("|u|", "speed") ]
                  + [(name, index) for index, name in enumerate(
                      ("dUx_dx", "dUx_dy", "dUx_dz", "dUy_dx", "dUy_dy", "dUy_dz",
                       "dUz_dx", "dUz_dy", "dUz_dz"), start=3)]
                  + [("omega_x", 12), ("omega_y", 13), ("omega_z", 14), ("|omega|", "omega")])
    def values(array, key):
        if key == "speed": return np.linalg.norm(array[..., :3], axis=-1)
        if key == "omega": return np.linalg.norm(array[..., 12:15], axis=-1)
        return array[..., key]
    fig, axes_plot = plt.subplots(len(quantities), 3, figsize=(13, 2.1 * len(quantities)),
                                  constrained_layout=True, squeeze=False)
    for row, (name, key) in enumerate(quantities):
        v_true = values(truth[plane_ids], key).T
        v_pred = values(prediction[plane_ids], key).T
        error = np.abs(v_pred - v_true)
        vmax = max(float(np.nanpercentile(np.abs(np.concatenate((v_true.ravel(), v_pred.ravel()))), 99.5)), 1e-12)
        for col, (value, label, cmap, lo, hi) in enumerate(((v_true, "True", "RdBu_r", -vmax, vmax),
                                                            (v_pred, "Predicted", "RdBu_r", -vmax, vmax),
                                                            (error, "Absolute error", "magma", 0.0,
                                                             max(float(np.nanpercentile(error, 99.5)), 1e-12)))):
            ax = axes_plot[row, col]
            image = ax.pcolormesh(axes[0], axes[2], value, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{name}: {label}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"{title} (central y={axes[1][iy]:.5g} m)")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_eulerian_velocity_plane(x, z, truth, prediction, output_path, title="Native Eulerian velocity"):
    """Plot native-grid velocity components and speed as true/prediction/error maps."""
    x, z = np.asarray(x), np.asarray(z)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    expected = (len(z), len(x), 3)
    if truth.shape != prediction.shape or truth.shape != expected:
        raise ValueError(f"Expected matching Eulerian velocity planes shaped {expected}; got {truth.shape}, {prediction.shape}")
    quantities = (("u_x", truth[..., 0], prediction[..., 0]),
                 ("u_y", truth[..., 1], prediction[..., 1]),
                 ("u_z", truth[..., 2], prediction[..., 2]),
                 ("|u|", np.linalg.norm(truth, axis=-1), np.linalg.norm(prediction, axis=-1)))
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True, squeeze=False)
    for row, (name, actual, predicted) in enumerate(quantities):
        err = np.abs(predicted - actual)
        vmax = max(float(np.nanpercentile(np.abs(np.concatenate((actual.ravel(), predicted.ravel()))), 99.5)), 1e-12)
        for col, (values, label, cmap, lo, hi) in enumerate(((actual, "True", "RdBu_r", -vmax, vmax),
                                                              (predicted, "Predicted", "RdBu_r", -vmax, vmax),
                                                              (err, "Absolute error", "magma", 0.0,
                                                               max(float(np.nanpercentile(err, 99.5)), 1e-12)))):
            image = axes[row, col].pcolormesh(x, z, values, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
            axes[row, col].set_title(f"{name}: {label}")
            axes[row, col].set_xlabel("x [m]")
            axes[row, col].set_ylabel("z [m]")
            axes[row, col].set_aspect("equal", adjustable="box")
            fig.colorbar(image, ax=axes[row, col], shrink=0.82)
    fig.suptitle(title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
