from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_velocity_magnitude_summary(x, z, true_velocity, predicted_velocity, freestream_speeds,
                                    output_path, phases=(0.3, 0.5, 0.7),
                                    x_stations=(0.2, 0.4, 0.6), y_plane=0.0):
    """Plot native-grid |U|/U_inf maps and downstream profiles at selected phases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x, z = np.asarray(x), np.asarray(z)
    true_velocity, predicted_velocity = np.asarray(true_velocity), np.asarray(predicted_velocity)
    expected = (len(phases), len(z), len(x), 3)
    if true_velocity.shape != expected or predicted_velocity.shape != expected:
        raise ValueError(f"Velocity slices must have shape {expected}")
    speeds = np.asarray(freestream_speeds, dtype=float).reshape(-1)
    if speeds.shape != (len(phases),) or np.any(~np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("Provide one finite, positive freestream speed per phase")
    if len(x_stations) == 0:
        raise ValueError("At least one downstream x station is required")
    true_magnitude = np.linalg.norm(true_velocity, axis=-1) / speeds[:, None, None]
    predicted_magnitude = np.linalg.norm(predicted_velocity, axis=-1) / speeds[:, None, None]
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(z)), float(np.max(z))]
    vmax = max(float(np.nanmax(true_magnitude)), float(np.nanmax(predicted_magnitude)), 1e-12)
    fig, axes = plt.subplots(len(phases), 3, figsize=(13, 3.8 * len(phases)), squeeze=False,
                             constrained_layout=True)
    for row, phase in enumerate(phases):
        true_im = axes[row, 0].imshow(true_magnitude[row], origin="lower", extent=extent,
                                      aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
        pred_im = axes[row, 1].imshow(predicted_magnitude[row], origin="lower", extent=extent,
                                      aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"Simulation |u|/U_inf, phase={phase:.2f}")
        axes[row, 1].set_title(f"GINO |u|/U_inf, phase={phase:.2f}")
        for axis in axes[row, :2]:
            axis.set_xlabel("x [m]")
            axis.set_ylabel("z [m]")
        colors = ("#176b87", "#d66b35", "#588157", "#a23b72", "#8c6d31")
        for index, station in enumerate(x_stations):
            ix = int(np.argmin(np.abs(x - float(station))))
            color = colors[index % len(colors)]
            axes[row, 2].plot(true_magnitude[row, :, ix], z, color=color,
                              label=f"True x={x[ix]:.2f}")
            axes[row, 2].plot(predicted_magnitude[row, :, ix], z, color=color, linestyle="--",
                              label=f"Pred x={x[ix]:.2f}")
        axes[row, 2].set_title(f"Downstream profiles, phase={phase:.2f}")
        axes[row, 2].set_xlabel("|u|/U_inf")
        axes[row, 2].set_ylabel("z [m]")
        axes[row, 2].xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[row, 2].tick_params(axis="x", labelsize=8)
        axes[row, 2].grid(alpha=0.2)
        axes[row, 2].legend(frameon=False, fontsize="x-small", ncol=2)
    fig.colorbar(true_im, ax=axes[:, :2].ravel().tolist(), shrink=0.85, label="|u|/U_inf")
    fig.suptitle(f"Velocity-magnitude reconstruction at y={y_plane:.4g} m")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_native_ux_comparison(x, z, true_ux, predicted_ux, output_path, phase=None, physical_time=None):
    """Show direct native truth, prediction, and absolute error for u_x."""
    x, z = np.asarray(x), np.asarray(z)
    true_ux, predicted_ux = np.asarray(true_ux), np.asarray(predicted_ux)
    if true_ux.shape != (len(z), len(x)) or predicted_ux.shape != true_ux.shape:
        raise ValueError("Expected matching [z,x] native-grid u_x slices")
    error = np.abs(predicted_ux - true_ux)
    lo = float(np.nanmin((true_ux, predicted_ux)))
    hi = float(np.nanmax((true_ux, predicted_ux)))
    extent = [float(x.min()), float(x.max()), float(z.min()), float(z.max())]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, field, title in ((axes[0], true_ux, "True simulation u_x"),
                               (axes[1], predicted_ux, "Predicted u_x")):
        im = axis.imshow(field, origin="lower", extent=extent, aspect="auto",
                         vmin=lo, vmax=hi, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
    err_im = axes[2].imshow(error, origin="lower", extent=extent, aspect="auto",
                            vmin=0.0, cmap="magma")
    axes[2].set_title("Absolute error |u_x,pred - u_x,true|")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("z [m]")
    fig.colorbar(im, ax=axes[:2], shrink=0.88, label="u_x [m/s]")
    fig.colorbar(err_im, ax=axes[2], shrink=0.88, label="Absolute error [m/s]")
    when = []
    if phase is not None:
        when.append(f"phase={phase:.3f}")
    if physical_time is not None:
        when.append(f"t={physical_time:.5g} s")
    fig.suptitle("Native-grid mid-slice velocity comparison" + (" | " + ", ".join(when) if when else ""))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_ux_profiles(x, z, true_velocity, predicted_velocity, output_path,
                                freestream_speed, chord, leading_edge_x,
                                x_stations_over_c=(0.2, 0.4, 0.6)):
    """Plot native-grid u_x/U_inf versus z/c at specified downstream x/c stations."""
    x, z = np.asarray(x), np.asarray(z)
    true_velocity, predicted_velocity = np.asarray(true_velocity), np.asarray(predicted_velocity)
    expected = (len(z), len(x), 3)
    if true_velocity.shape != expected or predicted_velocity.shape != expected:
        raise ValueError(f"Expected matching velocity slices with shape {expected}")
    if not np.isfinite(freestream_speed) or freestream_speed <= 0 or not np.isfinite(chord) or chord <= 0:
        raise ValueError("Freestream speed and reference chord must be finite and positive")
    if len(x_stations_over_c) == 0:
        raise ValueError("At least one x/c station is required")
    fig, axes = plt.subplots(1, len(x_stations_over_c), figsize=(4.2 * len(x_stations_over_c), 5),
                             sharey=True, squeeze=False, constrained_layout=True)
    for axis, station in zip(axes[0], x_stations_over_c):
        requested_x = leading_edge_x + float(station) * chord
        ix = int(np.argmin(np.abs(x - requested_x)))
        axis.plot(true_velocity[:, ix, 0] / freestream_speed, z / chord,
                  color="#176b87", label="True", linewidth=1.6)
        axis.plot(predicted_velocity[:, ix, 0] / freestream_speed, z / chord,
                  color="#d66b35", linestyle="--", label="Predicted", linewidth=1.6)
        axis.set_title(f"x/c={float(station):.2f} (grid x={x[ix]:.4g} m)")
        axis.set_xlabel("u_x/U_inf")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("z/c")
    fig.suptitle("Normalized downstream streamwise-velocity profiles")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_downstream_profiles(x, z, true_field, predicted_field, output_path,
                             freestream_speed=1.0, x_stations=None, component_label="u_x",
                             y_plane=0.0, phase=0.0):
    """Plot native-grid velocity profiles u/U_inf versus z at downstream x stations."""
    x, z = np.asarray(x), np.asarray(z)
    true_field, predicted_field = np.asarray(true_field), np.asarray(predicted_field)
    if true_field.ndim != 2 or true_field.shape != predicted_field.shape or true_field.shape != (len(z), len(x)):
        raise ValueError("Expected matching native x-z field slices")
    if not np.isfinite(freestream_speed) or freestream_speed <= 0:
        raise ValueError(f"Freestream speed must be positive, got {freestream_speed}")
    if x_stations is not None and len(x_stations):
        stations = np.asarray(x_stations, dtype=float)
    else:
        # Keep stations downstream of the origin when the mesh includes upstream points.
        downstream = x[x >= max(0.0, float(x.min()))]
        if downstream.size < 2:
            downstream = x
        stations = np.linspace(float(downstream.min()), float(downstream.max()), 5)[1:]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(stations), figsize=(4 * len(stations), 5), squeeze=False,
                             sharey=True, constrained_layout=True)
    for axis, station in zip(axes[0], stations):
        ix = int(np.argmin(np.abs(x - station)))
        axis.plot(true_field[:, ix] / freestream_speed, z, color="#176b87", label="Simulation")
        axis.plot(predicted_field[:, ix] / freestream_speed, z, color="#d66b35", label="GINO")
        axis.set_title(f"x={x[ix]:.4g} m")
        axis.set_xlabel(f"{component_label} / U_inf")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("z [m]")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Downstream velocity profiles, y={y_plane:.4g} m, phase={phase:.3f}")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
