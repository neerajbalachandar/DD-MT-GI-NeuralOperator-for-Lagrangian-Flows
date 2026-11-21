"""Readable figures for the VortexGINO evaluation (scripts/evaluate_vortex.py).

Every figure states *what* is plotted in its title, labels both axes with the
physical quantity and unit, carries a colorbar for every colored field and a
legend whenever more than one series is drawn.  Spatial panels use an equal
aspect ratio so ring shapes are not distorted.

Colors
------
* Case / ring identity: Okabe-Ito colorblind-safe order (fixed, never cycled);
  cases additionally get a distinct marker so identity never relies on color alone.
* |u| and other magnitudes: ``viridis`` (sequential).
* Signed vorticity: ``RdBu_r`` (diverging, symmetric around 0).
* Absolute errors: ``Reds`` (sequential, white = no error).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DPI = 130
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
EXTRA = ["#882255", "#44AA99", "#999933", "#AA4499", "#332288", "#661100"]
CASE_COLORS = OKABE_ITO[:7] + EXTRA          # yellow dropped: unreadable as a thin line on white
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "d"]
RING_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
TRUTH_GRAY = "#8c8c8c"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.titlesize": 11, "figure.titleweight": "bold",
})


# --------------------------------------------------------------------- naming
SEEN_IC_NOTE = "seen IC: field super-resolution check only"


def is_seen_ic_sr(name: str) -> bool:
    """The spatial-SR case reuses the initial condition of the training case leapfrog_N320_d1p000_g1p000."""
    return "spatial_sr" in name


def fmt(v, nd=2) -> str:
    """Fixed-point format that never prints '-0.00' (round first, then add 0.0 to drop the negative zero)."""
    v = float(v)
    if not math.isfinite(v):
        return str(v)
    return f"{round(v, nd) + 0.0:.{nd}f}"


def pretty_case(name: str) -> str:
    """Human-readable case label, e.g. 'leapfrog N320 d=1.25 Γ=1.0 [out-of-regime]'."""
    tag = ""
    for prefix, label in (("test_outside_regime__", " [out-of-regime]"),
                          ("test_spatial_sr__", f" [64³ grid; {SEEN_IC_NOTE}]"),
                          ("test_temporal_sr__", " [dt/4]")):
        if name.startswith(prefix):
            name, tag = name[len(prefix):], label
    if is_seen_ic_sr(name) and SEEN_IC_NOTE not in tag:
        tag += f" [{SEEN_IC_NOTE}]"
    name = re.sub(r"(\d)p(\d)", r"\1.\2", name)
    name = name.replace("leapfrog_spatial", "leapfrog").replace("_", " ")
    name = re.sub(r"\bd(\d)", r"d=\1", name)
    name = re.sub(r"\bg(\d)", r"Γ=\1", name)
    name = re.sub(r"\bgm(\d)", r"Γ2/Γ1=-\1", name)
    name = re.sub(r"\bsep(\d)", r"sep=\1", name)
    name = re.sub(r"\bsigma(\d)", r"σ=\1", name)
    name = re.sub(r"(\d\.\d+?)0+\b", r"\1", name)      # 0.750 -> 0.75, 1.000 -> 1.0
    return name + tag


def case_style(i: int) -> dict:
    return {"color": CASE_COLORS[i % len(CASE_COLORS)], "marker": MARKERS[i % len(MARKERS)]}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


# ------------------------------------------------------------- field slices
def plot_field_slice(path, *, title, X, Z, speed_true, speed_pred, wy_true, wy_pred, y_plane,
                     particles_true=None, particles_pred=None, crop_z=None, stats=None, err_u=None, err_w=None):
    """2x3 panel: rows |u| and omega_y; columns truth, prediction, |error|.

    X, Z: [nz, nx] physical coordinates of the grid plane nearest y=0.
    particles_*: [M, 2] (x, z) positions of particles with |y| < 0.15 (dots).
    stats: optional dict with 'u_rel_l2', 'w_rel_l2' (full-grid values) for the titles.
    err_u: optional [nz, nx] magnitude of the vector error |u_pred - u_true| (default: error of |u|).
    err_w: optional [nz, nx] |omega_y,pred - omega_y,true| (default computed from wy_*).
    """
    if err_u is None:
        err_u, err_u_title = np.abs(speed_pred - speed_true), "| |u|_pred - |u|_true |"
    else:
        err_u_title = "vector error |u_pred - u_true|"
    if err_w is None:
        err_w = np.abs(wy_pred - wy_true)
    if crop_z is not None:
        rows = np.where((Z[:, 0] >= crop_z[0]) & (Z[:, 0] <= crop_z[1]))[0]
        if len(rows) >= 2:
            sl = slice(rows[0], rows[-1] + 1)
            X, Z = X[sl], Z[sl]
            speed_true, speed_pred, wy_true, wy_pred, err_u, err_w = (
                a[sl] for a in (speed_true, speed_pred, wy_true, wy_pred, err_u, err_w))
    xr = float(X.max() - X.min())
    zr = float(Z.max() - Z.min())
    panel_w = 3.2
    panel_h = float(np.clip(panel_w * zr / max(xr, 1e-9), 1.6, 7.0))
    fig, axes = plt.subplots(2, 3, figsize=(3 * panel_w + 2.4, 2 * panel_h + 1.6), constrained_layout=True,
                             sharex=True, sharey=True)
    vmax_u = float(np.nanmax(speed_true))
    vmax_w = float(np.nanmax(np.abs(wy_true)))
    s = stats or {}

    def rl2(p, t):
        return float(np.sqrt(np.sum((p - t) ** 2) / max(np.sum(t ** 2), 1e-30)))

    specs = [
        (0, 0, speed_true, "viridis", (0, vmax_u), "|u| truth", "|u|  [length/time]", "max"),
        (0, 1, speed_pred, "viridis", (0, vmax_u), f"|u| prediction  (slice relL2 of |u| = {rl2(speed_pred, speed_true):.3f})",
         "|u|  [length/time]", "max"),
        (0, 2, err_u, "Reds", (0, max(float(err_u.max()), 1e-12)), err_u_title, "velocity error  [length/time]", "neither"),
        (1, 0, wy_true, "RdBu_r", (-vmax_w, vmax_w), "ω_y truth", "ω_y  [1/time]", "both"),
        (1, 1, wy_pred, "RdBu_r", (-vmax_w, vmax_w), f"ω_y prediction  (slice relL2 = {rl2(wy_pred, wy_true):.3f})",
         "ω_y  [1/time]", "both"),
        (1, 2, err_w, "Reds", (0, max(float(err_w.max()), 1e-12)), "|ω_y,pred - ω_y,true|", "abs. error of ω_y  [1/time]", "neither"),
    ]
    for r, c, data, cmap, (lo, hi), ttl, clabel, extend in specs:
        ax = axes[r, c]
        im = ax.pcolormesh(X, Z, data, cmap=cmap, vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(ttl, fontsize=8.5)
        if r == 1:
            ax.set_xlabel("x  [length]")
        if c == 0:
            ax.set_ylabel("z (axial)  [length]")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, extend=extend)
        cb.set_label(clabel, fontsize=8)
        dot_color = "white" if cmap == "viridis" else "black"
        if c == 0 and particles_true is not None and len(particles_true):
            ax.scatter(particles_true[:, 0], particles_true[:, 1], s=1.2, c=dot_color, alpha=0.7, lw=0)
        if c == 1 and particles_pred is not None and len(particles_pred):
            ax.scatter(particles_pred[:, 0], particles_pred[:, 1], s=1.2, c=dot_color, alpha=0.7, lw=0)
        ax.set_xlim(float(X.min()), float(X.max()))
        ax.set_ylim(float(Z.min()), float(Z.max()))
    sub = f"grid plane y = {fmt(y_plane, 3)}"
    if s:
        sub += f"   |   full-grid relL2:  u = {s.get('u_rel_l2', float('nan')):.3f},  ω = {s.get('w_rel_l2', float('nan')):.3f}"
    note = "dots = particles with |y| < 0.15"
    if particles_pred is not None:
        note += " (truth particles on truth panels, predicted particles on prediction panels)"
    fig.suptitle(f"{title}\n{sub}\n{note}", fontsize=10)
    return _save(fig, path)


# -------------------------------------------------------- rollout snapshots
def plot_rollout_snapshots(path, *, case_label, snapshots, labels, n_rings, ref_radius, ring_desc=None,
                           diverged_at=None):
    """snapshots: list of dicts {step, time, x_true [N,3], x_pred [N,3], err}; x-z side view."""
    n = len(snapshots)
    lims = []
    for snap in snapshots:
        xt, xp = snap["x_true"], snap["x_pred"]
        ok = np.isfinite(xp).all(1)
        pts = np.concatenate([xt[:, [0, 2]], xp[ok][:, [0, 2]]]) if ok.any() else xt[:, [0, 2]]
        lo_t, hi_t = xt[:, [0, 2]].min(0), xt[:, [0, 2]].max(0)
        # keep wildly diverged particles from shrinking the rings to dots
        lo = np.maximum(pts.min(0), lo_t - 3 * ref_radius)
        hi = np.minimum(pts.max(0), hi_t + 3 * ref_radius)
        pad = 0.15 * max(float((hi - lo).max()), 1e-6)
        lims.append((lo - pad, hi + pad))
    widths = [float(hi[0] - lo[0]) / float(hi[1] - lo[1]) for lo, hi in lims]
    ratios = [max(0.6, min(w, 2.5)) for w in widths]
    fig_w = min(18.0, sum(2.6 * r for r in ratios) + 1.0)
    panel_h = (fig_w - 1.0) / sum(ratios) * max(1.0 / w for w in widths)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, float(np.clip(panel_h, 1.8, 7.0)) + 1.6),
                             gridspec_kw={"width_ratios": [max(0.6, min(w, 2.5)) for w in widths]},
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, snap, (lo, hi) in zip(axes, snapshots, lims):
        xt, xp = snap["x_true"], snap["x_pred"]
        ax.scatter(xt[:, 0], xt[:, 2], s=2.0, c=TRUTH_GRAY, alpha=0.6, lw=0, rasterized=True)
        for k in range(n_rings):
            m = labels == k
            ax.scatter(xp[m, 0], xp[m, 2], s=1.2, c=RING_COLORS[k % len(RING_COLORS)], alpha=0.8, lw=0,
                       rasterized=True)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x  [length]")
        ax.set_title(f"step {snap['step']}  (t = {snap['time']:.3f})\nmean |Δx| = {snap['err']:.3g}", fontsize=8.5)
    axes[0].set_ylabel("z (axial)  [length]")
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=TRUTH_GRAY, label="truth (FLOWVPM)")]
    for k in range(n_rings):
        handles.append(Line2D([], [], ls="", marker="o", ms=5, color=RING_COLORS[k % len(RING_COLORS)],
                              label=f"prediction, ring {k + 1}" + (ring_desc[k] if ring_desc else "")))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    div = (f"\nrollout DIVERGED at step {diverged_at}; snapshots span the steps before divergence"
           if diverged_at else "")
    fig.suptitle(f"Autoregressive rollout, side view (x-z projection of all particles): {case_label}{div}")
    return _save(fig, path)


# ------------------------------------------------------------ ring dynamics
def plot_ring_dynamics(path, *, case_label, time, z_true, z_pred, r_true, r_pred, ring_desc=None,
                       diverged_time=None, note=None):
    """z_*, r_*: [T, K] ring centroid axial position and radius about the ring's own axis (circulation-weighted).

    note: optional physics hint for the title (e.g. what the leapfrogging signature looks like).
    """
    K = z_true.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for k in range(K):
        col = RING_COLORS[k % len(RING_COLORS)]
        axes[0].plot(time, z_true[:, k], color=col, lw=2.0, ls="-")
        axes[0].plot(time, z_pred[:, k], color=col, lw=2.0, ls="--")
        axes[1].plot(time, r_true[:, k], color=col, lw=2.0, ls="-")
        axes[1].plot(time, r_pred[:, k], color=col, lw=2.0, ls="--")
    axes[0].set_title("Ring centroid axial position z(t)")
    axes[0].set_ylabel("z centroid  [length]")
    axes[1].set_title("Ring radius R(t) (circulation-weighted distance from the ring's own axis)")
    axes[1].set_ylabel("ring radius R  [length]")
    for ax in axes:
        ax.set_xlabel("physical time t")
        if diverged_time is not None:
            ax.axvline(diverged_time, color="black", lw=1.0, ls=":")
            ax.text(diverged_time, 0.02, " rollout diverged", transform=ax.get_xaxis_transform(), fontsize=8)
    handles = [Line2D([], [], color=RING_COLORS[k % len(RING_COLORS)], lw=2.5,
                      label=f"ring {k + 1}" + (ring_desc[k] if ring_desc else "")) for k in range(K)]
    handles += [Line2D([], [], color="black", lw=1.5, ls="-", label="truth (FLOWVPM)"),
                Line2D([], [], color="black", lw=1.5, ls="--", label="VortexGINO rollout")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Ring dynamics: {case_label}" + (f"\n{note}" if note else ""))
    return _save(fig, path)


# ------------------------------------------------------- all-case summaries
def divergence_rule_text(results) -> str:
    """Legend text describing the divergence rule, built from the criteria stored in the results."""
    texts = []
    for res in results:
        crit = (res.get("rollout") or {}).get("divergence_criteria") or {}
        parts = ["non-finite values"]
        if crit.get("sum_abs_gamma_rel_change") is not None:
            parts.append(f"Σ|Γ| off its start value by > {fmt(100 * crit['sum_abs_gamma_rel_change'], 0)}%")
        if crit.get("ring_spread_ratio") is not None:
            parts.append(f"ring spread > {crit['ring_spread_ratio']:g}x its start value")
        # criteria of older metrics.json files (truth-based)
        if crit.get("position_error_over_R") is not None:
            parts.append(f"|Δx| > {crit['position_error_over_R']:g} R₀")
        if crit.get("gamma_rel_l2") is not None:
            parts.append(f"Γ relL2 > {crit['gamma_rel_l2']:g}")
        t = ", ".join(parts[:-1]) + " or " + parts[-1] if len(parts) > 1 else parts[0]
        if t not in texts:
            texts.append(t)
    return "; ".join(texts)


def _positive(v):
    v = np.array(v, dtype=float)
    return np.where(v > 0, v, np.nan)        # log axes: exact zeros (step 0) are not drawable


def plot_error_vs_step(path, results):
    """results: list of per-case result dicts (see evaluate_vortex.py) with a 'rollout' block.

    2x2: position error (+ persistence baseline), circulation error, grid velocity from rolled-out particles,
    and a conservation panel (total |Γ| and linear impulse, both relative to the truth at the same step).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), constrained_layout=True)
    ax_pos, ax_gam, ax_fld, ax_con = axes.ravel()
    handles = []
    max_step = 1
    for i, res in enumerate(results):
        ro = res.get("rollout")
        if not ro:
            continue
        st = case_style(i)
        steps = np.asarray(ro["steps"])
        max_step = max(max_step, int(steps.max()))
        me = max(1, len(steps) // 8)
        pe = _arr(ro["position_error_over_R"])
        pb = _arr(ro["persistence_error_over_R"])
        ge = _arr(ro["gamma_rel_l2"])
        sg = _positive(_arr(ro.get("sum_abs_gamma_drift", [np.nan] * len(steps))))
        im = _positive(_arr(ro["impulse_drift"]))
        sel = steps > 0
        if ro.get("diverged_at_step"):
            kd = int(ro["diverged_at_step"]) - 1          # last finite step
            for ax, v in ((ax_pos, pe), (ax_gam, ge), (ax_con, sg)):
                if kd >= 1 and np.isfinite(v[kd]):
                    ax.plot([steps[kd]], [v[kd]], marker="X", ms=11, color=st["color"], mec="black", mew=1.0,
                            ls="", zorder=5)
        ax_pos.plot(steps[sel], pe[sel], color=st["color"], marker=st["marker"], markevery=me, ms=4, lw=1.6)
        ax_pos.plot(steps[sel], pb[sel], color=st["color"], lw=1.0, ls=":", alpha=0.8)
        ax_gam.plot(steps[sel], ge[sel], color=st["color"], marker=st["marker"], markevery=me, ms=4, lw=1.6)
        fs = np.asarray(ro["field_steps"])
        fu = _arr(ro["field_u_rel_l2"])
        ax_fld.plot(fs, fu, color=st["color"], marker=st["marker"], ms=4, lw=1.6)
        ax_con.plot(steps[sel], sg[sel], color=st["color"], marker=st["marker"], markevery=me, ms=4, lw=1.6)
        ax_con.plot(steps[sel], im[sel], color=st["color"], lw=1.3, ls="--", alpha=0.9)
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], ms=5, lw=1.6,
                              label=f"{pretty_case(res['case'])} ({res['split']})"))
    ax_pos.set_title("Mean particle position error / initial ring radius")
    ax_pos.set_ylabel("mean |x_pred - x_true| / R₀")
    ax_gam.set_title("Circulation error  ‖Γ_pred - Γ_true‖ / ‖Γ_true‖")
    ax_gam.set_ylabel("relative L2 error of Γ")
    ax_fld.set_title("Velocity field decoded from ROLLED-OUT particles\nvs truth grid  ‖u_pred - u_true‖/‖u_true‖")
    ax_fld.set_ylabel("relative L2 error of grid velocity")
    ax_con.set_title("Conservation vs truth at the same step\n"
                     "solid: |Σ|Γ|_pred - Σ|Γ|_true| / Σ|Γ|_true    dashed: |I_pred - I_true| / Σ_rings |I_ring|")
    ax_con.set_ylabel("relative drift")
    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.set_xlim(-0.02 * max_step, 1.02 * max_step)     # same step range on every panel
        ax.set_xlabel("rollout step (saved-frame intervals)")
        ax.grid(True, which="both", alpha=0.25)
    handles.append(Line2D([], [], color="black", lw=1.0, ls=":",
                          label="persistence baseline (particles frozen at start), top-left panel"))
    handles.append(Line2D([], [], color="black", lw=1.6, ls="-", label="total |Γ| drift, bottom-right panel"))
    handles.append(Line2D([], [], color="black", lw=1.3, ls="--",
                          label="linear impulse I = ½ Σ x × Γ drift, bottom-right panel"))
    if any((r.get("rollout") or {}).get("diverged_at_step") for r in results):
        handles.append(Line2D([], [], color="white", marker="X", ms=10, mec="black", ls="",
                              label=f"rollout diverged after this step ({divergence_rule_text(results)})"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Autoregressive rollout error growth, all evaluated cases (lower is better)")
    return _save(fig, path)


def plot_field_error_vs_time(path, results):
    """Teacher-forced errors per frame: grid u, grid ω and the particle circulation rate (dΓ/dt)/|Γ|."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), constrained_layout=True, sharex=True)  # same frame range
    handles = []
    for i, res in enumerate(results):
        tf = res.get("teacher_forced")
        if not tf:
            continue
        st = case_style(i)
        fr = np.asarray(tf["frames"])
        axes[0].plot(fr, _arr(tf["u_rel_l2"]), color=st["color"], marker=st["marker"], ms=4, lw=1.5)
        axes[1].plot(fr, _arr(tf["w_rel_l2"]), color=st["color"], marker=st["marker"], ms=4, lw=1.5)
        gr = _arr(tf.get("particle_gamma_rate_rel_l2", [np.nan] * len(fr)))
        ok = np.isfinite(gr)
        axes[2].plot(fr[ok], gr[ok], color=st["color"], marker=st["marker"], ms=4, lw=1.5)
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], ms=5, lw=1.5,
                              label=f"{pretty_case(res['case'])} ({res['split']})"))
    axes[0].set_title("Velocity  ‖u_pred - u_true‖ / ‖u_true‖  on the full grid")
    axes[0].set_ylabel("relative L2 error of u")
    axes[1].set_title("Vorticity (curl of predicted J)  ‖ω_pred - ω_true‖ / ‖ω_true‖")
    axes[1].set_ylabel("relative L2 error of ω")
    axes[2].set_title("Particle circulation rate (dΓ/dt)/|Γ| at the true particles\n"
                      "‖rate_pred - (J_pᵀΓ)/|Γ|‖ / ‖(J_pᵀΓ)/|Γ|‖  (frame 0: no FLOWVPM J_p)")
    axes[2].set_ylabel("relative L2 error of Γ-rate")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("frame index")
        ax.grid(True, which="both", alpha=0.25)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Teacher-forced reconstruction (decoded from the TRUE particles at each frame)")
    return _save(fig, path)


def _parity_panel(ax, t, p, title, xlabel, ylabel, color):
    lo = float(min(t.min(), p.min()))
    hi = float(max(t.max(), p.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.0, ls="--", label="y = x (perfect)")
    ax.scatter(t, p, s=3, alpha=0.4, color=color, lw=0, rasterized=True, label="particles")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False)


def plot_particle_parity(path, *, case_label, frame, time, u_true, u_pred, rate_true=None, rate_pred=None):
    """Parity of particle velocity components (+ the circulation rate (dΓ/dt)/|Γ| when given)."""
    has_rate = rate_true is not None and rate_pred is not None
    n = 4 if has_rate else 3
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.6), constrained_layout=True)
    total = float(np.linalg.norm(u_pred - u_true) / max(np.linalg.norm(u_true), 1e-30))
    for c in range(3):
        t, p = u_true[:, c], u_pred[:, c]
        rl = float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-30))
        comp = "xyz"[c]
        _parity_panel(axes[c], t, p, f"u_{comp}:  relL2 = {rl:.3f}", f"true u_{comp} at particle  [length/time]",
                      f"predicted u_{comp}  [length/time]", OKABE_ITO[0])
    sub = f"velocity, all components relL2 = {total:.3f}"
    if has_rate:
        rl = float(np.linalg.norm(rate_pred - rate_true) / max(np.linalg.norm(rate_true), 1e-30))
        _parity_panel(axes[3], rate_true.reshape(-1), rate_pred.reshape(-1),
                      f"circulation rate (dΓ/dt)/|Γ|, x/y/z pooled:\nrelL2 = {rl:.3f}",
                      "exact (J_pᵀΓ)/|Γ|  [1/time]", "predicted rate  [1/time]", OKABE_ITO[3])
        sub += f";  circulation rate relL2 = {rl:.3f}"
    fig.suptitle(f"Particle parity (teacher-forced, frame {frame}, t = {time:.3f}): {case_label}\n"
                 f"{sub}, N = {len(u_true)} particles")
    return _save(fig, path)


# ---------------------------------------------------------- training curves
TERM_TITLES = {
    "grid_u": "Grid velocity u", "grid_w": "Grid vorticity ω", "particle_u": "Particle velocity u_p",
    "particle_J": "Particle velocity gradient J_p", "gamma_rate": "Particle circulation rate (dΓ/dt)/|Γ|",
    "step_x": "One-step displacement Δx", "step_gamma": "One-step circulation change ΔΓ",
}


def _empty(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_yticks([])


def plot_training_curves(path, history, title="Training history", operator_epochs=None, total_epochs=None,
                         rollout_steps=None, rollout_start=None, selected_epoch=None):
    """rollout_steps / rollout_start: the train_config's val_rollout_steps / val_rollout_start (panel title).
    selected_epoch: epoch of the evaluated checkpoint, marked (dotted line) in every panel."""
    if not history:
        return None
    ep = np.array([h.get("epoch", i + 1) for i, h in enumerate(history)], dtype=float)
    joint = [h["epoch"] for h in history if h.get("phase") == "joint"]
    switch = (min(joint) - 0.5) if joint else None
    planned = switch is None and operator_epochs is not None
    if planned:
        switch = float(operator_epochs) + 0.5

    def series(key):
        return np.array([h.get(key, np.nan) if h.get(key) is not None else np.nan for h in history], dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    axes = axes.ravel()
    n_terms = len(TERM_TITLES)
    for ax, (term, ttl) in zip(axes[:n_terms], TERM_TITLES.items()):
        tr = np.sqrt(series(f"train_{term}"))
        va = series(f"val_{term}")
        if np.isfinite(tr).any():
            ax.plot(ep, tr, color=OKABE_ITO[0], lw=1.6, marker="o", ms=2.5, label="train (√ of mean rel. sq. error)")
        if np.isfinite(va).any():
            ax.plot(ep, va, color=OKABE_ITO[3], lw=1.6, marker="s", ms=2.5, label="validation (rel. L2)")
        ax.set_title(ttl)
        ax.set_ylabel("relative L2 error")
        if not (np.isfinite(tr).any() or np.isfinite(va).any()):
            _empty(ax, "not trained yet\n(joint phase only)" if term.startswith("step_") else "not logged")
        else:
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="upper right")
    ax = axes[n_terms]
    any_ro = False
    for key, lab, col in (("val_rollout_mean_position_error", "mean particle position error [length]", OKABE_ITO[0]),
                          ("val_rollout_ring_z_error", "ring centroid |Δz| [length]", OKABE_ITO[1]),
                          ("val_rollout_ring_radius_error", "ring radius |ΔR| [length]", OKABE_ITO[2]),
                          ("val_rollout_gamma_rel_l2", "circulation rel. L2 [-]", OKABE_ITO[4]),
                          ("val_rollout_sum_abs_gamma_drift", "Σ|Γ| drift [-]", OKABE_ITO[5])):
        v = series(key)
        if np.isfinite(v).any():
            any_ro = True
            ax.plot(ep, v, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
    if rollout_steps is not None and rollout_start is not None:
        ax.set_title(f"Validation rollout ({rollout_steps} steps from frame {rollout_start})")
    elif rollout_steps is not None:
        ax.set_title(f"Validation rollout ({rollout_steps} steps)")
    else:
        ax.set_title("Validation rollout")
    ax.set_ylabel("error at the end of the rollout")
    if any_ro:
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="upper right")
    else:
        _empty(ax, "no rollout validation yet\n(joint phase only)")
    ax = axes[n_terms + 1]
    sc = series("score")
    ax.plot(ep, sc, color="black", lw=1.6, marker="o", ms=2.5, label="validation score (model selection)")
    ax.set_yscale("log")
    ax.set_title("Validation score (not comparable across phases)")
    ax.set_ylabel("score")
    lr = series("lr")
    if np.isfinite(lr).any():
        ax.plot([], [], " ", label=f"lr: {np.nanmax(lr):.1e} -> {lr[-1]:.1e}")
    ax.legend(frameon=False, loc="upper right")
    for ax in axes[n_terms + 2:]:
        ax.set_visible(False)
    for ax in axes[:n_terms + 2]:
        if selected_epoch is not None:
            ax.axvline(float(selected_epoch), color="black", ls=":", lw=1.2)
        ax.set_xlabel("epoch")
        ax.set_xlim(0.5, max(float(ep.max()), float(total_epochs or 0), (switch or 0) + 3) + 0.5)
        ax.grid(True, which="both", alpha=0.25)
        if switch is not None:
            ax.axvline(switch, color="gray", lw=1.2, ls="--")
            ax.text(switch, 0.02, " joint phase →" + (" (planned)" if planned else ""), transform=ax.get_xaxis_transform(), va="bottom", ha="left",
                    fontsize=7.5, color="gray")
    fig.suptitle(f"{title}  (dashed vertical line: operator → joint phase switch)")
    return _save(fig, path)


__all__ = ["SEEN_IC_NOTE", "is_seen_ic_sr", "fmt", "pretty_case", "plot_field_slice", "plot_rollout_snapshots",
           "plot_ring_dynamics", "plot_error_vs_step", "plot_field_error_vs_time", "plot_particle_parity",
           "plot_training_curves", "divergence_rule_text"]
