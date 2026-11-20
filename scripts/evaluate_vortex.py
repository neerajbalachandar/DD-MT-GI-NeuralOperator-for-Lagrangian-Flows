"""Evaluate a trained VortexGINO: field reconstruction + long autoregressive particle rollouts.

    python scripts/evaluate_vortex.py --checkpoint runs/vortex_gino/final_model.pt
    python scripts/evaluate_vortex.py --quick                      # smoke test: 1 case, 20 steps
    python scripts/evaluate_vortex.py --cases leapfrog_N160_d1p500_g1p000 --rollout-steps 60

Outputs (default: <checkpoint dir>/evaluation_<checkpoint stem>/):
    metrics.json   per-case, per-frame / per-step arrays (teacher-forced fields, rollout) -- ALL cases
    summary.csv    one row per case with the headline numbers (column "group" says how to read it)
    summary.md     overall means per case group (validation / in-regime test / out-of-regime test) with a
                   plain-language explanation of every metric (also printed)
    plots/         PNG figures (see visualization/vortex_plots.py)

Targets: the archives hold the EXACT Biot-Savart grid velocity of the stored particles (FLOWVPM's probe-
contaminated field is kept as grid_velocity_flowvpm) and particle J including the self-induced term.

Metric definitions
------------------
relL2(a, b) = ||a - b||_2 / ||b||_2 over all nodes/particles and components.
Teacher-forced: the operator is evaluated on the TRUE particles of a frame.
    amplitude ratio = <pred, true> / <true, true> (1 = right magnitude, <1 = too weak)
    vortex region   = grid nodes with |omega_true| >= 95th percentile of |omega_true|; far field = the rest.
    *_err_vortex_frac_of_total / *_err_far_frac_of_total = ||err on region|| / ||truth on the FULL grid||:
                   the share of the full-grid relL2 that comes from that region (the squares of the two add
                   up to the square of the full-grid relL2).  Normalising the far-field error by the (tiny)
                   far-field truth norm instead would give misleadingly large numbers.
    particle Gamma-rate relL2: the predicted relative circulation rate (dGamma/dt)/|Gamma| at the true particles
                   (model.gamma_rate on decoder features; J^T Gamma/|Gamma| of the predicted J for
                   gamma_dynamics='stretching') vs the exact (J_p^T Gamma)/|Gamma|.
Rollout: particles are advanced with model.step (model.substeps coupled RK4 substeps per saved interval +
    learned residual) from --start-frame, using only the model's own predictions (no truth after the start).
    position error / R0: mean_p |x_pred - x_true| divided by the largest initial ring radius R0.
    persistence baseline: particles frozen at the start frame (what "no model" scores).
    impulse drift: |I_pred - I_true| / sum_k |I_k,true(start)|, I = 0.5 sum_p x_p x Gamma_p
                   (normalised by the per-ring impulse magnitudes so opposite-sign pairs are well defined).
    sum|Gamma| drift: |S_pred - S_true| / S_true with S = sum_p |Gamma_p| (total circulation magnitude).
    out-of-box fraction: particles outside the co-moving latent box (model.box_half around the
                   circulation centroid); there the operator extrapolates.
    divergence (uses ONLY the predicted state, never the truth): a rollout stops (later steps NaN, "DIV@k" in
                   the table) when values become non-finite, when sum|Gamma| deviates from its start value by more
                   than --diverge-gamma-sum (fraction), or when the mean distance of a ring's particles from that
                   ring's circulation centroid grows beyond --diverge-spread x its start value.
Rings: particles are assigned to rings once, at frame 0 (they keep their identity); coaxial rings are
    clustered on z, the side-by-side free pair on x.  Ring radius = circulation-weighted distance from the
    ring's OWN axis (through its circulation centroid, along its own impulse direction).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gino.data.vortex_frames import CaseData, ring_membership  # noqa: E402
from gino.model.vortex_operator import VortexGINO, circulation_centroid  # noqa: E402
import gino.model.vortex_operator as vo  # noqa: E402
from gino.training.vortex_trainer import decode_chunked, gamma_rate_target, vorticity_from_jacobian  # noqa: E402
from visualization import vortex_plots as vp  # noqa: E402

CHUNK = 16384

# Case groups for the overall summary.  The spatial-SR case reuses the initial condition of the TRAINING case
# leapfrog_N320_d1p000_g1p000 (only the output grid is finer), so it says nothing about generalization.
IN_REGIME_TEST = ("leapfrog_N160_d1p500_g1p000", "leapfrog_N320_d1p250_g1p000")
OUT_OF_REGIME_PREFIXES = ("test_outside_regime", "test_temporal_sr")
GROUPS = [("val", "(a) validation cases"),
          ("test_in_regime", "(b) in-regime test cases (" + ", ".join(IN_REGIME_TEST) + ")"),
          ("test_out_of_regime", "(c) out-of-regime test cases (test_outside_regime*, test_temporal_sr*)")]


def case_group(name, split):
    if vp.is_seen_ic_sr(name):
        return "seen_ic_field_sr_check"
    if split == "val":
        return "val"
    if name in IN_REGIME_TEST:
        return "test_in_regime"
    if name.startswith(OUT_OF_REGIME_PREFIXES):
        return "test_out_of_regime"
    return split if split == "train" else "other_test"


# ------------------------------------------------------------------ helpers
# The model's radius_pairs is now fast and exact; the evaluator's former monkeypatch is kept as an alias only.
fast_radius_pairs = vo.radius_pairs


def rel_l2(pred, true):
    return float(torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true).clamp_min(1e-30))


def field_stats(pred, true):
    """relL2, amplitude ratio and Pearson correlation of two [Q,3] fields."""
    p, t = pred.reshape(-1).double(), true.reshape(-1).double()
    tt = float((t * t).sum())
    amp = float((p * t).sum() / max(tt, 1e-30))
    pc, tc = p - p.mean(), t - t.mean()
    corr = float((pc * tc).sum() / ((pc.norm() * tc.norm()).clamp_min(1e-30)))
    return rel_l2(pred, true), amp, corr


def region_frac_of_total(pred, true, mask):
    """||(pred - true)[mask]|| / ||true|| (FULL-field norm): the region's share of the full relL2."""
    return float(torch.linalg.vector_norm((pred - true)[mask]) / torch.linalg.vector_norm(true).clamp_min(1e-30))


def n_rings_for(name):
    if "leapfrog" in name or "free_pair" in name:
        return 2
    return 1


def cluster_1d(v, n):
    """1-D k-means of the values v into n clusters (labels ordered by cluster position)."""
    centers = torch.quantile(v, torch.linspace(0.1, 0.9, n, device=v.device, dtype=v.dtype))
    for _ in range(20):
        label = (v.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(1)
        centers = torch.stack([v[label == k].mean() if (label == k).any() else centers[k] for k in range(n)])
    return label


def ring_labels_for(case):
    """Ring membership from frame 0 (particles keep their identity).

    Coaxial (leapfrogging) rings are clustered on z; the side-by-side free pair (rings at x = 0 and x = sep,
    both with axes along z) is clustered on x.
    """
    n = n_rings_for(case.name)
    x0 = case.x[0]
    if n == 1:
        return torch.zeros(case.n_particles, dtype=torch.long, device=x0.device), 1
    if "free_pair" in case.name:
        return cluster_1d(x0[:, 0], 2), 2
    return ring_membership(x0, 2), 2


def ring_state(x, gamma, label, n_rings):
    """Per ring: circulation-weighted centroid z, radius about the ring's OWN axis, impulse and spread.

    The axis passes through the ring's circulation centroid along its own impulse direction (z when the
    impulse vanishes).  vortex_frames.ring_diagnostics measures the radius from the global z axis, which is
    wrong for the off-axis ring of the free pair; for coaxial rings both definitions coincide.
    Returns (z, R, impulse 0.5 sum x x Gamma, spread = mean distance of the ring's particles from its
    circulation centroid), each a list over rings.
    """
    w = gamma.norm(dim=-1)
    zs, rs, imps, spreads = [], [], [], []
    ez = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
    for k in range(n_rings):
        m = label == k
        if not bool(m.any()):
            zs.append(float("nan")); rs.append(float("nan")); spreads.append(float("nan"))
            imps.append(torch.full((3,), float("nan"), dtype=torch.float64))
            continue
        wk = w[m] / w[m].sum().clamp_min(1e-30)
        c = (wk.unsqueeze(-1) * x[m]).sum(0)
        imp = 0.5 * torch.cross(x[m], gamma[m], dim=-1).sum(0)
        nrm = imp.norm()
        axis = imp / nrm if bool(torch.isfinite(nrm)) and float(nrm) > 1e-12 else ez
        d = x[m] - c
        rad = (d - (d @ axis).unsqueeze(-1) * axis).norm(dim=-1)
        zs.append(float(c[2]))
        rs.append(float((wk * rad).sum()))
        spreads.append(float(d.norm(dim=-1).mean()))
        imps.append(imp.double().cpu())
    return zs, rs, imps, spreads


def ring_descriptions(case, labels, n_rings):
    """Legend suffix per ring describing where it starts (frame 0), e.g. ' (starts at z=0.00, rel. |Γ| 1.00)'."""
    if n_rings == 1:
        return None
    out = []
    x0, g0 = case.x[0], case.gamma[0]
    for k in range(n_rings):
        m = labels == k
        c = x0[m].mean(0)
        circ = float(g0[m].norm(dim=-1).sum())
        out.append((c, circ))
    total = max(v[1] for v in out) or 1.0
    key = 0 if "free_pair" in case.name else 2
    return [f" (starts at {'xyz'[key]}={vp.fmt(float(c[key]), 2)}, rel. |Γ| {vp.fmt(circ / total, 2)})"
            for c, circ in out]


def ring_note(name):
    if "leapfrog" in name:
        return ("leapfrogging signature: the rear ring shrinks, speeds up and passes through the front ring, "
                "which grows and slows; z-curves cross, radii oscillate in antiphase")
    if "free_pair" in name:
        return ("counter-rotating side-by-side pair (rings at x = 0 and x = sep, clustered on x): the two rings "
                "travel in opposite z directions at nearly constant radius about their own axes")
    if "perturbed" in name:
        return "single ring with a mode-3 azimuthal perturbation (dt 4x smaller than training)"
    return "single ring: steady translation in z at nearly constant radius"


def outside_box_fraction(model, x, gamma):
    center = circulation_centroid(x, gamma)
    half = torch.tensor(model.box_half, device=x.device, dtype=x.dtype)
    return float(((x - center).abs() > half).any(-1).float().mean())


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def grid_plane(case):
    """Indices / coordinates of the grid plane nearest y=0: returns (j, X[nz,nx], Z[nz,nx], y, flat_idx[nz*nx])."""
    nz, ny, nx = case.grid_shape_zyx
    nodes = case.nodes.reshape(nz, ny, nx, 3)
    j = int(nodes[0, :, 0, 1].abs().argmin())
    plane = nodes[:, j]
    flat = (torch.arange(nz).view(nz, 1) * ny + j) * nx + torch.arange(nx).view(1, nx)
    return j, plane[..., 0].cpu().numpy(), plane[..., 2].cpu().numpy(), float(plane[0, 0, 1]), flat.reshape(-1)


# --------------------------------------------------------------- evaluation
@torch.no_grad()
def decode_field(model, x, gamma, sigma, nodes, device):
    enc = model.encode(x, gamma, sigma)
    u, J = decode_chunked(model, enc, nodes.to(device), chunk=CHUNK)
    return enc, u, vorticity_from_jacobian(J)


@torch.no_grad()
def decode_particles(model, enc, x, gamma, chunk=CHUNK):
    """Velocity, gradient and the predicted relative circulation rate (dGamma/dt)/|Gamma| at the particles."""
    us, Js, rates = [], [], []
    for s in range(0, len(x), chunk):
        u, J, h = model.decode(enc, x[s:s + chunk], return_features=True)
        g = gamma[s:s + chunk]
        if getattr(model, "gamma_dynamics", "rate") == "stretching":
            rates.append(gamma_rate_target(J, g))          # the model integrates J^T Gamma itself
        else:
            rates.append(model.gamma_rate(h, g))
        us.append(u); Js.append(J)
    return torch.cat(us), torch.cat(Js), torch.cat(rates)


TF_KEYS = ("u_rel_l2", "w_rel_l2", "u_amp", "w_amp", "u_corr", "w_corr",
           "u_err_vortex_frac_of_total", "u_err_far_frac_of_total",
           "w_err_vortex_frac_of_total", "w_err_far_frac_of_total",
           "particle_u_rel_l2", "particle_J_rel_l2", "particle_gamma_rate_rel_l2")


@torch.no_grad()
def teacher_forced(model, case, frames, device):
    out = {k: [] for k in TF_KEYS}
    out["frames"] = list(frames)
    out["time"] = [float(case.time[t]) for t in frames]
    for t in frames:
        x, g, s = case.frame(t)
        enc, u, w = decode_field(model, x, g, s, case.nodes, device)
        U, W = case.U[t].to(device), case.W[t].to(device)
        wmag = W.norm(dim=-1)
        thr = torch.quantile(wmag[:min(len(wmag), 16_000_000)], 0.95)
        core = wmag >= thr
        ru, au, cu = field_stats(u, U)
        rw, aw, cw = field_stats(w, W)
        out["u_rel_l2"].append(ru); out["u_amp"].append(au); out["u_corr"].append(cu)
        out["w_rel_l2"].append(rw); out["w_amp"].append(aw); out["w_corr"].append(cw)
        out["u_err_vortex_frac_of_total"].append(region_frac_of_total(u, U, core))
        out["u_err_far_frac_of_total"].append(region_frac_of_total(u, U, ~core))
        out["w_err_vortex_frac_of_total"].append(region_frac_of_total(w, W, core))
        out["w_err_far_frac_of_total"].append(region_frac_of_total(w, W, ~core))
        if bool(case.valid_p[t]):
            up, Jp, rate = decode_particles(model, enc, x, g)
            out["particle_u_rel_l2"].append(rel_l2(up, case.u_p[t]))
            out["particle_J_rel_l2"].append(rel_l2(Jp, case.J_p[t]))
            out["particle_gamma_rate_rel_l2"].append(rel_l2(rate, gamma_rate_target(case.J_p[t], g)))
        else:                                   # FLOWVPM writes zeros for frame 0
            for k in ("particle_u_rel_l2", "particle_J_rel_l2", "particle_gamma_rate_rel_l2"):
                out[k].append(float("nan"))
        del enc, u, w, U, W
    return out


def divergence_reason(x, g, labels, n_rings, S0, spread0, crit):
    """Truth-free divergence test of a predicted state; returns a reason string or None."""
    if not (torch.isfinite(x).all() and torch.isfinite(g).all()):
        return "non-finite values"
    S = float(g.norm(dim=-1).sum())
    dev = abs(S / S0 - 1.0)
    if dev > crit["sum_abs_gamma_rel_change"]:
        return f"sum|Γ| changed by {dev:.0%} from its start value"
    sp = ring_state(x, g, labels, n_rings)[3]
    ratio = max(a / b for a, b in zip(sp, spread0) if b > 0)
    if ratio > crit["ring_spread_ratio"]:
        return f"ring spread grew to {ratio:.2f}x its start value"
    return None


@torch.no_grad()
def rollout(model, case, start, steps, field_every, integrator, device, labels, n_rings, crit, substeps=None):
    s = case.sigma[start]
    x, g = case.x[start].clone(), case.gamma[start].clone()
    x_start = x.clone()
    _, r0, imp0, spread0 = ring_state(x, g, labels, n_rings)
    R0 = max(v for v in r0 if math.isfinite(v))
    imp_norm = float(sum(float(v.norm()) for v in imp0)) or 1.0
    S0 = float(g.norm(dim=-1).sum())
    rec = {k: [] for k in ("steps", "time", "position_error", "position_error_over_R", "persistence_error",
                           "persistence_error_over_R", "gamma_rel_l2", "impulse_drift", "true_impulse_drift",
                           "sum_abs_gamma_drift", "sum_abs_gamma_change", "true_sum_abs_gamma_change",
                           "ring_spread_ratio", "true_ring_spread_ratio",
                           "outside_box_fraction", "field_steps", "field_u_rel_l2", "field_w_rel_l2")}
    rec.update({"ring_z_true": [], "ring_z_pred": [], "ring_R_true": [], "ring_R_pred": []})
    I_true0 = sum(imp0)
    x_hist, g_hist = [x.cpu().numpy()], [g.cpu().numpy()]
    diverged_at, reason = None, None
    t_wall = time.time()
    for k in range(steps + 1):
        f = start + k
        if k > 0:
            if diverged_at is None:
                x, g, _ = model.step(x, g, s, float(case.dt[f - 1]), integrator, substeps=substeps)
                reason = divergence_reason(x, g, labels, n_rings, S0, spread0, crit)
                if reason is not None:
                    diverged_at = k      # this and all later steps are recorded as NaN
            if diverged_at is not None:
                x = torch.full_like(x, float("nan")); g = torch.full_like(g, float("nan"))
            x_hist.append(x.cpu().numpy())
            g_hist.append(g.cpu().numpy())
        xt, gt = case.x[f], case.gamma[f]
        err = float((x - xt).norm(dim=-1).mean())
        pers = float((x_start - xt).norm(dim=-1).mean())
        zt, rt, it, spt = ring_state(xt, gt, labels, n_rings)
        zp, rp, ip, spp = ring_state(x, g, labels, n_rings)
        St, Sp = float(gt.norm(dim=-1).sum()), float(g.norm(dim=-1).sum())
        rec["steps"].append(k); rec["time"].append(float(case.time[f]))
        rec["position_error"].append(err); rec["position_error_over_R"].append(err / R0)
        rec["persistence_error"].append(pers); rec["persistence_error_over_R"].append(pers / R0)
        rec["gamma_rel_l2"].append(rel_l2(g, gt))
        rec["impulse_drift"].append(float((sum(ip) - sum(it)).norm()) / imp_norm)
        rec["true_impulse_drift"].append(float((sum(it) - I_true0).norm()) / imp_norm)
        rec["sum_abs_gamma_drift"].append(abs(Sp - St) / max(St, 1e-30))
        rec["sum_abs_gamma_change"].append(Sp / S0 - 1.0)
        rec["true_sum_abs_gamma_change"].append(St / S0 - 1.0)
        rec["ring_spread_ratio"].append(max(a / b for a, b in zip(spp, spread0)))
        rec["true_ring_spread_ratio"].append(max(a / b for a, b in zip(spt, spread0)))
        rec["outside_box_fraction"].append(outside_box_fraction(model, x, g) if diverged_at is None else float("nan"))
        rec["ring_z_true"].append(zt); rec["ring_z_pred"].append(zp)
        rec["ring_R_true"].append(rt); rec["ring_R_pred"].append(rp)
        if (k % field_every == 0 or k == steps) and diverged_at is None:
            _, u, w = decode_field(model, x, g, s, case.nodes, device)
            rec["field_steps"].append(k)
            rec["field_u_rel_l2"].append(rel_l2(u, case.U[f].to(device)))
            rec["field_w_rel_l2"].append(rel_l2(w, case.W[f].to(device)))
            del u, w
    rec["R0"] = R0
    rec["start_frame"] = start
    rec["substeps"] = int(substeps or model.substeps)
    rec["diverged_at_step"] = diverged_at
    rec["diverged_reason"] = reason if diverged_at is not None else None
    rec["divergence_criteria"] = dict(crit)
    rec["steps_completed"] = (diverged_at - 1) if diverged_at is not None else steps
    rec["wall_time_s"] = time.time() - t_wall
    return rec, np.stack(x_hist), np.stack(g_hist)


# ------------------------------------------------------------------- plots
@torch.no_grad()
def slice_prediction(model, x, gamma, sigma, case, plane_idx, device):
    nodes = case.nodes[plane_idx.to(case.nodes.device)]
    enc = model.encode(x, gamma, sigma)
    u, J = decode_chunked(model, enc, nodes.to(device), chunk=CHUNK)
    return u.cpu().numpy(), vorticity_from_jacobian(J).cpu().numpy()


def make_slice_plot(path, title, model, case, frame, x_pred, g_pred, device, x_true_particles, stats):
    j, X, Z, y0, flat = grid_plane(case)
    nz, nx = X.shape
    u_p, w_p = slice_prediction(model, x_pred, g_pred, case.sigma[frame], case, flat, device)
    U = case.U[frame][flat.to(case.U.device)].cpu().numpy()
    W = case.W[frame][flat.to(case.W.device)].cpu().numpy()
    err_u = np.linalg.norm(u_p - U, axis=-1).reshape(nz, nx)
    xt = x_true_particles.cpu().numpy()
    pt = xt[np.abs(xt[:, 1]) < 0.15][:, [0, 2]]
    pp = None
    zs = [xt[:, 2]]
    if x_pred is not x_true_particles:
        xp = x_pred.cpu().numpy()
        ok = np.isfinite(xp).all(1)
        pp = xp[ok & (np.abs(xp[:, 1]) < 0.15)][:, [0, 2]]
        if ok.any():
            zs.append(np.clip(xp[ok, 2], xt[:, 2].min() - 3, xt[:, 2].max() + 3))
    zall = np.concatenate(zs)
    crop = (float(zall.min()) - 2.0, float(zall.max()) + 2.0)
    vp.plot_field_slice(path, title=title, X=X, Z=Z,
                        speed_true=np.linalg.norm(U, axis=-1).reshape(nz, nx),
                        speed_pred=np.linalg.norm(u_p, axis=-1).reshape(nz, nx),
                        wy_true=W[:, 1].reshape(nz, nx), wy_pred=w_p[:, 1].reshape(nz, nx), y_plane=y0,
                        particles_true=pt, particles_pred=pp, crop_z=crop, stats=stats, err_u=err_u)


def case_plots(model, case, res, x_hist, g_hist, labels, n_rings, plot_dir, device, field_frames_for_slices):
    name, label = case.name, vp.pretty_case(case.name)
    tf = res["teacher_forced"]
    tf_lookup = {f: i for i, f in enumerate(tf["frames"])}
    for f in field_frames_for_slices:
        x, g, _ = case.frame(f)
        i = tf_lookup.get(f)
        stats = {"u_rel_l2": tf["u_rel_l2"][i], "w_rel_l2": tf["w_rel_l2"][i]} if i is not None else None
        make_slice_plot(plot_dir / f"field_slice_{name}_f{f:03d}.png",
                        f"Teacher-forced field reconstruction from the TRUE particles: {label}, frame {f} "
                        f"(t = {float(case.time[f]):.3f})", model, case, f, x, g, device, x, stats)
    ro = res.get("rollout")
    if ro is None:
        return
    start = ro["start_frame"]
    valid_steps = [k for k in ro["steps"] if np.isfinite(x_hist[k]).all()]
    k_last = valid_steps[-1]
    f_last = start + k_last
    xp = torch.as_tensor(x_hist[k_last], device=device)
    gp = torch.as_tensor(g_hist[k_last], device=device)
    fi = ro["field_steps"].index(k_last) if k_last in ro["field_steps"] else None
    stats = {"u_rel_l2": ro["field_u_rel_l2"][fi], "w_rel_l2": ro["field_w_rel_l2"][fi]} if fi is not None else None
    make_slice_plot(plot_dir / f"field_slice_rollout_{name}.png",
                    f"Field decoded from ROLLED-OUT particles after {k_last} autoregressive steps: {label}, "
                    f"frame {f_last} (t = {float(case.time[f_last]):.3f})", model, case, f_last, xp, gp, device,
                    case.x[f_last], stats)
    # rollout snapshots
    ks = sorted(set(int(round(v)) for v in np.linspace(0, k_last, 5)))
    lab_np = labels.cpu().numpy()
    snaps = [{"step": k, "time": ro["time"][k], "x_true": case.x[start + k].cpu().numpy(), "x_pred": x_hist[k],
              "err": ro["position_error"][k]} for k in ks]
    desc = ring_descriptions(case, labels, n_rings)
    vp.plot_rollout_snapshots(plot_dir / f"rollout_snapshots_{name}.png", case_label=label, snapshots=snaps,
                              labels=lab_np, n_rings=n_rings, ref_radius=ro["R0"], ring_desc=desc,
                              diverged_at=ro["diverged_at_step"])
    vp.plot_ring_dynamics(plot_dir / f"ring_dynamics_{name}.png", case_label=label, time=np.array(ro["time"]),
                          z_true=np.array(ro["ring_z_true"], float), z_pred=np.array(ro["ring_z_pred"], float),
                          r_true=np.array(ro["ring_R_true"], float), r_pred=np.array(ro["ring_R_pred"], float),
                          ring_desc=desc,
                          diverged_time=ro["time"][ro["diverged_at_step"]] if ro["diverged_at_step"] else None,
                          note=ring_note(case.name))


@torch.no_grad()
def parity_plot(model, case, plot_dir, device):
    f = case.n_frames // 2
    if not bool(case.valid_p[f]):
        return
    x, g, s = case.frame(f)
    enc = model.encode(x, g, s)
    up, _, rate = decode_particles(model, enc, x, g)
    vp.plot_particle_parity(plot_dir / f"particle_velocity_parity_{case.name}.png",
                            case_label=vp.pretty_case(case.name), frame=f, time=float(case.time[f]),
                            u_true=case.u_p[f].cpu().numpy(), u_pred=up.cpu().numpy(),
                            rate_true=gamma_rate_target(case.J_p[f], g).cpu().numpy(), rate_pred=rate.cpu().numpy())


# ------------------------------------------------------------------ summary
def _mean(v):
    a = np.array([np.nan if x is None else x for x in v], float) if v else np.array([np.nan])
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def summarize(res):
    tf, ro = res["teacher_forced"], res.get("rollout") or {}
    last = lambda v: float(v[-1]) if v else float("nan")  # noqa: E731
    row = {"case": res["case"], "split": res["split"], "group": res["group"], "particles": res["particles"],
           "grid": "x".join(str(v) for v in res["grid_shape_zyx"]),
           "tf_u_rel_l2": _mean(tf["u_rel_l2"]), "tf_w_rel_l2": _mean(tf["w_rel_l2"]),
           "tf_u_err_vortex_frac_of_total": _mean(tf["u_err_vortex_frac_of_total"]),
           "tf_u_err_far_frac_of_total": _mean(tf["u_err_far_frac_of_total"]),
           "tf_w_err_vortex_frac_of_total": _mean(tf["w_err_vortex_frac_of_total"]),
           "tf_w_err_far_frac_of_total": _mean(tf["w_err_far_frac_of_total"]),
           "tf_u_amp": _mean(tf["u_amp"]), "tf_w_amp": _mean(tf["w_amp"]),
           "tf_u_corr": _mean(tf["u_corr"]), "tf_w_corr": _mean(tf["w_corr"]),
           "tf_particle_u_rel_l2": _mean(tf["particle_u_rel_l2"]),
           "tf_particle_J_rel_l2": _mean(tf["particle_J_rel_l2"]),
           "tf_particle_gamma_rate_rel_l2": _mean(tf["particle_gamma_rate_rel_l2"])}
    if ro:
        zt, zp = np.array(ro["ring_z_true"], float), np.array(ro["ring_z_pred"], float)
        rt, rp = np.array(ro["ring_R_true"], float), np.array(ro["ring_R_pred"], float)
        div = ro["diverged_at_step"] is not None
        row.update({
            "rollout_steps": ro["steps"][-1], "rollout_steps_completed": ro["steps_completed"],
            "rollout_diverged_at": ro["diverged_at_step"] if div else "",
            "rollout_diverged_reason": ro["diverged_reason"] or "",
            "ro_diverged": 1.0 if div else 0.0,
            "R0": ro["R0"],
            "ro_final_pos_err_over_R": last(ro["position_error_over_R"]),
            "ro_mean_pos_err_over_R": _mean(ro["position_error_over_R"][1:]),
            "persistence_final_pos_err_over_R": last(ro["persistence_error_over_R"]),
            "ro_final_gamma_rel_l2": last(ro["gamma_rel_l2"]),
            "ro_final_ring_z_err": float(np.nanmean(np.abs(zp[-1] - zt[-1]))) if np.isfinite(zp[-1]).any() else float("nan"),
            "ro_final_ring_R_err": float(np.nanmean(np.abs(rp[-1] - rt[-1]))) if np.isfinite(rp[-1]).any() else float("nan"),
            "ro_final_impulse_drift": last(ro["impulse_drift"]),
            "ro_final_sum_abs_gamma_drift": last(ro["sum_abs_gamma_drift"]),
            # field metrics stop at divergence: "final" is NaN then, like the other final values
            "ro_final_field_u_rel_l2": last(ro["field_u_rel_l2"]) if not div else float("nan"),
            "ro_final_field_w_rel_l2": last(ro["field_w_rel_l2"]) if not div else float("nan"),
            "ro_max_outside_box_fraction": float(np.nanmax(np.array(ro["outside_box_fraction"], float))),
            "ro_wall_time_s": ro["wall_time_s"],
        })
    return row


GROUP_ABBREV = {"val": "val", "test_in_regime": "in-reg", "test_out_of_regime": "OOR",
                "seen_ic_field_sr_check": "seenIC", "train": "train", "other_test": "test"}


def print_table(rows):
    cols = [("case", 46, "s"), ("group_str", 6, "s"), ("tf_u_rel_l2", 7, ".3f"), ("tf_w_rel_l2", 7, ".3f"),
            ("tf_particle_u_rel_l2", 7, ".3f"), ("tf_particle_gamma_rate_rel_l2", 7, ".3f"), ("steps_str", 7, "s"),
            ("ro_final_pos_err_over_R", 9, ".3f"), ("persistence_final_pos_err_over_R", 8, ".3f"),
            ("ro_final_gamma_rel_l2", 7, ".3f"), ("ro_final_sum_abs_gamma_drift", 7, ".3f"),
            ("ro_final_field_u_rel_l2", 7, ".3f"), ("ro_max_outside_box_fraction", 6, ".2f")]
    heads = ["case", "group", "TF u", "TF ω", "TF u_p", "TF Γ̇", "steps", "RO |dx|/R", "persist", "RO Γ",
             "RO Σ|Γ|", "RO u", "outBox"]
    print("  ".join(h.ljust(w) if i < 2 else h.rjust(w) for i, (h, (_, w, _f)) in enumerate(zip(heads, cols))))
    for r in rows:
        r = dict(r)
        div = r.get("rollout_diverged_at", "")
        r["steps_str"] = f"DIV@{div}" if div != "" else str(r.get("rollout_steps", "-"))
        r["group_str"] = GROUP_ABBREV.get(r.get("group"), r.get("group", ""))
        cells = []
        for i, (k, w, fmt) in enumerate(cols):
            v = r.get(k, "")
            try:
                s = format(v, fmt) if v != "" else "-"
            except (TypeError, ValueError):
                s = str(v)
            s = s[:w]
            cells.append(s.ljust(w) if i < 2 else s.rjust(w))
        print("  ".join(cells))


# metric key, short name, plain-language explanation (used by summary.md and the printed summary)
OVERALL_METRICS = [
    ("tf_u_rel_l2", "TF grid velocity relL2",
     "error of the velocity field decoded from the TRUE particles, relative to the field's size (0 = perfect, 1 = "
     "as large as the field itself)"),
    ("tf_w_rel_l2", "TF grid vorticity relL2",
     "same for the vorticity (curl of the predicted velocity gradient); harder than velocity because vorticity "
     "is concentrated in thin cores"),
    ("tf_u_err_vortex_frac_of_total", "TF velocity error share: vortex region",
     "part of the full-grid velocity relL2 that comes from the 5% of nodes with the strongest vorticity"),
    ("tf_u_err_far_frac_of_total", "TF velocity error share: far field",
     "part of the full-grid velocity relL2 that comes from the remaining 95% of nodes (squares of the two "
     "shares add up to the square of the full relL2)"),
    ("tf_particle_u_rel_l2", "TF particle velocity relL2",
     "error of the velocity the model predicts AT the particles, which is what moves them in a rollout"),
    ("tf_particle_J_rel_l2", "TF particle velocity-gradient relL2",
     "error of the predicted velocity gradient at the particles (includes the self-induced term)"),
    ("tf_particle_gamma_rate_rel_l2", "TF particle circulation-rate relL2",
     "error of the predicted rate of change of each particle's circulation (vortex stretching), which "
     "drives the circulation in a rollout"),
    ("ro_diverged", "rollout diverged (fraction of cases)",
     "share of rollouts that blew up ({rule}); final rollout values below are averaged over the remaining "
     "cases only"),
    ("rollout_steps_completed", "rollout steps completed",
     "how many saved-frame intervals the model was rolled out before stopping or diverging"),
    ("ro_final_pos_err_over_R", "rollout final position error / R0",
     "mean distance between predicted and true particles at the end of the rollout, in units of the ring "
     "radius (0.1 = one tenth of a ring radius off)"),
    ("persistence_final_pos_err_over_R", "persistence baseline / R0",
     "the same error if the particles had simply stayed where they started; the model must be well below this"),
    ("ro_final_gamma_rel_l2", "rollout final circulation relL2",
     "error of the predicted particle circulation vectors at the end of the rollout"),
    ("ro_final_sum_abs_gamma_drift", "rollout final total-|Γ| drift vs truth",
     "relative difference between predicted and true total circulation magnitude (conservation check)"),
    ("ro_final_impulse_drift", "rollout final impulse drift vs truth",
     "difference between predicted and true linear impulse, relative to the rings' impulse "
     "(conservation check; the impulse of an inviscid flow is constant)"),
    ("ro_final_field_u_rel_l2", "rollout final grid velocity relL2",
     "velocity-field error when the field is decoded from the model's OWN rolled-out particles"),
]


def overall_summary(rows, meta):
    """Markdown text with mean metrics per case group (seen-IC super-resolution case excluded)."""
    lines = ["# VortexGINO evaluation: overall summary", "",
             f"checkpoint `{meta['checkpoint']}` (epoch {meta['epoch']}), integrator {meta['integrator']} with "
             f"{meta['substeps']} substeps per saved interval" + (" -- QUICK smoke test" if meta.get("quick") else ""),
             "", "Means over the cases of each group (finite values only; n = cases contributing / cases in group). "
             "TF = teacher-forced (decoded from the TRUE particles); rollout = autoregressive from the start frame.",
             ""]
    present = [g for g, _ in GROUPS if any(r["group"] == g for r in rows)]
    head = "| metric | " + " | ".join(dict(GROUPS)[g] for g in present) + " |"
    lines += [head, "|" + "---|" * (len(present) + 1)]
    for key, short, _ in OVERALL_METRICS:
        cells = []
        for g in present:
            vals = np.array([r.get(key, np.nan) if r.get(key, "") != "" else np.nan
                             for r in rows if r["group"] == g], float)
            n_ok = int(np.isfinite(vals).sum())
            cells.append(f"{np.nanmean(vals):.4g} (n={n_ok}/{len(vals)})" if n_ok else f"- (n=0/{len(vals)})")
        lines.append(f"| {short} | " + " | ".join(cells) + " |")
    missing = [lab for g, lab in GROUPS if g not in present]
    if missing:
        lines += ["", "Not evaluated in this run: " + "; ".join(missing) + "."]
    lines += ["", "## Cases per group", ""]
    for g, lab in GROUPS:
        names = [r["case"] for r in rows if r["group"] == g]
        if names:
            lines.append(f"* {lab}: " + ", ".join(names))
    excluded = [r for r in rows if r["group"] not in dict(GROUPS)]
    if excluded:
        lines += ["", "Excluded from the averages above (still in metrics.json / summary.csv):"]
        for r in excluded:
            why = ("seen IC: field super-resolution check only (same initial condition as the training case "
                   "leapfrog_N320_d1p000_g1p000; only the output grid is finer)"
                   if r["group"] == "seen_ic_field_sr_check" else f"group '{r['group']}'")
            lines.append(f"* {r['case']}: {why}")
    lines += ["", "## What the metrics mean", ""]
    rule = vp.divergence_rule_text([{"rollout": {"divergence_criteria": meta["divergence_criteria"]}}])
    for key, short, expl in OVERALL_METRICS:
        lines.append(f"* **{short}** (`{key}`): {expl.format(rule=rule)}.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(HERE / "runs/vortex_gino/final_model.pt"))
    ap.add_argument("--data-root", default=str(HERE / "processed_data/vortex_rings_v2"))
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case names (default: all in --splits)")
    ap.add_argument("--rollout-steps", type=int, default=None, help="default: to the end of each case")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--field-every", type=int, default=10,
                    help="teacher-forced field metrics every N frames; rollout field metrics every N steps")
    ap.add_argument("--integrator", default=None, help="default: the checkpoint's training integrator (rk4)")
    ap.add_argument("--substeps", type=int, default=None, help="RK substeps per saved interval (default: model's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 case, 20 rollout steps, 3 field frames")
    ap.add_argument("--diverge-gamma-sum", type=float, default=0.5,
                    help="rollout counts as diverged when sum|Gamma| deviates from its start value by more than "
                         "this fraction (then NaN)")
    ap.add_argument("--diverge-spread", type=float, default=3.0,
                    help="rollout counts as diverged when the mean distance of a ring's particles from its "
                         "circulation centroid exceeds this multiple of its start value (then NaN)")
    ap.add_argument("--exact-pairs", action="store_true",
                    help="no-op (kept for compatibility): the model's neighbour search is already fast and exact")
    args = ap.parse_args()
    crit = {"non_finite": True, "sum_abs_gamma_rel_change": args.diverge_gamma_sum,
            "ring_spread_ratio": args.diverge_spread}

    device = torch.device(args.device)
    ck_path = Path(args.checkpoint)
    out = Path(args.out) if args.out else ck_path.parent / f"evaluation_{ck_path.stem}"
    plot_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = VortexGINO(ck["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])          # drops neuralop's '_metadata' itself
    model.eval()
    train_cfg = ck.get("train_config") or {}
    integrator = args.integrator or train_cfg.get("integrator", "rk4")
    substeps = int(args.substeps or model.substeps)
    print(f"[model] {ck_path}  epoch={ck.get('epoch')}  score={ck.get('score')}  integrator={integrator} "
          f"x{substeps} substeps  gamma_dynamics={model.gamma_dynamics}  box_half={model.box_half}")

    root = Path(args.data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = [e for e in manifest["cases"] if e["split"] in args.splits and (not args.cases or e["name"] in args.cases)]
    entries.sort(key=lambda e: (args.splits.index(e["split"]), e["name"]))
    if args.quick:
        entries = entries[:1]
    if not entries:
        raise SystemExit(f"no cases match splits={args.splits} cases={args.cases}")
    print(f"[data] {len(entries)} case(s): {[e['name'] for e in entries]}")

    results, rows = [], []
    for e in entries:
        t0 = time.time()
        # Particles on the device, grid fields on CPU (moved per frame) so many cases fit in little memory.
        case = CaseData(root / e["file"], device, True, "cpu")
        case.meta = e
        F = case.n_frames
        start = min(max(0, args.start_frame), F - 2)
        steps = (F - 1 - start) if args.rollout_steps is None else min(args.rollout_steps, F - 1 - start)
        if args.quick:
            steps = min(steps, 20)
            frames = sorted({0, F // 2, F - 1})
        else:
            frames = sorted(set(range(0, F, args.field_every)) | {F - 1})
        labels, n_rings = ring_labels_for(case)
        group = case_group(case.name, e["split"])
        res = {"case": case.name, "split": e["split"], "group": group, "particles": case.n_particles,
               "grid_shape_zyx": list(case.grid_shape_zyx), "categories": e.get("evaluation_categories", []),
               "n_rings": n_rings, "dt_mean": float(case.dt.mean())}
        if group == "seen_ic_field_sr_check":
            res["note"] = vp.SEEN_IC_NOTE
        res["teacher_forced"] = teacher_forced(model, case, frames, device)
        ro, x_hist, g_hist = rollout(model, case, start, steps, args.field_every, integrator, device,
                                     labels, n_rings, crit, substeps)
        res["rollout"] = ro
        row = summarize(res)
        rows.append(row)
        print(f"[case] {vp.pretty_case(case.name)}: TF u={row['tf_u_rel_l2']:.3f} w={row['tf_w_rel_l2']:.3f} "
              f"u_p={row['tf_particle_u_rel_l2']:.3f} Γrate={row['tf_particle_gamma_rate_rel_l2']:.3f} | rollout "
              f"{steps} steps{'' if ro['diverged_at_step'] is None else ' (DIVERGED at step %d: %s)' % (ro['diverged_at_step'], ro['diverged_reason'])}: "
              f"|dx|/R0={row['ro_final_pos_err_over_R']:.3f} (persistence {row['persistence_final_pos_err_over_R']:.3f}) "
              f"Γ={row['ro_final_gamma_rel_l2']:.3f} Σ|Γ| drift={row['ro_final_sum_abs_gamma_drift']:.3f} "
              f"field u={row['ro_final_field_u_rel_l2']:.3f} outBox={row['ro_max_outside_box_fraction']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not args.no_plots:
            slice_frames = sorted({0, F // 2, F - 1})
            case_plots(model, case, res, x_hist, g_hist, labels, n_rings, plot_dir, device, slice_frames)
            parity_plot(model, case, plot_dir, device)
        results.append(res)
        del case, x_hist, g_hist
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {"checkpoint": str(ck_path), "epoch": ck.get("epoch"), "score": ck.get("score"),
            "integrator": integrator, "substeps": substeps, "gamma_dynamics": model.gamma_dynamics,
            "start_frame": args.start_frame, "field_every": args.field_every,
            "quick": args.quick, "box_half_extent": model.box_half, "divergence_criteria": crit,
            "groups": {g: lab for g, lab in GROUPS}}
    (out / "metrics.json").write_text(json.dumps(nan_to_none({"meta": meta, "cases": results}), indent=1))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary_md = overall_summary(rows, meta)
    (out / "summary.md").write_text(summary_md)
    if not args.no_plots:
        vp.plot_error_vs_step(plot_dir / "error_vs_step.png", results)
        vp.plot_field_error_vs_time(plot_dir / "field_error_vs_time.png", results)
        vp.plot_training_curves(plot_dir / "training_curves.png", ck.get("history", []),
                                title=f"Training history of {ck_path.name} (epoch {ck.get('epoch')})",
                                operator_epochs=train_cfg.get("operator_epochs"), total_epochs=train_cfg.get("epochs"),
                                rollout_steps=train_cfg.get("val_rollout_steps"),
                                rollout_start=train_cfg.get("val_rollout_start"))
    print()
    print_table(rows)
    print("\nTF = teacher-forced from true particles (mean over frames); Γ̇ = particle circulation rate; RO = "
          "autoregressive rollout at the final step (DIV@k: diverged at step k, final values NaN); |dx|/R = mean "
          "position error / initial ring radius; persist = same for frozen particles; Σ|Γ| = total-circulation drift "
          "vs truth; outBox = max fraction of particles outside the latent box.\ngroup: val = validation, in-reg = "
          "in-regime test, OOR = out-of-regime test, seenIC = " + vp.SEEN_IC_NOTE + " (excluded from the summary).")
    print("\n" + summary_md)
    if device.type == "cuda":
        print(f"[gpu] peak allocated memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    print(f"[done] metrics: {out / 'metrics.json'}\n       summary: {out / 'summary.csv'}\n"
          f"       overall: {out / 'summary.md'}" + ("" if args.no_plots else f"\n       plots:   {plot_dir}"))


if __name__ == "__main__":
    main()
