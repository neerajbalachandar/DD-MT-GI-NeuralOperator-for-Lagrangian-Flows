"""Training for the co-moving VortexGINO (joint field reconstruction + particle dynamics).

Curriculum
----------
Phase 1 (operator): supervise the shared particle->(u, J) operator directly with
    * grid velocity U and grid vorticity W (antisymmetric part of predicted J),
    * exact VPM particle velocity u_p and gradient J_p (dense, at the particles).
Phase 2 (joint): keep the operator losses and add coupled-RK4 transition losses
    on 1..H-step autoregressive windows (BPTT with per-step checkpointing).

All losses are *relative* squared L2 errors so that slow far-field nodes,
fast core particles, small circulation changes and tiny displacements are all
weighted by their own magnitude (the legacy pipeline normalised positions by the
global coordinate std, so a 100%-wrong displacement looked like a 0.1% error).
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from gino.data.vortex_frames import ring_diagnostics, ring_membership
from gino.data.vortex_raw import winckelmans_velocity_jacobian


def vorticity_from_jacobian(J):
    return torch.stack((J[..., 2, 1] - J[..., 1, 2], J[..., 0, 2] - J[..., 2, 0], J[..., 1, 0] - J[..., 0, 1]), -1)


def rel_sq(pred, true, eps=1e-12):
    return (pred - true).square().sum() / true.square().sum().clamp_min(eps)


def rel_l2(pred, true, eps=1e-12):
    return float(torch.sqrt(rel_sq(pred, true, eps)))


def gamma_rate_target(J, gamma):
    """Exact relative circulation rate (J^T Gamma)/|Gamma| (FLOWVPM transposed scheme)."""
    return torch.einsum("nji,nj->ni", J, gamma) / gamma.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def estimate_scales(cases, model, n_frames=12):
    """Data scales.  gamma_scale normalises the *latent* deposit |Gamma|/cell_volume to O(1)."""
    us, Js, rates, deposits, dts = [], [], [], [], []
    for case in cases:
        dts.append(case.dt)
        frames = torch.linspace(1, case.n_frames - 1, n_frames).long().tolist()
        for t in frames:
            x, g, _ = case.frame(t)
            if bool(case.valid_p[t]):
                us.append(case.u_p[t].reshape(-1))
                Js.append(case.J_p[t].reshape(-1))
                rates.append(gamma_rate_target(case.J_p[t], g).reshape(-1))
            center = x.new_tensor(0.0) + (g.norm(dim=-1, keepdim=True) * x).sum(0) / g.norm(dim=-1).sum()
            half, spacing = model.box(x.device, x.dtype)
            idx = torch.round((x - center + half) / spacing).long()
            res = torch.tensor(model.res, device=x.device)
            ok = ((idx >= 0) & (idx < res)).all(-1)
            flat = (idx[ok, 2] * model.res[1] + idx[ok, 1]) * model.res[0] + idx[ok, 0]
            grid = torch.zeros(int(np.prod(model.res)), device=x.device)
            grid.index_add_(0, flat, g.norm(dim=-1)[ok] / spacing.prod())
            deposits.append(grid[grid > 0])
    dep = torch.cat(deposits)
    return {"velocity_scale": float(torch.cat(us).square().mean().sqrt()),
            "gradient_scale": float(torch.cat(Js).square().mean().sqrt()),
            "gamma_rate_scale": float(torch.cat(rates).square().mean().sqrt()),
            "gamma_scale": float(dep.square().mean().sqrt()),
            "dt_ref": float(torch.cat(dts).median())}


def decode_chunked(model, enc, q, chunk=16384):
    us, Js = [], []
    for s in range(0, len(q), chunk):
        u, J = model.decode(enc, q[s:s + chunk])
        us.append(u)
        Js.append(J)
    return torch.cat(us), torch.cat(Js)


class VortexTrainer:
    def __init__(self, model, train_cases, val_cases, cfg, run_dir, device):
        self.model, self.train_cases, self.val_cases = model, train_cases, val_cases
        self.cfg, self.device = cfg, device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Deterministic scatter-adds make each rollout step bitwise reproducible, which
        # activation checkpointing needs (the radius-neighbour sets must not change on
        # recomputation).  grid_sample's backward has no deterministic kernel -> warn only.
        torch.use_deterministic_algorithms(True, warn_only=True)
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            (no_decay if p.ndim <= 1 or "near" in name else decay).append(p)
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(cfg.get("weight_decay", 1e-5))},
                                      {"params": no_decay, "weight_decay": 0.0}],
                                     lr=float(cfg.get("learning_rate", 1e-3)))
        self.epochs = int(cfg.get("epochs", 60))
        self.steps_per_epoch = int(cfg.get("steps_per_epoch", 300))
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup = int(cfg.get("warmup_steps", 300))
        self.history = []
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.w = cfg.get("loss_weights", {})
        self.ring_labels = {c.name: ring_membership(c.x[0], 2) for c in val_cases}
        # Exponential moving average of the weights.  At the peak learning rate the raw Adam
        # iterate oscillates (end-of-epoch weights were 2-3x worse than the epoch-mean loss,
        # with a spurious mean velocity bias); validation, checkpoints and evaluation use EMA.
        self.buffer, self.buffer_steps = [], []
        self.ema_decay = float(cfg.get("ema_decay", 0.998))
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self):
        for e, p in zip(self.ema.parameters(), self.model.parameters()):
            e.lerp_(p.detach(), 1.0 - self.ema_decay)
        for e, b in zip(self.ema.buffers(), self.model.buffers()):
            e.copy_(b)

    # ---------------------------------------------------------------- schedule
    def lr_at(self, step):
        base = float(self.cfg.get("learning_rate", 1e-3))
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        p = (step - self.warmup) / max(1, self.total_steps - self.warmup)
        floor = float(self.cfg.get("min_lr_ratio", 0.02))
        return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def horizon_at(self, epoch):
        sched = self.cfg.get("rollout_horizon_schedule", [[0, 1]])
        h = 1
        for start, value in sched:
            if epoch >= int(start):
                h = int(value)
        return h

    # ------------------------------------------------------------------ losses
    def operator_loss(self, case, t, state=None):
        """Operator losses on frame t of a case, or on an explicit off-trajectory ``state``.

        Off-trajectory sets (random perturbations or on-policy rollout states) use the exact
        regularised Biot-Savart targets of that particle set.
        """
        m, cfg = self.model, self.cfg
        aug = cfg.get("augmentation", {}) or {}
        if state is not None:
            x, g, s = state
            augmented = True
        else:
            x, g, s = case.frame(t)
            augmented = self.rng.random() < float(aug.get("probability", 0.0))
        if augmented and state is None:
            # Off-trajectory particle sets like those a rollout produces (scattered core
            # particles, perturbed circulation); targets are the exact regularised Biot-Savart
            # field of the perturbed set, i.e. what the VPM solver itself would compute.
            level = float(self.rng.random())
            x = x + level * float(aug.get("position_std_over_sigma", 0.3)) * s.unsqueeze(-1) * torch.randn_like(x)
            g = g * (1.0 + level * float(aug.get("gamma_rel_std", 0.05)) * torch.randn_like(g[:, :1])) \
                + level * float(aug.get("gamma_rel_std", 0.05)) * g.norm(dim=-1, keepdim=True) * torch.randn_like(g)
        enc = m.encode(x, g, s)
        gi = self.sample_grid_nodes(case, x, enc)
        q = case.nodes[gi]
        u, J = m.decode(enc, q)
        if augmented:
            with torch.no_grad():
                U, Jg = winckelmans_velocity_jacobian(q, x, g, s)
                W = vorticity_from_jacobian(Jg)
        else:
            U = case.U[t][gi.to(case.U.device)].to(x.device)
            W = case.W[t][gi.to(case.W.device)].to(x.device)
        terms = {"grid_u": rel_sq(u, U), "grid_w": rel_sq(vorticity_from_jacobian(J), W)}
        if augmented or bool(case.valid_p[t]):
            n_p = min(int(cfg.get("particle_queries", 1024)), case.n_particles)
            pi = torch.randperm(case.n_particles, device=x.device)[:n_p]
            up, Jp, hp = m.decode(enc, x[pi], return_features=True)
            if augmented:
                with torch.no_grad():
                    up_t, Jp_t = winckelmans_velocity_jacobian(x[pi], x, g, s)
            else:
                up_t, Jp_t = case.u_p[t][pi], case.J_p[t][pi]
            terms["particle_u"] = rel_sq(up, up_t)
            terms["particle_J"] = rel_sq(Jp, Jp_t)
            terms["gamma_rate"] = rel_sq(m.gamma_rate(hp, g[pi]), gamma_rate_target(Jp_t, g[pi]))
        total = sum(float(self.w.get(k, 1.0)) * v for k, v in terms.items())
        return total, terms

    @torch.no_grad()
    def refresh_onpolicy_buffer(self, weights):
        """Roll the EMA model out from a random training frame and store the visited state."""
        op = self.cfg.get("onpolicy", {}) or {}
        ci = int(self.rng.choice(len(self.train_cases), p=weights))
        case = self.train_cases[ci]
        max_steps = int(op.get("max_steps", 80))
        t0 = int(self.rng.integers(0, max(1, case.n_frames - 6)))
        k = int(self.rng.integers(3, max(4, min(max_steps, case.n_frames - 1 - t0) + 1)))
        x, g, s = case.frame(t0)
        g0 = g.norm(dim=-1).sum()
        for j in range(k):
            x, g, _ = self.ema.step(x, g, s, float(case.dt[min(t0 + j, case.n_frames - 2)]),
                                    self.cfg.get("integrator", "rk4"))
            if not (torch.isfinite(x).all() and torch.isfinite(g).all()):
                return
            if abs(float(g.norm(dim=-1).sum() / g0) - 1.0) > float(op.get("max_gamma_drift", 0.6)):
                break
        self.buffer.append((ci, x.clone(), g.clone(), s))
        if len(self.buffer) > int(op.get("buffer_size", 64)):
            self.buffer.pop(0)
        self.buffer_steps.append(k)

    def sample_grid_nodes(self, case, x, enc):
        """Importance-sample grid nodes: near particles / inside the latent box / anywhere.

        Uniform sampling over the 5x5x17 grid puts ~60% of queries outside the co-moving box
        (0.1-0.5% of the energy) and only ~80 nodes per frame near the cores.
        """
        n = min(int(self.cfg.get("grid_queries", 8192)), len(case.nodes))
        fr = self.cfg.get("grid_sampling_fractions", [0.5, 0.35, 0.15])
        nodes = case.nodes
        with torch.no_grad():
            dmin = torch.cat([torch.cdist(nodes[s:s + 8192], x).min(1).values for s in range(0, len(nodes), 8192)])
            near = dmin < float(self.cfg.get("near_particle_distance", 0.5))
            inbox = ((nodes - enc["center"]).abs() <= enc["half"]).all(-1) & ~near
            pools = [torch.nonzero(near, as_tuple=True)[0], torch.nonzero(inbox, as_tuple=True)[0],
                     torch.arange(len(nodes), device=nodes.device)]
            chosen, remaining = [], n
            for k, (pool, f) in enumerate(zip(pools, fr)):
                take = remaining if k == 2 else min(int(round(f * n)), len(pool))
                if take > 0 and len(pool):
                    chosen.append(pool[torch.randint(len(pool), (take,), device=nodes.device)])
                    remaining -= take
        return torch.cat(chosen)

    def dynamics_loss(self, case, t, horizon):
        m = self.model
        integrator = self.cfg.get("integrator", "rk4")
        x, g, s = case.frame(t)
        loss_x, loss_g = 0.0, 0.0

        def one_step(xc, gc, dt):
            xn, gn, _ = m.step(xc, gc, s, dt, integrator)
            return xn, gn

        for k in range(horizon):
            dt = float(case.dt[t + k])
            if horizon > 1 and torch.is_grad_enabled():
                x, g = checkpoint(one_step, x, g, dt, use_reentrant=False)
            else:
                x, g = one_step(x, g, dt)
            xt, gt = case.x[t + k + 1], case.gamma[t + k + 1]
            dx = (case.x[t + k + 1] - case.x[t + k]).square().sum().clamp_min(1e-12)
            dg = (case.gamma[t + k + 1] - case.gamma[t + k]).square().sum().clamp_min(1e-12)
            loss_x = loss_x + (x - xt).square().sum() / dx
            loss_g = loss_g + (g - gt).square().sum() / dg
        loss_x, loss_g = loss_x / horizon, loss_g / horizon
        total = float(self.w.get("step_x", 1.0)) * loss_x + float(self.w.get("step_gamma", 0.3)) * loss_g
        return total, {"step_x": loss_x, "step_gamma": loss_g}

    # -------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, joint):
        m = self.ema
        frames = self.cfg.get("val_frames", [10, 50, 90, 130, 170])
        rows = {k: [] for k in ("grid_u", "grid_w", "particle_u", "particle_J", "gamma_rate", "step_x", "step_gamma")}
        for case in self.val_cases:
            for t in frames:
                x, g, s = case.frame(t)
                enc = m.encode(x, g, s)
                u, J = decode_chunked(m, enc, case.nodes)
                rows["grid_u"].append(rel_l2(u, case.U[t].to(u.device)))
                rows["grid_w"].append(rel_l2(vorticity_from_jacobian(J), case.W[t].to(u.device)))
                up, Jp, hp = m.decode(enc, x, return_features=True)
                if not bool(case.valid_p[t]):      # FLOWVPM writes zero particle u/J at frame 0
                    continue
                rows["particle_u"].append(rel_l2(up, case.u_p[t]))
                rows["particle_J"].append(rel_l2(Jp, case.J_p[t]))
                rows["gamma_rate"].append(rel_l2(m.gamma_rate(hp, g), gamma_rate_target(case.J_p[t], g)))
                if joint and t + 1 < case.n_frames:
                    xn, gn, _ = m.step(x, g, s, float(case.dt[t]), self.cfg.get("integrator", "rk4"))
                    rows["step_x"].append(rel_l2(xn - x, case.x[t + 1] - x))
                    rows["step_gamma"].append(rel_l2(gn - g, case.gamma[t + 1] - g))
        out = {f"val_{k}": float(np.mean(v)) for k, v in rows.items() if v}
        if joint:
            out.update(self.validate_rollout())
        return out

    @torch.no_grad()
    def validate_rollout(self):
        steps = int(self.cfg.get("val_rollout_steps", 40))
        start = int(self.cfg.get("val_rollout_start", 20))
        errs, zerr, rerr, gerr, sdrift = [], [], [], [], []
        for case in self.val_cases:
            if case.n_particles > int(self.cfg.get("val_rollout_max_particles", 3000)):
                continue
            n = min(steps, case.n_frames - 1 - start)
            x, g, s = case.frame(start)
            label = self.ring_labels[case.name]
            for k in range(n):
                x, g, _ = self.ema.step(x, g, s, float(case.dt[start + k]), self.cfg.get("integrator", "rk4"))
            if not (torch.isfinite(x).all() and torch.isfinite(g).all()):
                x = torch.nan_to_num(x, nan=1e3, posinf=1e3, neginf=-1e3)
                g = torch.nan_to_num(g, nan=1e3, posinf=1e3, neginf=-1e3)
            xt, gt = case.x[start + n], case.gamma[start + n]
            errs.append(min(float((x - xt).norm(dim=-1).mean()), 10.0))
            gerr.append(min(rel_l2(g, gt), 10.0))
            sdrift.append(min(abs(float(g.norm(dim=-1).sum() / gt.norm(dim=-1).sum()) - 1.0), 10.0))
            pred = ring_diagnostics(x, g, label, 2)
            true = ring_diagnostics(xt, gt, label, 2)
            zerr.append(np.mean([abs(a[0] - b[0]) for a, b in zip(pred, true)]))
            rerr.append(np.mean([abs(a[1] - b[1]) for a, b in zip(pred, true)]))
        return {"val_rollout_mean_position_error": float(np.mean(errs)),
                "val_rollout_gamma_rel_l2": float(np.mean(gerr)),
                "val_rollout_sum_abs_gamma_drift": float(np.mean(sdrift)),
                "val_rollout_ring_z_error": float(np.mean(zerr)),
                "val_rollout_ring_radius_error": float(np.mean(rerr))}

    # -------------------------------------------------------------------- loop
    def score(self, metrics, joint):
        s = (metrics["val_grid_u"] + metrics["val_particle_u"] + 0.25 * metrics["val_particle_J"]
             + 0.5 * metrics["val_gamma_rate"])
        if joint:
            s += metrics.get("val_step_x", 0.0) + 0.25 * metrics.get("val_step_gamma", 0.0) \
                + metrics.get("val_rollout_mean_position_error", 0.0) \
                + 0.5 * metrics.get("val_rollout_gamma_rel_l2", 0.0) + metrics.get("val_rollout_sum_abs_gamma_drift", 0.0)
        return s

    def save(self, name, extra):
        torch.save({"model_state_dict": self.ema.state_dict(), "raw_model_state_dict": self.model.state_dict(),
                    "ema_decay": self.ema_decay, "model_config": self.model.cfg,
                    "train_config": self.cfg, "history": self.history,
                    "optimizer_state_dict": self.opt.state_dict(),
                    "rng_state": self.rng.bit_generator.state, **extra}, self.run_dir / name)

    def fit(self, start_epoch=0, global_step=0):
        cfg = self.cfg
        batch = int(cfg.get("frames_per_step", 4))
        op_epochs = int(cfg.get("operator_epochs", 20))
        clip = float(cfg.get("grad_clip_norm", 1.0))
        # On resume, keep the best score of the current phase instead of overwriting best_model.pt.
        phase_now = "joint" if start_epoch >= op_epochs else "operator"
        prior = [h["score"] for h in self.history if h.get("phase") == phase_now and np.isfinite(h["score"])]
        best = min(prior or [float("inf")])
        weights = np.array([c.n_frames - 1 for c in self.train_cases], dtype=float)
        weights /= weights.sum()
        for epoch in range(start_epoch, self.epochs):
            joint = epoch >= op_epochs
            horizon = self.horizon_at(epoch) if joint else 0
            t0 = time.time()
            sums = {}
            self.model.train()
            for it in range(self.steps_per_epoch):
                lr = self.lr_at(global_step)
                for group in self.opt.param_groups:
                    group["lr"] = lr
                self.opt.zero_grad(set_to_none=True)
                op = cfg.get("onpolicy", {}) or {}
                if float(op.get("probability", 0.0)) > 0 and it % int(op.get("refresh_every", 20)) == 0:
                    self.refresh_onpolicy_buffer(weights)
                for b in range(batch):
                    if self.buffer and self.rng.random() < float(op.get("probability", 0.0)):
                        ci, bx, bg, bs = self.buffer[int(self.rng.integers(0, len(self.buffer)))]
                        loss, terms = self.operator_loss(self.train_cases[ci], None, state=(bx, bg, bs))
                        terms = {f"onpolicy_{k}": v for k, v in terms.items()}
                    else:
                        case = self.train_cases[self.rng.choice(len(self.train_cases), p=weights)]
                        t = int(self.rng.integers(0, case.n_frames))
                        loss, terms = self.operator_loss(case, t)
                    (loss / batch).backward()
                    for k, v in terms.items():
                        sums.setdefault(k, []).append(float(v))
                if joint and self.rng.random() < float(cfg.get("dynamics_probability", 1.0)):
                    case = self.train_cases[self.rng.choice(len(self.train_cases), p=weights)]
                    t = int(self.rng.integers(0, case.n_frames - horizon))
                    loss, terms = self.dynamics_loss(case, t, horizon)
                    (float(cfg.get("dynamics_weight", 1.0)) * loss).backward()
                    for k, v in terms.items():
                        sums.setdefault(k, []).append(float(v))
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if torch.isfinite(gnorm):
                    self.opt.step()
                    self.update_ema()
                sums.setdefault("grad_norm", []).append(float(gnorm))
                global_step += 1
            train = {f"train_{k}": float(np.mean(v)) for k, v in sums.items()}
            val = self.validate(joint)
            score = self.score(val, joint)
            row = {"epoch": epoch + 1, "phase": "joint" if joint else "operator", "horizon": horizon,
                   "lr": lr, "time_s": time.time() - t0, "score": score, **train, **val}
            self.history.append(row)
            (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
            extra = {"epoch": epoch + 1, "global_step": global_step, "score": score}
            self.save("last_model.pt", extra)
            # Scores are only comparable within a phase (the joint score adds dynamics terms),
            # so the best checkpoint is tracked per phase; best_model.pt follows the latest phase.
            if joint and epoch == op_epochs:
                best = float("inf")
            if score < best:
                best = score
                self.save(f"best_{row['phase']}.pt", extra)
                self.save("best_model.pt", extra)
            msg = " ".join(f"{k.replace('train_', 't_').replace('val_', 'v_')}={v:.4g}" for k, v in row.items()
                           if isinstance(v, float) and k not in ("lr", "time_s"))
            print(f"epoch {epoch + 1:03d} [{row['phase']} H={horizon}] lr={lr:.2e} {row['time_s']:.0f}s | {msg}", flush=True)
        return self.history
