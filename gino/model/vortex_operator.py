"""Co-moving, physics-structured GINO for vortex-particle flows.

One shared operator maps a set of vortex particles P = {x_p, Gamma_p, sigma_p}
to the velocity u(q) and velocity gradient J(q) = du/dq at arbitrary query
points q.  The *same* operator therefore

  (1) reconstructs the Eulerian velocity field (queries = grid nodes), and
  (2) drives the Lagrangian dynamics (queries = the particles themselves):
          dx/dt = u(x),   dGamma/dt = J(x)^T Gamma   (FLOWVPM "transposed" scheme)
      integrated with a coupled RK4 step plus a small learned correction.

Architecture (GINO = GNO encoder -> FNO processor -> GNO decoder):

* Co-moving frame: coordinates are taken relative to the circulation-weighted
  particle centroid, and the latent FNO grid is a fixed physical box around it.
  The rings travel ~11 units in z; an absolute frame wastes most latent nodes and
  must learn the same physics at every location.
* Encoder: learned particle-to-grid deposition on a 3x3x3 stencil.  Messages are
  *linear in Gamma* and *summed* (superposition), so particle-resolution changes
  (N=1440/2880/5760 with Gamma_p ~ 1/N) give the same latent field.
* Processor: neuralop FNO with domain padding (the box is not periodic).
* Decoder: trilinear latent sample + an equivariant learned near-field kernel
      V_c(q) = sum_{|q-x_p|<r_c} a_c(|q-x_p|/sigma_p) (q-x_p) x Gamma_p / sigma_p^3
  (the Biot-Savart structure with a learned radial profile) whose exact gradient
  is computed analytically, resolving the sub-grid vortex cores (sigma ~ 0.1 vs a
  latent spacing ~ 0.25).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def circulation_centroid(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    w = gamma.norm(dim=-1, keepdim=True)
    return (w * x).sum(0) / w.sum().clamp_min(1e-12)


def fourier_features(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    if n_freq <= 0:
        return x
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    ang = x.unsqueeze(-1) * freqs
    return torch.cat((x, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)), dim=-1)


def _mlp(sizes, act=nn.GELU, zero_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if zero_last:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


_LEVI = torch.zeros(3, 3, 3)
_LEVI[0, 1, 2] = _LEVI[1, 2, 0] = _LEVI[2, 0, 1] = 1.0
_LEVI[0, 2, 1] = _LEVI[2, 1, 0] = _LEVI[1, 0, 2] = -1.0


def radius_pairs(queries: torch.Tensor, sources: torch.Tensor, radius: float, chunk: int = 8192):
    """All (query, source) index pairs closer than ``radius`` (no gradient).

    Queries outside the source bounding box (grown by ``radius``) are skipped, candidates
    come from the fast matmul cdist with a slightly enlarged radius, and membership is
    decided by the exact difference norm.  The final pair set is therefore bitwise
    reproducible, which activation checkpointing requires when it recomputes a step.
    """
    with torch.no_grad():
        lo = sources.min(0).values - radius
        hi = sources.max(0).values + radius
        near = torch.nonzero(((queries >= lo) & (queries <= hi)).all(-1), as_tuple=True)[0]
        qi, sj = [], []
        for start in range(0, len(near), chunk):
            idx = near[start:start + chunk]
            d = torch.cdist(queries[idx], sources)
            i, j = torch.nonzero(d < radius * 1.01 + 1e-6, as_tuple=True)
            gi = idx[i]
            keep = (queries[gi] - sources[j]).norm(dim=-1) < radius
            qi.append(gi[keep])
            sj.append(j[keep])
    if not qi:
        e = torch.zeros(0, dtype=torch.long, device=queries.device)
        return e, e
    return torch.cat(qi), torch.cat(sj)


class NearFieldKernel(nn.Module):
    """Learned, rotation-equivariant, Gamma-linear short-range kernel with exact gradient."""

    def __init__(self, channels: int = 4, n_rbf: int = 12, cutoff: float = 0.3, rho_max: float = 4.0):
        super().__init__()
        self.channels, self.cutoff = int(channels), float(cutoff)
        self.register_buffer("centers", torch.linspace(0.0, rho_max, n_rbf))
        self.width = float(rho_max / (n_rbf - 1)) * 1.0
        # Initialise channel 0 close to the regularised Biot-Savart profile so the
        # learned kernel starts from a sensible short-range interaction.
        self.theta = nn.Parameter(0.01 * torch.randn(channels, n_rbf))

    def radial(self, rho):
        z = (rho.unsqueeze(-1) - self.centers) / self.width
        phi = torch.exp(-z * z)
        dphi = phi * (-2.0 * z / self.width)
        return phi @ self.theta.t(), dphi @ self.theta.t()

    def forward(self, queries, x, gamma, sigma, need_grad=True, pairs=None):
        """Return V [Q, C, 3] and (optionally) T = dV/dq [Q, C, 3, 3]."""
        Q, C = len(queries), self.channels
        qi, sj = radius_pairs(queries, x, self.cutoff) if pairs is None else pairs
        V = queries.new_zeros(Q, C, 3)
        T = queries.new_zeros(Q, C, 9) if need_grad else None
        if len(qi) == 0:
            return V, (T.view(Q, C, 3, 3) if need_grad else None)
        r = queries[qi] - x[sj]
        dist = r.norm(dim=-1).clamp_min(1e-9)
        s = sigma[sj]
        rho = dist / s
        a, da = self.radial(rho)
        inside = (dist < self.cutoff).to(r.dtype)
        env = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * inside
        denv = -0.5 * math.pi / self.cutoff * torch.sin(math.pi * dist / self.cutoff) * inside
        g = gamma[sj]
        rxg = torch.cross(r, g, dim=-1)
        inv_s3 = (1.0 / s ** 3).unsqueeze(-1)
        coef = a * env.unsqueeze(-1) * inv_s3                    # [P, C]
        V.index_add_(0, qi, coef.unsqueeze(-1) * rxg.unsqueeze(1))
        if need_grad:
            levi = _LEVI.to(r)
            M = torch.einsum("ijl,pl->pij", levi, g)             # d(r x G)_i / dr_j
            rhat = r / dist.unsqueeze(-1)
            dcoef = (da / s.unsqueeze(-1) * env.unsqueeze(-1) + a * denv.unsqueeze(-1)) * inv_s3
            outer = (rxg.unsqueeze(-1) * rhat.unsqueeze(-2)).reshape(-1, 9)
            T.index_add_(0, qi, coef.unsqueeze(-1) * M.reshape(-1, 1, 9)
                         + dcoef.unsqueeze(-1) * outer.unsqueeze(1))
            T = T.view(Q, C, 3, 3)
        return V, T


class StencilEncoder(nn.Module):
    """Learned particle-to-grid deposition (Gamma-linear, summed over particles)."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.channels = channels
        self.kernel = _mlp([4, hidden, hidden, 3 * channels])
        offs = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("offsets", offs)  # (dx, dy, dz) node offsets

    def forward(self, idx_coords, gamma_hat, sigma_cells, res_xyz):
        """idx_coords: particle positions in latent index units (x, y, z); returns [C, nz, ny, nx]."""
        nx, ny, nz = res_xyz
        nearest = torch.round(idx_coords).long()
        node = nearest.unsqueeze(1) + self.offsets.unsqueeze(0)           # [N, 27, 3]
        delta = node.to(idx_coords.dtype) - idx_coords.unsqueeze(1)       # [N, 27, 3]
        valid = ((node[..., 0] >= 0) & (node[..., 0] < nx) & (node[..., 1] >= 0) & (node[..., 1] < ny)
                 & (node[..., 2] >= 0) & (node[..., 2] < nz))
        window = torch.cos(math.pi * delta.clamp(-1.5, 1.5) / 3.0).pow(2).prod(-1)  # C1 deposit, 0 at |d|=1.5
        feats = torch.cat((delta, sigma_cells.view(-1, 1, 1).expand(-1, delta.shape[1], 1)), -1)
        K = self.kernel(feats).view(*delta.shape[:2], self.channels, 3)
        msg = torch.einsum("nkcj,nj->nkc", K, gamma_hat) * (window * valid).unsqueeze(-1)
        flat = (node[..., 2] * ny + node[..., 1]) * nx + node[..., 0]
        flat = torch.where(valid, flat, torch.zeros_like(flat))
        grid = msg.new_zeros(nx * ny * nz, self.channels)
        grid.index_add_(0, flat.reshape(-1), msg.reshape(-1, self.channels))
        return grid.t().reshape(self.channels, nz, ny, nx)


class VortexGINO(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        from neuralop.models import FNO
        self.cfg = dict(cfg)
        self.box_half = [float(v) for v in cfg.get("box_half_extent", [2.8, 2.8, 3.3])]
        self.res = [int(v) for v in cfg.get("latent_resolution", [24, 24, 28])]       # (nx, ny, nz)
        width = int(cfg.get("hidden_channels", 64))
        enc_c = int(cfg.get("encoder_channels", 32))
        self.n_freq = int(cfg.get("query_fourier_frequencies", 3))
        self.encoder = StencilEncoder(enc_c, int(cfg.get("encoder_hidden", 64)))
        modes = [int(v) for v in cfg.get("fno_modes", [12, 12, 14])]                    # (x, y, z)
        self.fno = FNO(n_modes=(modes[2], modes[1], modes[0]), in_channels=enc_c, out_channels=width,
                       hidden_channels=width, n_layers=int(cfg.get("fno_layers", 4)),
                       positional_embedding="grid", domain_padding=float(cfg.get("domain_padding", 0.25)))
        nf = self.cfg.get("near_field", {}) or {}
        self.use_near = bool(nf.get("enabled", True))
        self.nf_channels = int(nf.get("channels", 4))
        if self.use_near:
            self.near = NearFieldKernel(self.nf_channels, int(nf.get("rbf", 12)), float(nf.get("cutoff", 0.3)),
                                        float(nf.get("rho_max", 4.0)))
            self.near_mix = nn.Parameter(torch.zeros(self.nf_channels))
        q_dim = 3 + 6 * self.n_freq + 1
        nf_dim = (12 * self.nf_channels) if self.use_near else 0
        mlp_w = int(cfg.get("decoder_hidden", 128))
        n_dec = int(cfg.get("decoder_layers", 3))
        self.decoder = _mlp([width + q_dim + nf_dim] + [mlp_w] * n_dec + [12])
        with torch.no_grad():  # start near u = J = 0 (relative losses start at ~1, no gradient spikes)
            self.decoder[-1].weight.mul_(0.05)
            self.decoder[-1].bias.zero_()
        self.residual_head = _mlp([width + q_dim + nf_dim + 4] + [mlp_w, mlp_w] + [6], zero_last=True)
        # Circulation dynamics.  The exact transposed stretching J^T Gamma is a ~3% residue of
        # |J||Gamma| (near cancellation), so a learned J with ~10% error gives spurious
        # exponential growth rates ~2x larger than the true ones.  'rate' predicts the
        # relative rate dGamma/dt / |Gamma| directly (supervised with J^T Gamma / |Gamma|),
        # whose error scales with the small rate itself; 'stretching' keeps J^T Gamma.
        self.gamma_dynamics = str(cfg.get("gamma_dynamics", "rate"))
        self.gamma_rate_head = _mlp([width + q_dim + nf_dim + 3] + [mlp_w, mlp_w] + [3])
        self.register_buffer("gamma_rate_scale", torch.tensor(1.0))
        # One saved interval spans several solver steps and core particles rotate ~1.5 rad
        # per interval; a single RK4 step over it collapses the cores even with exact physics.
        self.substeps = int(cfg.get("substeps", 2))
        # FLOWVPM corrected-Pedrizzetti relaxation (rlxf per solver step; per_interval = solver
        # steps per saved interval at dt_ref, scaled with dt).
        self.relaxation = dict(cfg.get("relaxation", {}) or {})
        # Physical scales (set from training data via set_scales; stored in the state dict).
        self.register_buffer("gamma_scale", torch.tensor(1.0))
        self.register_buffer("velocity_scale", torch.tensor(1.0))
        self.register_buffer("gradient_scale", torch.tensor(1.0))
        self.register_buffer("residual_scale", torch.ones(6))
        self.register_buffer("dt_ref", torch.tensor(0.0129))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # neuralop's FNO writes a non-tensor '_metadata' entry into the parent state dict.
        state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}
        if "dt_ref" not in state_dict:
            state_dict["dt_ref"] = self.dt_ref.clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd.pop("_metadata", None)
        return sd

    # ----------------------------------------------------------------- geometry
    def box(self, device, dtype):
        half = torch.tensor(self.box_half, device=device, dtype=dtype)
        res = torch.tensor(self.res, device=device, dtype=dtype)
        return half, (2.0 * half) / (res - 1.0)

    def set_scales(self, gamma_scale, velocity_scale, gradient_scale, residual_scale=None,
                   gamma_rate_scale=None, dt_ref=None):
        self.gamma_scale.fill_(float(gamma_scale))
        self.velocity_scale.fill_(float(velocity_scale))
        self.gradient_scale.fill_(float(gradient_scale))
        if residual_scale is not None:
            self.residual_scale.copy_(torch.as_tensor(residual_scale, dtype=self.residual_scale.dtype))
        if gamma_rate_scale is not None:
            self.gamma_rate_scale.fill_(float(gamma_rate_scale))
        if dt_ref is not None:
            self.dt_ref.fill_(float(dt_ref))

    # ------------------------------------------------------------------ encoder
    def encode(self, x, gamma, sigma):
        """Encode one particle set -> latent dict (co-moving frame)."""
        center = circulation_centroid(x, gamma)
        half, spacing = self.box(x.device, x.dtype)
        idx = (x - center + half) / spacing                      # latent index units (x, y, z)
        cell_volume = spacing.prod()
        gamma_hat = gamma / (self.gamma_scale * cell_volume)
        sigma_cells = sigma / spacing.mean()
        grid = self.encoder(idx, gamma_hat, sigma_cells, self.res).unsqueeze(0)
        latent = self.fno(grid)                                   # [1, W, nz, ny, nx]
        return {"center": center, "latent": latent, "x": x, "gamma": gamma, "sigma": sigma,
                "half": half, "spacing": spacing}

    # ------------------------------------------------------------------ decoder
    def _query_features(self, enc, q):
        rel = (q - enc["center"]) / enc["half"]                  # [-1, 1] inside the box
        inside = rel.clamp(-1.0, 1.0)
        sample = F.grid_sample(enc["latent"], inside.view(1, -1, 1, 1, 3), mode="bilinear",
                               align_corners=True).view(enc["latent"].shape[1], -1).t()
        outside = (rel - inside).norm(dim=-1, keepdim=True)        # distance beyond the box
        return torch.cat((sample, fourier_features(inside, self.n_freq), outside), dim=-1)

    def decode(self, enc, q, need_gradient=True, pairs=None, return_features=False):
        """Velocity [Q,3] and gradient J [Q,3,3] (J[i,j] = du_i/dq_j) in physical units."""
        feats = [self._query_features(enc, q)]
        u_near = J_near = None
        if self.use_near:
            V, T = self.near(q, enc["x"], enc["gamma"], enc["sigma"], need_grad=True, pairs=pairs)
            u_near = torch.einsum("c,qci->qi", self.near_mix, V)
            J_near = torch.einsum("c,qcij->qij", self.near_mix, T)
            feats.append(V.reshape(len(q), -1) / self.velocity_scale)
            feats.append(T.reshape(len(q), -1) / self.gradient_scale)
        h = torch.cat(feats, dim=-1)
        out = self.decoder(h)
        u = out[:, :3] * self.velocity_scale
        J = (out[:, 3:] * self.gradient_scale).view(-1, 3, 3)
        if u_near is not None:
            u = u + u_near
            J = J + J_near
        if return_features:
            return u, J, h
        return u, J

    # ----------------------------------------------------------------- dynamics
    def gamma_rate(self, h, gamma):
        """Predicted relative circulation rate (dGamma/dt)/|Gamma| at particles [N,3] (1/time)."""
        gamma_unit = gamma / gamma.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.gamma_rate_head(torch.cat((h, gamma_unit), dim=-1)) * self.gamma_rate_scale

    def velocity_rhs(self, x, gamma, sigma, return_jacobian=False):
        enc = self.encode(x, gamma, sigma)
        u, J, h = self.decode(enc, x, return_features=True)
        dgamma = self._dgamma(J, h, gamma)
        if return_jacobian:
            return u, dgamma, h, enc, J
        return u, dgamma, h, enc

    def _dgamma(self, J, h, gamma):
        if self.gamma_dynamics == "stretching":
            return torch.einsum("nji,nj->ni", J, gamma)      # (Gamma . grad^T) u  (transposed scheme)
        return gamma.norm(dim=-1, keepdim=True) * self.gamma_rate(h, gamma)

    def relax(self, gamma, J, sigma):
        """FLOWVPM corrected Pedrizzetti relaxation with the model's own vorticity estimate.

        Gamma <- [(1-f) Gamma + f |Gamma| w_hat] / sqrt(b2) rotates each particle strength toward
        the local vorticity direction while preserving |Gamma| (FLOWVPM applies it every solver
        step with f = 0.3).  w is the curl of J without the particle's self-induced term, as in
        FLOWVPM.  It keeps Gamma aligned with omega, which is what keeps the stretching term
        (a ~3% residue of |J||Gamma| when aligned) from feeding back on itself.
        """
        f = float(self.relaxation.get("rlxf", 0.3))
        s3 = (2.5 / (4.0 * math.pi * sigma ** 3)).view(-1, 1)
        # curl of the self term 2.5/(4 pi s^3) [Gamma]_x is 2 * 2.5/(4 pi s^3) * Gamma
        w = torch.stack((J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]), -1) - 2.0 * s3 * gamma
        nw = w.norm(dim=-1, keepdim=True)
        ng = gamma.norm(dim=-1, keepdim=True)
        ok = (nw > 1e-12) & (ng > 1e-12)
        w_hat = w / nw.clamp_min(1e-12)
        cos = (gamma * w_hat).sum(-1, keepdim=True) / ng.clamp_min(1e-12)
        b2 = (1.0 - 2.0 * (1.0 - f) * f * (1.0 - cos)).clamp_min(1e-12)
        relaxed = ((1.0 - f) * gamma + f * ng * w_hat) / b2.sqrt()
        return torch.where(ok, relaxed, gamma)

    def _substep(self, x, gamma, sigma, h, integrator, first=None, n_relax=0):
        if first is None:
            first = self.velocity_rhs(x, gamma, sigma, return_jacobian=True)
        k1x, k1g, h1, enc, J1 = first
        if n_relax > 0:
            # Relaxation is applied at the start of the substep with the J already computed for the
            # first RK stage (no extra operator evaluation).  This approximates FLOWVPM, which relaxes
            # after each solver step with a freshly evaluated J: here k1x and the features h1 still
            # come from the pre-relaxation Gamma (~0.3% difference), and repeated relaxations reuse J.
            for _ in range(n_relax):
                gamma = self.relax(gamma, J1, sigma)
            k1g = self._dgamma(J1, h1, gamma)
        if integrator == "euler":
            return x + h * k1x, gamma + h * k1g
        if integrator == "heun":
            k2x, k2g, _, _ = self.velocity_rhs(x + h * k1x, gamma + h * k1g, sigma)
            return x + 0.5 * h * (k1x + k2x), gamma + 0.5 * h * (k1g + k2g)
        if integrator == "rk4":
            k2x, k2g, _, _ = self.velocity_rhs(x + 0.5 * h * k1x, gamma + 0.5 * h * k1g, sigma)
            k3x, k3g, _, _ = self.velocity_rhs(x + 0.5 * h * k2x, gamma + 0.5 * h * k2g, sigma)
            k4x, k4g, _, _ = self.velocity_rhs(x + h * k3x, gamma + h * k3g, sigma)
            return (x + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
                    gamma + h / 6.0 * (k1g + 2 * k2g + 2 * k3g + k4g))
        raise ValueError(f"unknown integrator {integrator!r}")

    def step(self, x, gamma, sigma, dt, integrator="rk4", use_residual=True, substeps=None):
        """Advance particles by one saved interval ``dt``; returns (x_next, gamma_next, aux).

        The interval is integrated with ``substeps`` coupled RK4 steps (every stage re-encodes
        the moved particle set), followed by one learned residual correction per interval.
        """
        dt = float(dt)
        n = int(substeps or self.substeps)
        h = dt / n
        # Relaxations per saved interval (FLOWVPM relaxes once per solver step; the solver step is
        # assumed to scale with dt), spread over the substeps so the total is exact (some may get 0).
        total_relax = 0
        if self.relaxation.get("enabled", False):
            total_relax = int(round(float(self.relaxation.get("per_interval", 4)) * dt / float(self.dt_ref)))
        counts = [(k + 1) * total_relax // n - k * total_relax // n for k in range(n)]
        first = self.velocity_rhs(x, gamma, sigma, return_jacobian=True)
        k1x, k1g, h1, enc, _ = first
        xn, gn = self._substep(x, gamma, sigma, h, integrator, first, counts[0])
        for k in range(1, n):
            xn, gn = self._substep(xn, gn, sigma, h, integrator, None, counts[k])
        aux = {"velocity": k1x, "dgamma": k1g, "encoding": enc}
        if use_residual:
            gamma_unit = gamma / gamma.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dt_feat = torch.full_like(gamma_unit[:, :1], dt / float(self.dt_ref))
            r = self.residual_head(torch.cat((h1, gamma_unit, dt_feat), dim=-1)) * self.residual_scale
            xn = xn + dt * r[:, :3]
            # Circulation corrections are relative rates (|Gamma_p| ~ 1/N varies with resolution).
            gn = gn + dt * gamma.norm(dim=-1, keepdim=True) * r[:, 3:]
            aux["residual"] = r
        return xn, gn, aux
