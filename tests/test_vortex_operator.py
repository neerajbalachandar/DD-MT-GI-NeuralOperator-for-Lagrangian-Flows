"""Unit tests for the co-moving VortexGINO operator (gino/model/vortex_operator.py).

All tests run on CPU with a tiny configuration.  NOTE: never call ``model.to(dtype)`` /
``model.double()`` on a VortexGINO: it casts neuralop's complex spectral weights to real
(discarding the imaginary part) and breaks the FNO.  The model is used in its native
float32; float64 is used only for the standalone NearFieldKernel and with a stubbed encoder.
"""
from __future__ import annotations

import math

import pytest
import torch

from gino.model.vortex_operator import NearFieldKernel, VortexGINO, circulation_centroid
from gino.training.vortex_trainer import decode_chunked, vorticity_from_jacobian

TINY_CFG = {
    "box_half_extent": [2.8, 2.8, 3.3],
    "latent_resolution": [8, 8, 10],
    "hidden_channels": 16,
    "encoder_channels": 8,
    "encoder_hidden": 16,
    "fno_modes": [4, 4, 4],
    "fno_layers": 2,
    "decoder_hidden": 32,
    "decoder_layers": 2,
    "query_fourier_frequencies": 2,
    "near_field": {"enabled": True, "channels": 2, "rbf": 8, "cutoff": 0.3, "rho_max": 4.0},
}


def _levi(dtype=torch.float64):
    e = torch.zeros(3, 3, 3, dtype=dtype)
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1.0
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1.0
    return e


def _skew(w):
    """[w]_x with [w]_x v = w x v."""
    return torch.einsum("ijk,j->ik", _levi(w.dtype), w)


def _random_rotation(gen, dtype=torch.float64):
    A = torch.randn(3, 3, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    model = VortexGINO({**TINY_CFG, **overrides}).eval()
    with torch.no_grad():
        # near_mix starts at zero; make the near-field path contribute so it is exercised.
        model.near_mix.copy_(torch.tensor([0.4, -0.2]))
        model.decoder[-1].weight.mul_(20.0)  # undo the 0.05 init damping -> non-trivial outputs
    return model


def _ring_particles(gen, n=240, R=1.0, core=0.08, z0=0.0, circulation=1.0, dtype=torch.float32):
    th = torch.rand(n, generator=gen, dtype=dtype) * 2 * math.pi
    rr = R + core * torch.randn(n, generator=gen, dtype=dtype)
    zz = z0 + core * torch.randn(n, generator=gen, dtype=dtype)
    x = torch.stack((rr * torch.cos(th), rr * torch.sin(th), zz), -1)
    tangent = torch.stack((-torch.sin(th), torch.cos(th), torch.zeros_like(th)), -1)
    gamma = circulation * (2 * math.pi * R / n) * tangent
    sigma = torch.full((n,), 0.1, dtype=dtype)
    return x, gamma, sigma


def _particles_and_queries(seed=1):
    gen = torch.Generator().manual_seed(seed)
    x, gamma, sigma = _ring_particles(gen)
    q_far = (torch.rand(96, 3, generator=gen) * 2 - 1) * torch.tensor([2.0, 2.0, 2.0])
    q_near = x[:48] + 0.05 * torch.randn(48, 3, generator=gen)  # inside the near-field cutoff
    return x, gamma, sigma, torch.cat((q_far, q_near))


# --------------------------------------------------------------------------- near field
def _near_kernel(seed=0):
    torch.manual_seed(seed)
    k = NearFieldKernel(channels=3, n_rbf=8, cutoff=0.3, rho_max=4.0).double()
    with torch.no_grad():
        k.theta.copy_(torch.randn_like(k.theta))
    return k


def _near_inputs(seed=3, n=60, nq=40):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, generator=gen, dtype=torch.float64) * 0.5
    gamma = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    sigma = 0.08 + 0.04 * torch.rand(n, generator=gen, dtype=torch.float64)
    q = torch.rand(nq, 3, generator=gen, dtype=torch.float64) * 0.5
    return x, gamma, sigma, q


def test_near_field_analytic_gradient_matches_autograd():
    kern = _near_kernel()
    x, gamma, sigma, q = _near_inputs()
    q = q.clone().requires_grad_(True)
    V, T = kern(q, x, gamma, sigma, need_grad=True)
    Q, C = V.shape[:2]
    assert T.shape == (Q, C, 3, 3)
    assert V.abs().max() > 1e-3, "kernel produced no interaction; test would be vacuous"
    # autograd dV[q, c, i]/dq[q, j]  (each query only depends on its own coordinates)
    T_auto = torch.zeros_like(T)
    for c in range(C):
        for i in range(3):
            (g,) = torch.autograd.grad(V[:, c, i].sum(), q, retain_graph=True)
            T_auto[:, c, i, :] = g
    torch.testing.assert_close(T.detach(), T_auto, rtol=1e-8, atol=1e-10)


def test_near_field_rotation_equivariance():
    kern = _near_kernel(seed=1)
    x, gamma, sigma, q = _near_inputs(seed=4)
    R = _random_rotation(torch.Generator().manual_seed(7))
    with torch.no_grad():
        V, T = kern(q, x, gamma, sigma)
        Vr, Tr = kern(q @ R.T, x @ R.T, gamma @ R.T, sigma)
    assert V.abs().max() > 1e-3
    torch.testing.assert_close(Vr, torch.einsum("ij,qcj->qci", R, V), rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(Tr, torch.einsum("ik,qckl,jl->qcij", R, T, R), rtol=1e-9, atol=1e-10)


# --------------------------------------------------------------------------- full model
def test_superposition_resolution_invariance():
    model = _tiny_model()
    x, gamma, sigma, q = _particles_and_queries()
    with torch.no_grad():
        enc1 = model.encode(x, gamma, sigma)
        u1, J1 = model.decode(enc1, q)
        x2, g2, s2 = torch.cat((x, x)), torch.cat((0.5 * gamma, 0.5 * gamma)), torch.cat((sigma, sigma))
        enc2 = model.encode(x2, g2, s2)
        u2, J2 = model.decode(enc2, q)
    assert u1.abs().max() > 1e-3 and J1.abs().max() > 1e-3
    torch.testing.assert_close(enc2["center"], enc1["center"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(enc2["latent"], enc1["latent"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(u2, u1, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(J2, J1, rtol=1e-4, atol=1e-4)


def test_decode_translation_equivariance():
    model = _tiny_model()
    x, gamma, sigma, q = _particles_and_queries(seed=2)
    shift = torch.tensor([0.0, 0.0, 3.7])
    with torch.no_grad():
        enc = model.encode(x, gamma, sigma)
        u, J = model.decode(enc, q)
        enc_s = model.encode(x + shift, gamma, sigma)
        u_s, J_s = model.decode(enc_s, q + shift)
    torch.testing.assert_close(enc_s["center"], enc["center"] + shift, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(enc_s["latent"], enc["latent"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(u_s, u, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(J_s, J, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_grid_sample_axis_order(axis):
    """Latent layout is [1, C, nz, ny, nx]; sampling a coordinate ramp must return that coordinate."""
    model = _tiny_model()
    x, gamma, sigma, _ = _particles_and_queries()
    with torch.no_grad():
        enc = dict(model.encode(x, gamma, sigma))
    center, half, spacing = enc["center"], enc["half"], enc["spacing"]
    nx, ny, nz = model.res
    axes = [center[k] - half[k] + spacing[k] * torch.arange(model.res[k], dtype=x.dtype) for k in range(3)]
    torch.testing.assert_close(axes[0][-1], center[0] + half[0])  # node grid spans center +/- half
    Z, Y, X = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")  # [nz, ny, nx]
    latent = torch.zeros_like(enc["latent"])
    assert latent.shape == (1, TINY_CFG["hidden_channels"], nz, ny, nx)
    latent[0, 0] = (X, Y, Z)[axis]
    enc["latent"] = latent
    gen = torch.Generator().manual_seed(11 + axis)
    q = center + (torch.rand(200, 3, generator=gen) * 2 - 1) * half * 0.98
    feats = model._query_features(enc, q)
    # trilinear interpolation of a linear ramp is exact
    torch.testing.assert_close(feats[:, 0], q[:, axis], rtol=0, atol=2e-5)
    other = [k for k in range(3) if k != axis]
    for k in other:  # and must NOT correlate with the other coordinates (x<->z swap guard)
        assert (feats[:, 0] - q[:, k]).abs().max() > 0.1


def test_decode_chunked_matches_decode():
    model = _tiny_model()
    x, gamma, sigma, q = _particles_and_queries()
    with torch.no_grad():
        enc = model.encode(x, gamma, sigma)
        u, J = model.decode(enc, q)
        uc, Jc = decode_chunked(model, enc, q, chunk=37)
    torch.testing.assert_close(uc, u, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(Jc, J, rtol=1e-5, atol=1e-6)


def test_circulation_centroid_weights_by_gamma_norm():
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    g = torch.tensor([[3.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    torch.testing.assert_close(circulation_centroid(x, g), torch.tensor([0.5, 0.0, 0.0]))


# --------------------------------------------------------------------------- dynamics
def _stub_dynamics(model, A, c):
    """Replace encode/decode by the exact linear field u = A (q - c), J = A."""

    def encode(x, gamma, sigma):
        return {}

    def decode(enc, q, need_gradient=True, pairs=None, return_features=False):
        Ad = A.to(q.dtype)
        u = (q - c.to(q.dtype)) @ Ad.T
        J = Ad.expand(len(q), 3, 3)
        if return_features:
            return u, J, q.new_zeros(len(q), 1)
        return u, J

    model.encode = encode
    model.decode = decode
    return model


def test_rk4_step_solid_body_rotation_exact():
    model = _tiny_model(gamma_dynamics="stretching", substeps=1)
    omega = torch.tensor([0.3, -0.5, 1.1], dtype=torch.float64)
    c = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    W = _skew(omega)  # J = [Omega]_x for u = Omega x (q - c)
    _stub_dynamics(model, W, c)
    gen = torch.Generator().manual_seed(5)
    x = c + torch.randn(50, 3, generator=gen, dtype=torch.float64)
    gamma = torch.randn(50, 3, generator=gen, dtype=torch.float64)
    sigma = torch.full((50,), 0.1, dtype=torch.float64)
    dt = 0.05
    with torch.no_grad():
        xn, gn, aux = model.step(x, gamma, sigma, dt, integrator="rk4", use_residual=False)
    assert "residual" not in aux
    Rot = torch.linalg.matrix_exp(dt * W)
    x_exact = c + (x - c) @ Rot.T
    assert torch.linalg.norm(xn - x_exact) / torch.linalg.norm(x_exact) < 1e-6
    disp_exact = x_exact - x  # stricter: error relative to the displacement itself
    assert torch.linalg.norm((xn - x) - disp_exact) / torch.linalg.norm(disp_exact) < 1e-5
    # Gamma under dGamma/dt = J^T Gamma with constant J: Gamma(dt) = expm(dt J^T) Gamma0
    g_exact = gamma @ torch.linalg.matrix_exp(dt * W.T).T
    assert torch.linalg.norm(gn - g_exact) / torch.linalg.norm(g_exact) < 1e-6


def test_step_gamma_uses_transposed_scheme():
    model = _tiny_model(gamma_dynamics="stretching", substeps=1)
    gen = torch.Generator().manual_seed(9)
    A = torch.randn(3, 3, generator=gen, dtype=torch.float64)  # general (non-symmetric) gradient
    c = torch.zeros(3, dtype=torch.float64)
    _stub_dynamics(model, A, c)
    x = torch.randn(30, 3, generator=gen, dtype=torch.float64)
    gamma = torch.randn(30, 3, generator=gen, dtype=torch.float64)
    sigma = torch.full((30,), 0.1, dtype=torch.float64)
    dt = 0.01
    with torch.no_grad():
        xn, gn, aux = model.step(x, gamma, sigma, dt, integrator="euler", use_residual=False)
    J = A.expand(30, 3, 3)
    expected = gamma + dt * torch.einsum("nji,nj->ni", J, gamma)  # (J^T Gamma)_i = J_ji Gamma_j
    torch.testing.assert_close(gn, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(xn, x + dt * (x - c) @ A.T, rtol=1e-12, atol=1e-12)
    classical = gamma + dt * torch.einsum("nij,nj->ni", J, gamma)  # (Gamma . grad) u: must differ
    assert (gn - classical).abs().max() > 1e-4


def test_step_residual_is_zero_at_init():
    """residual_head is zero-initialised, so use_residual=True must not change a fresh model's step."""
    model = _tiny_model()
    x, gamma, sigma, _ = _particles_and_queries()
    with torch.no_grad():
        xa, ga, _ = model.step(x, gamma, sigma, 0.01, integrator="euler", use_residual=False)
        xb, gb, aux = model.step(x, gamma, sigma, 0.01, integrator="euler", use_residual=True)
    assert aux["residual"].abs().max() == 0
    torch.testing.assert_close(xb, xa)
    torch.testing.assert_close(gb, ga)


# --------------------------------------------------------------------------- utilities
def test_vorticity_from_jacobian_matches_curl_of_linear_field():
    gen = torch.Generator().manual_seed(13)
    A = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    q = torch.randn(17, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    u = q @ A.T  # u = A q
    J = torch.stack([torch.autograd.grad(u[:, i].sum(), q, retain_graph=True)[0] for i in range(3)], 1)
    torch.testing.assert_close(J, A.expand(17, 3, 3))
    curl = torch.einsum("ijk,jk->i", _levi(), A.T)  # w_i = eps_ijk d u_k / d x_j = eps_ijk A_kj
    torch.testing.assert_close(vorticity_from_jacobian(J), curl.expand(17, 3))
    # sanity: solid-body rotation u = Omega x q has vorticity 2 Omega
    omega = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float64)
    torch.testing.assert_close(vorticity_from_jacobian(_skew(omega)), 2 * omega)


def test_rk4_substeps_converge_to_exact_rotation():
    """More substeps must not hurt: 2 RK4 substeps are at least as accurate as 1 for a stiff rotation."""
    omega = torch.tensor([0.0, 0.0, 25.0], dtype=torch.float64)  # ~1.6 rad per dt, like a vortex core
    c = torch.zeros(3, dtype=torch.float64)
    W = _skew(omega)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(20, 3, generator=gen, dtype=torch.float64)
    gamma = torch.randn(20, 3, generator=gen, dtype=torch.float64)
    sigma = torch.full((20,), 0.1, dtype=torch.float64)
    dt = 0.065
    x_exact = (x - c) @ torch.linalg.matrix_exp(dt * W).T + c
    errs = []
    for n in (1, 2, 4):
        model = _stub_dynamics(_tiny_model(gamma_dynamics="stretching", substeps=n), W, c)
        with torch.no_grad():
            xn, _, _ = model.step(x, gamma, sigma, dt, integrator="rk4", use_residual=False)
        errs.append(float((xn - x_exact).norm() / (x_exact - x).norm()))
    assert errs[1] < 0.1 * errs[0] and errs[2] < errs[1], errs


def test_rate_mode_gamma_update_is_relative_rate():
    """gamma_dynamics='rate': dGamma/dt = |Gamma| * gamma_rate(h, Gamma); one Euler step checks the wiring."""
    model = _tiny_model(gamma_dynamics="rate", substeps=1)
    with torch.no_grad():
        model.gamma_rate_head[-1].bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
        model.gamma_rate_head[-1].weight.zero_()
    x, gamma, sigma, _ = _particles_and_queries()
    dt = 0.01
    with torch.no_grad():
        _, gn, _ = model.step(x, gamma, sigma, dt, integrator="euler", use_residual=False)
    rate = torch.tensor([0.5, -1.0, 2.0], dtype=gamma.dtype) * float(model.gamma_rate_scale)
    expected = gamma + dt * gamma.norm(dim=-1, keepdim=True) * rate
    torch.testing.assert_close(gn, expected, rtol=1e-5, atol=1e-6)


def test_state_dict_roundtrip_without_metadata():
    model = _tiny_model()
    sd = model.state_dict()
    assert "_metadata" not in sd
    fresh = VortexGINO(TINY_CFG)
    fresh.load_state_dict({**sd, "_metadata": {"anything": 1}})  # neuralop metadata is tolerated


def _skew_batch(w):
    z = torch.zeros(len(w), 3, 3, dtype=w.dtype)
    z[:, 0, 1], z[:, 0, 2] = -w[:, 2], w[:, 1]
    z[:, 1, 0], z[:, 1, 2] = w[:, 2], -w[:, 0]
    z[:, 2, 0], z[:, 2, 1] = -w[:, 1], w[:, 0]
    return z


def test_relaxation_matches_flowvpm_corrected_pedrizzetti():
    """relax() = FLOWVPM correctedpedrizzetti: preserves |Gamma|, identity when Gamma || omega,
    rotates Gamma toward omega otherwise; omega excludes the particle's self-induced term."""
    model = _tiny_model(relaxation={"enabled": True, "rlxf": 0.3})
    gen = torch.Generator().manual_seed(4)
    n = 64
    gamma = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    sigma = torch.full((n,), 0.1, dtype=torch.float64)
    omega = torch.randn(n, 3, generator=gen, dtype=torch.float64) * 50.0
    # J whose antisymmetric part has curl omega, plus the self term that relax() must remove
    J = 0.5 * _skew_batch(omega) + (2.5 / (4 * math.pi * 0.1 ** 3)) * _skew_batch(gamma)
    J = J + torch.randn(n, 3, 3, generator=gen, dtype=torch.float64).transpose(1, 2).add(
        torch.randn(n, 3, 3, generator=gen, dtype=torch.float64))  # random symmetric-ish strain
    sym = 0.5 * (J + J.transpose(1, 2))
    J = 0.5 * _skew_batch(omega) + (2.5 / (4 * math.pi * 0.1 ** 3)) * _skew_batch(gamma) + sym
    g1 = model.relax(gamma, J, sigma)
    torch.testing.assert_close(g1.norm(dim=-1), gamma.norm(dim=-1), rtol=1e-10, atol=1e-12)
    w_hat = omega / omega.norm(dim=-1, keepdim=True)
    cos0 = (gamma * w_hat).sum(-1) / gamma.norm(dim=-1)
    cos1 = (g1 * w_hat).sum(-1) / g1.norm(dim=-1)
    assert torch.all(cos1 >= cos0 - 1e-12)
    # reference formula (FLOWVPM_relaxation.jl, relax_correctedpedrizzetti)
    f = 0.3
    ng = gamma.norm(dim=-1, keepdim=True)
    b2 = 1 - 2 * (1 - f) * f * (1 - cos0.unsqueeze(-1))
    ref = ((1 - f) * gamma + f * ng * w_hat) / b2.sqrt()
    torch.testing.assert_close(g1, ref, rtol=1e-10, atol=1e-12)
    # aligned strengths are left unchanged
    aligned = w_hat * ng
    J2 = 0.5 * _skew_batch(omega) + (2.5 / (4 * math.pi * 0.1 ** 3)) * _skew_batch(aligned)
    torch.testing.assert_close(model.relax(aligned, J2, sigma), aligned, rtol=1e-10, atol=1e-12)
