"""Particle-surrogate rollout utilities (quantities model + physics state update).

This module makes the modeling choice explicit:
1) the neural model predicts local interaction quantities (U and gradU),
2) particle states are advanced with rVPM-style update equations.

Why this matters:
- It preserves physical structure better than fully black-box next-state prediction.
- It avoids requiring perfect particle identity tracking between frames.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io import ensure_dir, load_config
from .train import ParticleGNOModel, relative_l2_loss


def _import_torch():
    """Import torch lazily so this module can still be imported without torch installed."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for particle rollout checks") from exc
    return torch


def _import_pyplot():
    """Import matplotlib in headless-safe mode and return pyplot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for rollout plots") from exc


@dataclass
class RolloutPhysicsConfig:
    """Physics update parameters for one-step particle rollout."""

    integration_mode: str = "reformulated"  # reformulated | classic
    transposed: bool = False
    f: float = 0.0
    g: float = 0.2
    zeta0: float = 1.0
    epsilon: float = 1.0e-12


def _idx(names: Sequence[str], key: str, required: bool = True) -> Optional[int]:
    """Get channel index from name list with clear error if required key is missing."""

    try:
        return list(names).index(key)
    except ValueError:
        if required:
            raise KeyError(f"Required channel `{key}` not found in names: {list(names)}")
        return None


def _extract_J(pred: np.ndarray, target_names: Sequence[str]) -> np.ndarray:
    """Build velocity-gradient tensor J from model outputs using target channel names."""

    # Layout expected from dataset_builder target names:
    # gradUx_x gradUx_y gradUx_z
    # gradUy_x gradUy_y gradUy_z
    # gradUz_x gradUz_y gradUz_z
    gxx = _idx(target_names, "gradUx_x")
    gxy = _idx(target_names, "gradUx_y")
    gxz = _idx(target_names, "gradUx_z")
    gyx = _idx(target_names, "gradUy_x")
    gyy = _idx(target_names, "gradUy_y")
    gyz = _idx(target_names, "gradUy_z")
    gzx = _idx(target_names, "gradUz_x")
    gzy = _idx(target_names, "gradUz_y")
    gzz = _idx(target_names, "gradUz_z")

    J = np.zeros((pred.shape[0], 3, 3), dtype=np.float64)
    J[:, 0, 0] = pred[:, gxx]
    J[:, 0, 1] = pred[:, gxy]
    J[:, 0, 2] = pred[:, gxz]
    J[:, 1, 0] = pred[:, gyx]
    J[:, 1, 1] = pred[:, gyy]
    J[:, 1, 2] = pred[:, gyz]
    J[:, 2, 0] = pred[:, gzx]
    J[:, 2, 1] = pred[:, gzy]
    J[:, 2, 2] = pred[:, gzz]
    return J


def _stretching_term(J: np.ndarray, gamma: np.ndarray, transposed: bool) -> np.ndarray:
    """Compute stretching term S from J and Gamma following FLOWVPM transposed/classic switch."""

    g = gamma[..., None]  # (N,3,1)
    if transposed:
        # Transposed scheme in FLOWVPM: S = J * Gamma
        return (J @ g)[:, :, 0]
    # Classic scheme in FLOWVPM: S = J' * Gamma
    return (np.transpose(J, (0, 2, 1)) @ g)[:, :, 0]


def apply_rvpm_one_step(
    x_frame: np.ndarray,
    pred_quantities: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    dt: float,
    uinf: Sequence[float],
    cfg: RolloutPhysicsConfig,
) -> Dict[str, np.ndarray]:
    """Advance particle state by one step using predicted U/gradU and rVPM-style update."""

    feat = list(feature_names)
    targ = list(target_names)

    ix = _idx(feat, "x")
    iy = _idx(feat, "y")
    iz = _idx(feat, "z")
    igx = _idx(feat, "Gamma_x")
    igy = _idx(feat, "Gamma_y")
    igz = _idx(feat, "Gamma_z")
    isgm = _idx(feat, "sigma", required=False)
    istatic = _idx(feat, "static", required=False)

    iux = _idx(targ, "velocity_x")
    iuy = _idx(targ, "velocity_y")
    iuz = _idx(targ, "velocity_z")

    X = np.asarray(x_frame[:, [ix, iy, iz]], dtype=np.float64).copy()
    G = np.asarray(x_frame[:, [igx, igy, igz]], dtype=np.float64).copy()
    sigma = np.ones((x_frame.shape[0],), dtype=np.float64)
    if isgm is not None:
        sigma = np.asarray(x_frame[:, isgm], dtype=np.float64).copy()

    U = np.asarray(pred_quantities[:, [iux, iuy, iuz]], dtype=np.float64)
    J = _extract_J(np.asarray(pred_quantities, dtype=np.float64), targ)

    # Skip updates on static particles.
    static_mask = np.zeros((x_frame.shape[0],), dtype=bool)
    if istatic is not None:
        static_mask = np.asarray(x_frame[:, istatic] > 0.5, dtype=bool)

    uinf_v = np.asarray(uinf, dtype=np.float64).reshape(-1)
    if uinf_v.shape[0] != 3:
        raise ValueError(f"freestream must have 3 components, got {uinf_v}")

    # 1) Position convection
    X_new = X.copy()
    X_new[~static_mask] += float(dt) * (U[~static_mask] + uinf_v[None, :])

    # 2) Circulation stretching
    S = _stretching_term(J, G, transposed=bool(cfg.transposed))

    G_new = G.copy()
    sigma_new = sigma.copy()

    mode = str(cfg.integration_mode).lower()
    if mode == "classic":
        G_new[~static_mask] += float(dt) * S[~static_mask]

    elif mode == "reformulated":
        # Reformulated update (without explicit SFS forcing term).
        f = float(cfg.f)
        g = float(cfg.g)
        eps = float(cfg.epsilon)
        den_fac = 1.0 + 3.0 * f

        Gnorm2 = np.sum(G * G, axis=1)
        dotSG = np.sum(S * G, axis=1)

        Z = np.zeros_like(Gnorm2)
        valid = Gnorm2 > eps
        Z[valid] = ((f + g) / den_fac) * (dotSG[valid] / Gnorm2[valid])

        G_new[~static_mask] += float(dt) * (
            S[~static_mask] - 3.0 * Z[~static_mask, None] * G[~static_mask]
        )
        sigma_new[~static_mask] -= float(dt) * sigma[~static_mask] * Z[~static_mask]

    else:
        raise ValueError(f"Unknown integration_mode={cfg.integration_mode}. Use classic or reformulated.")

    # Keep sigma positive.
    sigma_new = np.maximum(sigma_new, 1.0e-12)

    out = {
        "X_next": X_new.astype(np.float32),
        "Gamma_next": G_new.astype(np.float32),
        "sigma_next": sigma_new.astype(np.float32),
        "U_pred": U.astype(np.float32),
        "J_pred": J.astype(np.float32),
    }
    return out


def _load_frame_contexts(ds: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Load per-frame contexts from dataset; fallback to minimal defaults when absent."""

    if "frame_contexts" in ds:
        raw = list(ds["frame_contexts"])
        out: List[Dict[str, Any]] = []
        for rec in raw:
            if isinstance(rec, dict):
                out.append(rec)
            else:
                try:
                    out.append(dict(rec))
                except Exception:
                    out.append({})
        return out

    # Backward-compat fallback.
    out = []
    for rec in ds["frame_offsets"]:
        case_name, frame_id, start, end = rec
        out.append(
            {
                "case_name": str(case_name),
                "frame_id": str(frame_id),
                "start": int(start),
                "end": int(end),
                "dt": None,
                "freestream": [0.0, 0.0, 0.0],
                "phase": 0.0,
                "angle_of_attack": 0.0,
                "stationary": False,
            }
        )
    return out


def evaluate_rollout(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate one-step rollout: quantity prediction + physics state update to t+dt."""

    torch = _import_torch()

    ds_path = Path(config["dataset_npz"])
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}")

    ds = np.load(ds_path, allow_pickle=True)
    X = np.asarray(ds["inputs_particle"], dtype=np.float32)
    Y = np.asarray(ds["targets_particle"], dtype=np.float32)
    feature_names = [str(v) for v in ds["feature_names"].tolist()]
    target_names = [str(v) for v in ds["target_names"].tolist()]
    frame_offsets = list(ds["frame_offsets"])
    frame_contexts = _load_frame_contexts(ds)

    # Build model with same dimensions/hyperparameters as training config.
    model_cfg = dict(config.get("model", {}))
    model = ParticleGNOModel(
        in_dim=int(X.shape[1]),
        out_dim=int(Y.shape[1]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        radius=float(model_cfg.get("radius", 0.2)),
        transform_type=str(model_cfg.get("transform_type", "linear")),
        reduction=str(model_cfg.get("reduction", "mean")),
        pos_embedding_type=str(model_cfg.get("pos_embedding_type", "transformer")),
        pos_embedding_channels=int(model_cfg.get("pos_embedding_channels", 16)),
        use_open3d_neighbor_search=bool(model_cfg.get("use_open3d_neighbor_search", False)),
        use_torch_scatter_reduce=bool(model_cfg.get("use_torch_scatter_reduce", False)),
    )

    ckpt = Path(config["checkpoint_path"])
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    phys_cfg = RolloutPhysicsConfig(**config.get("physics", {}))
    dt_default = float(config.get("dt_default", 0.0))
    fs_default = np.asarray(config.get("freestream_default", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    max_pairs = int(config.get("max_pairs", -1))

    rel_quantity: List[float] = []
    mse_quantity: List[float] = []
    pos_l2: List[float] = []
    gamma_l2: List[float] = []
    sigma_l2: List[float] = []
    evaluated_pairs = 0
    skipped_count_mismatch = 0

    # Parse frame offsets once.
    offsets: List[Tuple[str, str, int, int]] = []
    for rec in frame_offsets:
        case_name, frame_id, s, e = rec
        offsets.append((str(case_name), str(frame_id), int(s), int(e)))

    for i in range(len(offsets) - 1):
        if max_pairs > 0 and evaluated_pairs >= max_pairs:
            break

        c0, f0, s0, e0 = offsets[i]
        c1, f1, s1, e1 = offsets[i + 1]

        # Only compare consecutive frames of same case.
        if c0 != c1:
            continue

        n0 = e0 - s0
        n1 = e1 - s1
        if n0 != n1:
            skipped_count_mismatch += 1
            continue

        x0 = X[s0:e0]
        y0 = Y[s0:e0]
        x1_true = X[s1:e1]

        with torch.no_grad():
            y0_pred = model(torch.from_numpy(x0)).cpu().numpy()

        # 1) Quantity-prediction quality at frame t.
        rel_q = float(relative_l2_loss(torch.from_numpy(y0_pred).unsqueeze(0), torch.from_numpy(y0).unsqueeze(0)).item())
        mse_q = float(np.mean((y0_pred - y0) ** 2))
        rel_quantity.append(rel_q)
        mse_quantity.append(mse_q)

        # 2) Roll state forward using physics update.
        ctx = frame_contexts[i] if i < len(frame_contexts) else {}
        dt = float(ctx.get("dt", dt_default) if ctx.get("dt", None) is not None else dt_default)
        if dt == 0.0:
            # No dt means no meaningful rollout; still evaluate quantity model.
            continue

        freestream = np.asarray(ctx.get("freestream", fs_default), dtype=np.float64).reshape(-1)
        if freestream.shape[0] != 3:
            freestream = fs_default

        step = apply_rvpm_one_step(
            x_frame=x0,
            pred_quantities=y0_pred,
            feature_names=feature_names,
            target_names=target_names,
            dt=dt,
            uinf=freestream,
            cfg=phys_cfg,
        )

        # Compare predicted next state against true next-frame state (index-wise).
        ix = _idx(feature_names, "x")
        iy = _idx(feature_names, "y")
        iz = _idx(feature_names, "z")
        igx = _idx(feature_names, "Gamma_x")
        igy = _idx(feature_names, "Gamma_y")
        igz = _idx(feature_names, "Gamma_z")
        isgm = _idx(feature_names, "sigma", required=False)

        X1_true = x1_true[:, [ix, iy, iz]]
        G1_true = x1_true[:, [igx, igy, igz]]
        S1_true = x1_true[:, isgm] if isgm is not None else np.ones((x1_true.shape[0],), dtype=np.float32)

        pX = step["X_next"]
        pG = step["Gamma_next"]
        pS = step["sigma_next"]

        pos_l2.append(float(np.sqrt(np.mean(np.sum((pX - X1_true) ** 2, axis=1)))))
        gamma_l2.append(float(np.sqrt(np.mean(np.sum((pG - G1_true) ** 2, axis=1)))))
        sigma_l2.append(float(np.sqrt(np.mean((pS - S1_true) ** 2))))

        evaluated_pairs += 1

    summary = {
        "dataset_npz": str(ds_path),
        "checkpoint_path": str(ckpt),
        "model_in_dim": int(X.shape[1]),
        "model_out_dim": int(Y.shape[1]),
        "n_frames": len(offsets),
        "evaluated_pairs": int(evaluated_pairs),
        "skipped_pairs_count_mismatch": int(skipped_count_mismatch),
        "quantity_rel_l2_mean": float(np.mean(rel_quantity)) if rel_quantity else None,
        "quantity_mse_mean": float(np.mean(mse_quantity)) if mse_quantity else None,
        "rollout_position_l2_mean": float(np.mean(pos_l2)) if pos_l2 else None,
        "rollout_gamma_l2_mean": float(np.mean(gamma_l2)) if gamma_l2 else None,
        "rollout_sigma_l2_mean": float(np.mean(sigma_l2)) if sigma_l2 else None,
        "physics_mode": phys_cfg.integration_mode,
        "physics_transposed": bool(phys_cfg.transposed),
        "physics_f": float(phys_cfg.f),
        "physics_g": float(phys_cfg.g),
    }

    out_dir = ensure_dir(Path(config.get("output_dir", "final/output/particle_rollout")))
    (out_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2))

    # Optional visualization: position scatter for first evaluable pair.
    if bool(config.get("plot_example", True)) and evaluated_pairs > 0:
        plt = _import_pyplot()

        # Find first usable pair again for plot.
        for i in range(len(offsets) - 1):
            c0, f0, s0, e0 = offsets[i]
            c1, f1, s1, e1 = offsets[i + 1]
            if c0 != c1 or (e0 - s0) != (e1 - s1):
                continue
            x0 = X[s0:e0]
            x1 = X[s1:e1]
            with torch.no_grad():
                y0_pred = model(torch.from_numpy(x0)).cpu().numpy()
            ctx = frame_contexts[i] if i < len(frame_contexts) else {}
            dt = float(ctx.get("dt", dt_default) if ctx.get("dt", None) is not None else dt_default)
            if dt == 0.0:
                continue
            freestream = np.asarray(ctx.get("freestream", fs_default), dtype=np.float64).reshape(-1)
            step = apply_rvpm_one_step(
                x_frame=x0,
                pred_quantities=y0_pred,
                feature_names=feature_names,
                target_names=target_names,
                dt=dt,
                uinf=freestream,
                cfg=phys_cfg,
            )

            ix = _idx(feature_names, "x")
            iz = _idx(feature_names, "z")
            pred_xz = step["X_next"][:, [0, 2]]
            true_xz = x1[:, [ix, iz]]

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.scatter(true_xz[:, 0], true_xz[:, 1], s=2, alpha=0.35, label="true t+dt")
            ax.scatter(pred_xz[:, 0], pred_xz[:, 1], s=2, alpha=0.35, label="pred t+dt")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_title(f"Rollout one-step check: {c0} {f0}->{f1}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "rollout_example_xz.png", dpi=150)
            plt.close(fig)
            break

    return summary


def main() -> None:
    """CLI wrapper for particle rollout evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate particle rollout using learned quantities + physics update")
    parser.add_argument("--config", type=str, default="final/configs/particle_rollout.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = evaluate_rollout(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
