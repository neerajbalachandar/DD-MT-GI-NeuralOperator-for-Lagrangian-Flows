"""Sanity checks and visual diagnostics for the `final/` pipeline.

This module is intentionally lightweight and read-only with respect to training data.
It helps answer:
- What raw/merged data exists?
- What goes into each model?
- What comes out of each model?
- Is training history behaving sensibly?

All figures and summaries are written to an output folder so you can inspect them
on any machine (including headless servers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset_builder import extract_particle_channels
from .io import ensure_dir, load_config


# -------------------------
# Plot utilities
# -------------------------

def _import_pyplot():
    """Import matplotlib in headless-safe mode and return `pyplot`."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for sanity visualizations. Install matplotlib first."
        ) from exc


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON payload with indentation and create parent folder automatically."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def _to_obj_list(x: np.ndarray) -> List[str]:
    """Convert object/string numpy arrays to Python string list."""

    return [str(v) for v in np.asarray(x).reshape(-1).tolist()]


def _sample_glob_patterns(root: Path, patterns: Sequence[str]) -> List[Path]:
    """Resolve multiple glob patterns relative to `root` into sorted unique paths."""

    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    return sorted(set(files))


def _tensor_channel_stats(t: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Compute min/max/mean/std stats for a `(C, ...)` tensor."""

    x = np.asarray(t)
    if x.ndim < 2:
        raise ValueError(f"Expected channel tensor `(C, ...)`, got shape={x.shape}")

    stats: Dict[str, Dict[str, float]] = {}
    for i, n in enumerate(names):
        c = x[i].astype(np.float64)
        stats[str(n)] = {
            "min": float(np.min(c)),
            "max": float(np.max(c)),
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
        }
    return stats


def _plot_slices(
    tensor: np.ndarray,
    channel_names: Sequence[str],
    out_path: Path,
    title: str,
    max_channels: int = 6,
) -> None:
    """Plot mid-plane 2D slices from a `(C, nx, ny, nz)` tensor."""

    plt = _import_pyplot()

    x = np.asarray(tensor)
    if x.ndim != 4:
        raise ValueError(f"Expected `(C, nx, ny, nz)`, got shape={x.shape}")

    c = min(int(x.shape[0]), int(max_channels))
    if c <= 0:
        return

    ncols = 3
    nrows = int(np.ceil(c / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)

    zmid = x.shape[-1] // 2
    for i in range(c):
        r, col = divmod(i, ncols)
        ax = axes[r][col]
        im = ax.imshow(x[i, :, :, zmid], origin="lower", cmap="viridis")
        nm = channel_names[i] if i < len(channel_names) else f"ch_{i}"
        ax.set_title(str(nm))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(c, nrows * ncols):
        r, col = divmod(j, ncols)
        axes[r][col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_loss_history(history_path: Path, out_path: Path, title: str) -> bool:
    """Plot training history curves if a history JSON exists."""

    if not history_path.exists():
        return False

    plt = _import_pyplot()
    data = json.loads(history_path.read_text())
    if not isinstance(data, list) or not data:
        return False

    epochs = [int(d.get("epoch", i + 1)) for i, d in enumerate(data)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    if "train_loss" in data[0]:
        ax.plot(epochs, [float(d.get("train_loss", np.nan)) for d in data], label="train_loss")
    if "val_rel_l2" in data[0]:
        ax.plot(epochs, [float(d.get("val_rel_l2", np.nan)) for d in data], label="val_rel_l2")
    if "val_mse" in data[0]:
        ax.plot(epochs, [float(d.get("val_mse", np.nan)) for d in data], label="val_mse")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / metric")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# -------------------------
# Data sanity checks
# -------------------------

def summarize_merged_inputs(
    pipeline_cfg: Mapping[str, Any],
    root: Path,
    out_dir: Path,
    max_files_per_case: int,
) -> Dict[str, Any]:
    """Summarize merged NPZ inputs referenced by `cases[].npz_glob` in pipeline config."""

    plt = _import_pyplot()

    cases = list(pipeline_cfg.get("cases", []))
    case_summaries: Dict[str, Any] = {}

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    scatter_pts: List[np.ndarray] = []
    key_counts: Dict[str, int] = {}

    for case in cases:
        name = str(case.get("name", "case"))
        npz_glob = case.get("npz_glob", [])
        patterns = [npz_glob] if isinstance(npz_glob, str) else list(npz_glob)
        files = _sample_glob_patterns(root, patterns)

        if not files:
            case_summaries[name] = {"n_files": 0, "n_checked": 0}
            continue

        checked = files[: max_files_per_case if max_files_per_case > 0 else len(files)]
        np_counts: List[int] = []

        for i, p in enumerate(checked):
            with np.load(p, allow_pickle=True) as d:
                payload = {k: d[k] for k in d.files}

            for k in payload.keys():
                key_counts[k] = key_counts.get(k, 0) + 1

            try:
                xyz, chans = extract_particle_channels(payload)
                np_counts.append(int(xyz.shape[0]))
                global_min = np.minimum(global_min, np.min(xyz, axis=0))
                global_max = np.maximum(global_max, np.max(xyz, axis=0))

                # Keep a thin sample for global scatter (for readability/speed).
                stride = max(1, xyz.shape[0] // 5000)
                scatter_pts.append(xyz[::stride])

                # Save one per-case sample scatter.
                if i == 0:
                    c = chans.get("gamma_mag", np.zeros(xyz.shape[0], dtype=np.float64))
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    sc = ax.scatter(xyz[:, 0], xyz[:, 2], c=c, s=2, cmap="turbo")
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")
                    ax.set_title(f"{name}: sample particle cloud (x-z)")
                    fig.colorbar(sc, ax=ax, label="gamma_mag")
                    fig.tight_layout()
                    sample_path = out_dir / "merged_inputs" / f"{name}_sample_scatter_xz.png"
                    ensure_dir(sample_path.parent)
                    fig.savefig(sample_path, dpi=150)
                    plt.close(fig)
            except Exception:
                # Not all NPZ files must carry particle arrays; keep report robust.
                continue

        case_summaries[name] = {
            "n_files": len(files),
            "n_checked": len(checked),
            "particles_min": int(np.min(np_counts)) if np_counts else 0,
            "particles_max": int(np.max(np_counts)) if np_counts else 0,
            "particles_mean": float(np.mean(np_counts)) if np_counts else 0.0,
        }

    if scatter_pts:
        all_pts = np.concatenate(scatter_pts, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(all_pts[:, 0], all_pts[:, 2], s=1, alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Merged inputs: global particle cloud sample (x-z)")
        fig.tight_layout()
        out_path = out_dir / "merged_inputs" / "global_particle_cloud_xz.png"
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    summary = {
        "cases": case_summaries,
        "global_xyz_min": global_min.tolist() if np.isfinite(global_min).all() else None,
        "global_xyz_max": global_max.tolist() if np.isfinite(global_max).all() else None,
        "npz_key_frequency": dict(sorted(key_counts.items(), key=lambda kv: kv[0])),
    }

    _save_json(out_dir / "merged_inputs" / "summary.json", summary)
    return summary


def inspect_field_dataset(
    pipeline_cfg: Mapping[str, Any],
    root: Path,
    out_dir: Path,
    preset_name: str,
    use_normalized: bool = True,
) -> Dict[str, Any]:
    """Inspect one field preset dataset and visualize model inputs/targets."""

    dataset_name = str(pipeline_cfg.get("dataset_name", "geometry_aware"))
    output_root = Path(pipeline_cfg.get("output_root", "final/output"))
    dataset_root = root / output_root / dataset_name

    preset_root = dataset_root / f"preset_{preset_name}"
    if not preset_root.exists():
        # Fallback: choose first available preset for resilience.
        available = sorted(dataset_root.glob("preset_*"))
        if not available:
            raise RuntimeError(f"No preset folders found under {dataset_root}")
        preset_root = available[0]

    split_path = preset_root / "split_files.json"
    if not split_path.exists():
        raise RuntimeError(f"split_files.json missing in {preset_root}")

    split = json.loads(split_path.read_text())
    rel_train = list(split.get("train", []))
    rel_val = list(split.get("val", []))
    rel_test = list(split.get("test", []))

    if not rel_train and not rel_val and not rel_test:
        raise RuntimeError(f"No samples listed in split files for {preset_root}")

    pick_rel = rel_train[0] if rel_train else (rel_val[0] if rel_val else rel_test[0])
    sample_path = preset_root / pick_rel

    with np.load(sample_path, allow_pickle=True) as d:
        x_key = "input_norm" if use_normalized and "input_norm" in d.files else "input"
        y_key = "target_norm" if use_normalized and "target_norm" in d.files else "target"
        x = np.asarray(d[x_key], dtype=np.float32)
        y = np.asarray(d[y_key], dtype=np.float32)
        in_ch = _to_obj_list(d["input_channels"])
        out_ch = _to_obj_list(d["target_channels"])

    _plot_slices(x, in_ch, out_dir / "field" / "sample_input_slices.png", "Field input channels (sample)")
    _plot_slices(y, out_ch, out_dir / "field" / "sample_target_slices.png", "Field target channels (sample)")

    summary = {
        "preset_root": str(preset_root),
        "sample_path": str(sample_path),
        "split_counts": {
            "train": len(rel_train),
            "val": len(rel_val),
            "test": len(rel_test),
        },
        "input_shape": [int(s) for s in x.shape],
        "target_shape": [int(s) for s in y.shape],
        "input_channels": in_ch,
        "target_channels": out_ch,
        "input_stats": _tensor_channel_stats(x, in_ch),
        "target_stats": _tensor_channel_stats(y, out_ch),
    }

    _save_json(out_dir / "field" / "sample_summary.json", summary)
    return summary


def inspect_particle_dataset(
    particle_dataset_npz: Path,
    split_idx_npz: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Inspect particle surrogate dataset and visualize representative frame data."""

    plt = _import_pyplot()

    if not particle_dataset_npz.exists():
        raise RuntimeError(f"Particle dataset not found: {particle_dataset_npz}")

    ds = np.load(particle_dataset_npz, allow_pickle=True)
    X = np.asarray(ds["inputs_particle"], dtype=np.float32)
    Y = np.asarray(ds["targets_particle"], dtype=np.float32)
    feat = _to_obj_list(ds["feature_names"])
    targ = _to_obj_list(ds["target_names"])
    offsets = list(ds["frame_offsets"])

    split_idx = np.load(split_idx_npz, allow_pickle=False)
    split_counts = {k: int(np.asarray(split_idx[k]).shape[0]) for k in split_idx.files}

    feat_stats = {
        feat[i]: {
            "min": float(np.min(X[:, i])),
            "max": float(np.max(X[:, i])),
            "mean": float(np.mean(X[:, i])),
            "std": float(np.std(X[:, i])),
        }
        for i in range(X.shape[1])
    }

    # Visualize one frame input cloud in x-z colored by gamma magnitude when available.
    case_name, frame_id, s, e = offsets[0]
    s = int(s)
    e = int(e)
    xf = X[s:e]

    xi = feat.index("x") if "x" in feat else 0
    zi = feat.index("z") if "z" in feat else min(2, xf.shape[1] - 1)
    gx = feat.index("Gamma_x") if "Gamma_x" in feat else None
    gy = feat.index("Gamma_y") if "Gamma_y" in feat else None
    gz = feat.index("Gamma_z") if "Gamma_z" in feat else None

    if gx is not None and gy is not None and gz is not None:
        c = np.sqrt(xf[:, gx] ** 2 + xf[:, gy] ** 2 + xf[:, gz] ** 2)
    else:
        c = np.zeros(xf.shape[0], dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sc = ax.scatter(xf[:, xi], xf[:, zi], c=c, s=2, cmap="turbo")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Particle frame sample: {case_name} / {frame_id}")
    fig.colorbar(sc, ax=ax, label="|Gamma|")
    fig.tight_layout()
    out_path = out_dir / "particle" / "sample_frame_scatter_xz.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    summary = {
        "dataset_npz": str(particle_dataset_npz),
        "split_idx_npz": str(split_idx_npz),
        "inputs_shape": [int(s) for s in X.shape],
        "targets_shape": [int(s) for s in Y.shape],
        "n_frames": len(offsets),
        "feature_names": feat,
        "target_names": targ,
        "feature_stats": feat_stats,
        "split_counts": split_counts,
    }
    _save_json(out_dir / "particle" / "summary.json", summary)
    return summary


# -------------------------
# Model sanity checks
# -------------------------

def _count_params(model) -> int:
    """Return number of trainable parameters in a torch model."""

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_field_forward_sanity(field_train_cfg: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Run one forward pass for field model and visualize prediction vs target."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model sanity checks") from exc

    from .train import build_field_model, relative_l2_loss

    preset_root = Path(field_train_cfg["preset_root"])
    split = json.loads((preset_root / "split_files.json").read_text())
    rel = (split.get("test") or split.get("val") or split.get("train"))[0]
    sample_path = preset_root / rel

    use_norm = bool(field_train_cfg.get("use_normalized", True))
    with np.load(sample_path, allow_pickle=True) as d:
        x_key = "input_norm" if use_norm and "input_norm" in d.files else "input"
        y_key = "target_norm" if use_norm and "target_norm" in d.files else "target"
        x = np.asarray(d[x_key], dtype=np.float32)
        y = np.asarray(d[y_key], dtype=np.float32)
        target_channels = _to_obj_list(d["target_channels"])

    model_cfg = dict(field_train_cfg.get("model", {}))
    model_cfg["in_channels"] = int(x.shape[0])
    model_cfg["out_channels"] = int(y.shape[0])

    device = torch.device("cpu")
    model = build_field_model(model_cfg, device=device)

    ckpt = Path(field_train_cfg.get("output_dir", preset_root / "training_field")) / "best_field_model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"Field checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    xt = torch.from_numpy(x).unsqueeze(0)
    yt = torch.from_numpy(y).unsqueeze(0)
    with torch.no_grad():
        pred = model(xt).cpu().numpy()[0]
        rel = float(relative_l2_loss(torch.from_numpy(pred).unsqueeze(0), yt).item())
        mse = float(np.mean((pred - y) ** 2))

    _plot_slices(pred, target_channels, out_dir / "field" / "pred_slices.png", "Field prediction slices")
    _plot_slices(np.abs(pred - y), target_channels, out_dir / "field" / "abs_error_slices.png", "Field abs-error slices")

    history_path = Path(field_train_cfg.get("output_dir", preset_root / "training_field")) / "history.json"
    _plot_loss_history(history_path, out_dir / "field" / "training_history.png", "Field training history")

    summary = {
        "checkpoint": str(ckpt),
        "sample_path": str(sample_path),
        "input_shape": [int(s) for s in x.shape],
        "target_shape": [int(s) for s in y.shape],
        "pred_shape": [int(s) for s in pred.shape],
        "relative_l2": rel,
        "mse": mse,
        "n_trainable_params": _count_params(model),
    }
    _save_json(out_dir / "field" / "forward_summary.json", summary)
    return summary


def run_particle_forward_sanity(particle_train_cfg: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Run one forward pass for particle GNO model and summarize prediction quality."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model sanity checks") from exc

    plt = _import_pyplot()

    from .train import ParticleGNOModel, relative_l2_loss

    ds_path = Path(particle_train_cfg["dataset_npz"])
    ds = np.load(ds_path, allow_pickle=True)
    X = np.asarray(ds["inputs_particle"], dtype=np.float32)
    Y = np.asarray(ds["targets_particle"], dtype=np.float32)
    offsets = list(ds["frame_offsets"])

    # Use first frame for quick deterministic sanity.
    _, _, s, e = offsets[0]
    s = int(s)
    e = int(e)
    x = X[s:e]
    y = Y[s:e]

    device = torch.device("cpu")
    model = ParticleGNOModel(
        in_dim=int(x.shape[1]),
        out_dim=int(y.shape[1]),
        hidden_dim=int(particle_train_cfg.get("hidden_dim", 128)),
        n_layers=int(particle_train_cfg.get("n_layers", 4)),
        radius=float(particle_train_cfg.get("radius", 0.2)),
        transform_type=str(particle_train_cfg.get("transform_type", "linear")),
        reduction=str(particle_train_cfg.get("reduction", "mean")),
        pos_embedding_type=str(particle_train_cfg.get("pos_embedding_type", "transformer")),
        pos_embedding_channels=int(particle_train_cfg.get("pos_embedding_channels", 16)),
        use_open3d_neighbor_search=bool(particle_train_cfg.get("use_open3d_neighbor_search", False)),
        use_torch_scatter_reduce=bool(particle_train_cfg.get("use_torch_scatter_reduce", False)),
    ).to(device)

    ckpt = Path(particle_train_cfg.get("output_dir", "final/output/particle_training")) / "best_particle_gno_model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"Particle checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
        rel = float(relative_l2_loss(torch.from_numpy(pred).unsqueeze(0), yt.unsqueeze(0)).item())
        mse = float(np.mean((pred - y) ** 2))

    # Pred-vs-true scatter for velocity components.
    names = ["vx", "vy", "vz"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax = axes[i]
        ax.scatter(y[:, i], pred[:, i], s=2, alpha=0.3)
        lo = float(min(np.min(y[:, i]), np.min(pred[:, i])))
        hi = float(max(np.max(y[:, i]), np.max(pred[:, i])))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_title(names[i])
    fig.suptitle("Particle model: predicted vs true velocity")
    fig.tight_layout()
    out_path = out_dir / "particle" / "pred_vs_true_velocity.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    history_path = Path(particle_train_cfg.get("output_dir", "final/output/particle_training")) / "history.json"
    _plot_loss_history(history_path, out_dir / "particle" / "training_history.png", "Particle training history")

    summary = {
        "checkpoint": str(ckpt),
        "frame_rows": int(x.shape[0]),
        "input_shape": [int(s) for s in x.shape],
        "target_shape": [int(s) for s in y.shape],
        "pred_shape": [int(s) for s in pred.shape],
        "relative_l2": rel,
        "mse": mse,
        "n_trainable_params": _count_params(model),
    }
    _save_json(out_dir / "particle" / "forward_summary.json", summary)
    return summary


# -------------------------
# CLI
# -------------------------

def main() -> None:
    """Run all enabled sanity checks and write visual artifacts to disk."""

    parser = argparse.ArgumentParser(description="Sanity checks for final preprocessing/training pipeline")
    parser.add_argument("--pipeline-config", type=str, default="final/configs/pipeline_config.yaml")
    parser.add_argument("--field-train-config", type=str, default="final/configs/train_field_fno.yaml")
    parser.add_argument("--particle-train-config", type=str, default="final/configs/train_particle_gno.yaml")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="final/output/sanity_checks")
    parser.add_argument("--preset", type=str, default="E", help="Preset name to inspect (A/B/C/D/E/F/CUSTOM)")
    parser.add_argument("--max-files-per-case", type=int, default=20)
    parser.add_argument("--skip-model-forward", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = ensure_dir(root / args.output_dir)

    pipeline_cfg = load_config(args.pipeline_config)

    report: Dict[str, Any] = {}

    try:
        print("[sanity] summarizing merged inputs referenced by pipeline config...")
        report["merged_inputs"] = summarize_merged_inputs(
            pipeline_cfg=pipeline_cfg,
            root=root,
            out_dir=out_dir,
            max_files_per_case=int(args.max_files_per_case),
        )
    except Exception as exc:
        report["merged_inputs"] = {"skipped": True, "reason": str(exc)}

    try:
        print("[sanity] inspecting built field dataset samples...")
        report["field_dataset"] = inspect_field_dataset(
            pipeline_cfg=pipeline_cfg,
            root=root,
            out_dir=out_dir,
            preset_name=str(args.preset),
            use_normalized=True,
        )
    except Exception as exc:
        report["field_dataset"] = {"skipped": True, "reason": str(exc)}

    # Particle dataset path defaults from pipeline layout.
    dataset_name = str(pipeline_cfg.get("dataset_name", "geometry_aware"))
    output_root = Path(pipeline_cfg.get("output_root", "final/output"))
    particle_dataset_npz = root / output_root / dataset_name / "particle_dataset" / "particle_dataset.npz"
    particle_split_idx_npz = root / output_root / dataset_name / "particle_dataset" / "particle_split_indices.npz"

    if particle_dataset_npz.exists() and particle_split_idx_npz.exists():
        print("[sanity] inspecting particle dataset...")
        report["particle_dataset"] = inspect_particle_dataset(
            particle_dataset_npz=particle_dataset_npz,
            split_idx_npz=particle_split_idx_npz,
            out_dir=out_dir,
        )
    else:
        report["particle_dataset"] = {"skipped": True, "reason": "particle dataset files not found"}

    if not args.skip_model_forward:
        # Field forward sanity
        try:
            field_train_cfg = load_config(args.field_train_config)
            print("[sanity] running field model forward sanity...")
            report["field_model_forward"] = run_field_forward_sanity(field_train_cfg, out_dir=out_dir)
        except Exception as exc:
            report["field_model_forward"] = {"skipped": True, "reason": str(exc)}

        # Particle forward sanity
        try:
            particle_train_cfg = load_config(args.particle_train_config)
            print("[sanity] running particle model forward sanity...")
            report["particle_model_forward"] = run_particle_forward_sanity(particle_train_cfg, out_dir=out_dir)
        except Exception as exc:
            report["particle_model_forward"] = {"skipped": True, "reason": str(exc)}

    _save_json(out_dir / "sanity_report.json", report)
    print("[sanity] done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
