"""Training entry points for field-FNO and particle-GNO models."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import ensure_dir, load_config


def set_seed(seed: int = 42) -> None:
    """Seed numpy/torch RNGs (including CUDA generators when available) for reproducible runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    """Resolve runtime device from config (`cuda`, `cpu`, `auto`) and print selected backend."""
    requested = str(cfg.get("device", "auto")).lower()

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested in config (`device: cuda`) but no CUDA device is available.")
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = "unknown-gpu"
        print(f"[device] {device} ({gpu_name})")
    else:
        print("[device] cpu")

    return device


def make_adam_optimizer(params, cfg: Dict[str, Any]):
    """Use NeuralOperator Adam/AdamW when available, else torch fallback."""
    opt_name = str(cfg.get("optimizer", "auto")).lower()

    candidates = {}
    try:
        import neuralop.training as ntrain  # type: ignore

        if hasattr(ntrain, "AdamW"):
            candidates["adamw"] = ntrain.AdamW
        if hasattr(ntrain, "Adam"):
            candidates["adam"] = ntrain.Adam
    except Exception:
        pass

    if "adamw" not in candidates and hasattr(torch.optim, "AdamW"):
        candidates["adamw"] = torch.optim.AdamW
    if "adam" not in candidates:
        candidates["adam"] = torch.optim.Adam

    if opt_name in ("adam", "adamw"):
        cls = candidates[opt_name]
    else:
        cls = candidates.get("adamw", candidates["adam"])

    opt = cls(params, lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    print(f"[opt] {opt.__class__.__module__}.{opt.__class__.__name__}")
    return opt


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute mean relative L2 error over batch samples."""
    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1)
    tgt = target.reshape(b, -1)
    rel = torch.linalg.norm(diff, dim=1) / torch.linalg.norm(tgt, dim=1).clamp_min(eps)
    return rel.mean()


class GridSampleDataset(Dataset):
    """Dataset loader for field samples stored under one preset directory."""
    def __init__(self, preset_root: Path, split: str, use_normalized: bool = True):
        """Load split file list and store absolute sample paths for one preset split."""
        split_info = json.loads((preset_root / "split_files.json").read_text())
        rels = split_info[split]
        self.paths = [preset_root / rel for rel in rels]
        self.use_normalized = use_normalized

    def __len__(self) -> int:
        """Return number of structured-grid samples in this split."""
        return len(self.paths)

    def __getitem__(self, idx: int):
        """Load one sample and return `(input_tensor, target_tensor)` as torch tensors."""
        p = self.paths[idx]
        with np.load(p, allow_pickle=True) as d:
            x_key = "input_norm" if self.use_normalized and "input_norm" in d.files else "input"
            y_key = "target_norm" if self.use_normalized and "target_norm" in d.files else "target"
            x = np.asarray(d[x_key], dtype=np.float32)
            y = np.asarray(d[y_key], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def build_field_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate NeuralOperator FNO with compatibility handling across API variants."""
    try:
        from neuralop.models import FNO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Field model requires neuralop package (`pip install neuraloperator`)."
        ) from exc

    sig = inspect.signature(FNO.__init__)
    params = sig.parameters

    fno_kwargs = {
        "n_modes": tuple(cfg.get("n_modes", [12, 12, 12])),
        "in_channels": int(cfg["in_channels"]),
        "out_channels": int(cfg["out_channels"]),
        "hidden_channels": int(cfg.get("hidden_channels", 20)),
    }

    if "n_layers" in params:
        fno_kwargs["n_layers"] = int(cfg.get("n_layers", 4))
    if "non_linearity" in params:
        fno_kwargs["non_linearity"] = torch.nn.functional.relu
    elif "activation" in params:
        fno_kwargs["activation"] = torch.nn.functional.relu
    elif "act" in params:
        fno_kwargs["act"] = "relu"

    if "lifting_channels" in params:
        fno_kwargs["lifting_channels"] = int(cfg.get("kernel_width", 256))
    if "projection_channels" in params:
        fno_kwargs["projection_channels"] = int(cfg.get("kernel_width", 256))

    # Explicitly disable BN-like options when available.
    if "norm" in params:
        fno_kwargs["norm"] = None
    if "fno_norm" in params:
        fno_kwargs["fno_norm"] = None
    if "batch_norm" in params:
        fno_kwargs["batch_norm"] = False
    if "use_batch_norm" in params:
        fno_kwargs["use_batch_norm"] = False

    model = FNO(**fno_kwargs).to(device)
    return model


def _eval_field(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate field model on a loader and report relative L2 and MSE metrics."""
    model.eval()
    rel_sum, mse_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            rel = relative_l2_loss(p, y)
            mse = torch.mean((p - y) ** 2)
            bs = x.shape[0]
            rel_sum += float(rel.item()) * bs
            mse_sum += float(mse.item()) * bs
            n += bs
    return {
        "rel_l2": rel_sum / max(1, n),
        "mse": mse_sum / max(1, n),
    }


def train_field_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end training loop for the structured-grid field reconstruction model (FNO)."""
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg)

    preset_root = Path(cfg["preset_root"])
    train_ds = GridSampleDataset(preset_root, "train", use_normalized=bool(cfg.get("use_normalized", True)))
    val_ds = GridSampleDataset(preset_root, "val", use_normalized=bool(cfg.get("use_normalized", True)))
    test_ds = GridSampleDataset(preset_root, "test", use_normalized=bool(cfg.get("use_normalized", True)))

    if len(train_ds) == 0:
        raise RuntimeError("No train samples found")

    # infer channels from first sample
    x0, y0 = train_ds[0]
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["in_channels"] = int(x0.shape[0])
    model_cfg["out_channels"] = int(y0.shape[0])

    model = build_field_model(model_cfg, device=device)

    opt_cfg = dict(cfg.get("optimizer_cfg", {}))
    opt = make_adam_optimizer(model.parameters(), opt_cfg)
    sch = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=int(opt_cfg.get("lr_step_size", 100)),
        gamma=float(opt_cfg.get("lr_gamma", 0.5)),
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=False)

    epochs = int(cfg.get("epochs", 500))
    loss_name = str(cfg.get("loss", "relative_l2")).lower()

    hist: List[Dict[str, float]] = []
    best_val = np.inf
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            p = model(x)
            if loss_name == "mse":
                loss = torch.mean((p - y) ** 2)
            else:
                loss = relative_l2_loss(p, y)
            loss.backward()
            opt.step()
            bs = x.shape[0]
            running += float(loss.item()) * bs
            n += bs

        sch.step()

        train_loss = running / max(1, n)
        val_metrics = _eval_field(model, val_loader, device)
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_rel_l2": val_metrics["rel_l2"],
            "val_mse": val_metrics["mse"],
        }
        hist.append(row)

        if val_metrics["rel_l2"] < best_val:
            best_val = val_metrics["rel_l2"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if ep % int(cfg.get("log_every", 10)) == 0 or ep == 1:
            print(
                f"[field][{ep:04d}] train={train_loss:.6f} "
                f"val_rel={val_metrics['rel_l2']:.6f} val_mse={val_metrics['mse']:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _eval_field(model, test_loader, device)

    out_dir = ensure_dir(Path(cfg.get("output_dir", preset_root / "training")))
    torch.save(model.state_dict(), out_dir / "best_field_model.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    summary = {
        "best_val_rel_l2": float(best_val),
        "test_rel_l2": float(test_metrics["rel_l2"]),
        "test_mse": float(test_metrics["mse"]),
        "epochs": epochs,
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[field] done", json.dumps(summary, indent=2))
    return summary


# ---------------------------
# Particle Graph Neural Operator (GNO)
# ---------------------------


class ParticleFrameDataset(Dataset):
    """One item = all particles from one frame."""

    def __init__(self, dataset_npz: Path, split_idx_npz: Path, split: str):
        """Load particle dataset and keep frame-wise row ranges for a specific split."""
        ds = np.load(dataset_npz, allow_pickle=True)
        self.x = np.asarray(ds["inputs_particle"], dtype=np.float32)
        self.y = np.asarray(ds["targets_particle"], dtype=np.float32)
        self.frame_offsets = list(ds["frame_offsets"])

        idx_npz = np.load(split_idx_npz, allow_pickle=False)
        split_idx = set(np.asarray(idx_npz[f"{split}_idx"], dtype=np.int64).tolist())

        self.ranges: List[Tuple[int, int]] = []
        for rec in self.frame_offsets:
            _, _, start, end = rec
            start_i = int(start)
            end_i = int(end)
            if start_i in split_idx:
                self.ranges.append((start_i, end_i))

    def __len__(self) -> int:
        """Return number of particle frames in this split."""
        return len(self.ranges)

    def __getitem__(self, idx: int):
        """Return all particles for one frame as `(X_frame, Y_frame)` tensors."""
        s, e = self.ranges[idx]
        return (
            torch.from_numpy(self.x[s:e]),
            torch.from_numpy(self.y[s:e]),
        )


def particle_collate(batch):
    # Keep variable-size frames as lists.
    """Custom collate that keeps variable-length per-frame particle tensors as Python lists."""
    xs, ys = zip(*batch)
    return list(xs), list(ys)


class ParticleGNOModel(nn.Module):
    """Stacked GNOBlock model for particle-to-particle regression."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        radius: float = 0.2,
        transform_type: str = "linear",
        reduction: str = "mean",
        pos_embedding_type: str = "transformer",
        pos_embedding_channels: int = 16,
        use_open3d_neighbor_search: bool = False,
        use_torch_scatter_reduce: bool = False,
    ):
        """Construct encoder + stacked GNOBlock operator layers + regression head."""
        super().__init__()

        try:
            from neuralop.layers.gno_block import GNOBlock  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Particle GNO model requires neuralop GNOBlock. "
                "Install/activate neuralop environment where `neuralop.layers.gno_block` exists."
            ) from exc

        if in_dim < 3:
            raise ValueError("Particle features must include x,y,z as the first 3 channels")

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                GNOBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    coord_dim=3,
                    radius=float(radius),
                    transform_type=transform_type,
                    reduction=reduction,
                    pos_embedding_type=pos_embedding_type,
                    pos_embedding_channels=int(pos_embedding_channels),
                    channel_mlp_layers=[hidden_dim, hidden_dim * 2, hidden_dim],
                    use_torch_scatter_reduce=bool(use_torch_scatter_reduce),
                    use_open3d_neighbor_search=bool(use_open3d_neighbor_search),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(n_layers))])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First three channels are expected to be particle coordinates.
        """Predict particle targets for one frame using coordinates as graph positions."""
        pos = x[:, :3]
        h = self.encoder(x)

        for block, norm in zip(self.blocks, self.norms):
            update = block(y=pos, x=pos, f_y=h)
            if update.ndim == 3 and update.shape[0] == 1:
                update = update.squeeze(0)
            h = norm(h + update)

        return self.head(h)


def _eval_particle(model, loader, device, max_nodes: int = 4096):
    """Evaluate the particle GNO model on frame-wise particle batches."""
    model.eval()
    rel_sum, mse_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xs, ys in loader:
            for x, y in zip(xs, ys):
                if x.shape[0] > max_nodes:
                    idx = torch.randperm(x.shape[0])[:max_nodes]
                    x = x[idx]
                    y = y[idx]

                x = x.to(device)
                y = y.to(device)
                p = model(x)
                rel = relative_l2_loss(p.unsqueeze(0), y.unsqueeze(0))
                mse = torch.mean((p - y) ** 2)
                rel_sum += float(rel.item())
                mse_sum += float(mse.item())
                n += 1
    return {
        "rel_l2": rel_sum / max(1, n),
        "mse": mse_sum / max(1, n),
    }


def train_particle_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end training loop for particle surrogate model built from GNO blocks."""
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg)

    dataset_npz = Path(cfg["dataset_npz"])
    split_idx_npz = Path(cfg["split_idx_npz"])

    train_ds = ParticleFrameDataset(dataset_npz, split_idx_npz, "train")
    val_ds = ParticleFrameDataset(dataset_npz, split_idx_npz, "val")
    test_ds = ParticleFrameDataset(dataset_npz, split_idx_npz, "test")

    if len(train_ds) == 0:
        raise RuntimeError("No particle train frames found")

    x0, y0 = train_ds[0]
    model = ParticleGNOModel(
        in_dim=int(x0.shape[1]),
        out_dim=int(y0.shape[1]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        n_layers=int(cfg.get("n_layers", 4)),
        radius=float(cfg.get("radius", 0.2)),
        transform_type=str(cfg.get("transform_type", "linear")),
        reduction=str(cfg.get("reduction", "mean")),
        pos_embedding_type=str(cfg.get("pos_embedding_type", "transformer")),
        pos_embedding_channels=int(cfg.get("pos_embedding_channels", 16)),
        use_open3d_neighbor_search=bool(cfg.get("use_open3d_neighbor_search", False)),
        use_torch_scatter_reduce=bool(cfg.get("use_torch_scatter_reduce", False)),
    ).to(device)

    opt = make_adam_optimizer(model.parameters(), cfg.get("optimizer_cfg", {}))
    sch = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=int(cfg.get("optimizer_cfg", {}).get("lr_step_size", 100)),
        gamma=float(cfg.get("optimizer_cfg", {}).get("lr_gamma", 0.5)),
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=particle_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=particle_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=particle_collate)

    epochs = int(cfg.get("epochs", 200))
    max_nodes = int(cfg.get("max_nodes_per_frame", 4096))

    best_val = np.inf
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for xs, ys in train_loader:
            x = xs[0]
            y = ys[0]
            if x.shape[0] > max_nodes:
                idx = torch.randperm(x.shape[0])[:max_nodes]
                x = x[idx]
                y = y[idx]

            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = relative_l2_loss(p.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1

        sch.step()

        tr = running / max(1, n)
        val_metrics = _eval_particle(model, val_loader, device, max_nodes=max_nodes)
        hist.append({"epoch": ep, "train_loss": tr, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["rel_l2"] < best_val:
            best_val = val_metrics["rel_l2"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if ep % int(cfg.get("log_every", 10)) == 0 or ep == 1:
            print(f"[particle-gno][{ep:04d}] train={tr:.6f} val_rel={val_metrics['rel_l2']:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _eval_particle(model, test_loader, device, max_nodes=max_nodes)

    out_dir = ensure_dir(Path(cfg.get("output_dir", "final/output/particle_training")))
    torch.save(model.state_dict(), out_dir / "best_particle_gno_model.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))

    summary = {
        "best_val_rel_l2": float(best_val),
        "test_rel_l2": float(test_metrics["rel_l2"]),
        "test_mse": float(test_metrics["mse"]),
        "epochs": epochs,
        "device": str(device),
        "n_train_frames": len(train_ds),
        "n_val_frames": len(val_ds),
        "n_test_frames": len(test_ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[particle-gno] done", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI wrapper for running either field-task or particle-task training."""
    parser = argparse.ArgumentParser(description="Train field or particle model on final dataset")
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON config")
    parser.add_argument("--task", type=str, choices=["field", "particle"], default="field")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.task == "field":
        train_field_model(cfg)
    else:
        train_particle_model(cfg)


if __name__ == "__main__":
    main()
