#!/usr/bin/env python
# coding: utf-8

# # Simple Task-1 U/gradU Training
# 
# This notebook is intentionally small. It focuses only on the particle `U, gradU` task and prints the few diagnostics needed to catch the `e11` validation/test relative-error problem.
# 
# Main safeguards:
# - drop zero/tiny-target startup frames before training/evaluation,
# - recompute normalization from the cleaned training frames,
# - skip tiny-target frames when reporting relative L2,
# - use variance-normalized physical loss for `u` and `gradU`,
# - use a simple increasing particle cap.

# In[ ]:


from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from neuralop.layers.neighbor_search import NeighborSearch
except Exception as error:
    NeighborSearch = None
    NEIGHBOR_ERROR = error

try:
    from neuralop.utils import count_model_params
except Exception:
    def count_model_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


# ## Config
# 
# Short names on purpose. Override any value from the shell/notebook environment, for example `EPOCHS=10` or `P_FINAL=2048`.

# In[ ]:


SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path.cwd().resolve()
if (ROOT / "FINAL").is_dir():
    FINAL = ROOT / "FINAL"
elif ROOT.name == "FINAL":
    FINAL = ROOT
    ROOT = FINAL.parent
else:
    FINAL = ROOT / "FINAL"

DATA_ENV = os.environ.get("DATA", "").strip()
if DATA_ENV:
    DATA = Path(DATA_ENV).expanduser()
else:
    choices = [
        FINAL / "processed_data" / "particle_ugradu_dataset.npz",
        ROOT / "final-2" / "processed_data_task1" / "particle_ugradu_dataset.npz",
    ]
    DATA = next((p for p in choices if p.exists()), choices[0])

OUT = FINAL / "result" / "task1"
OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("TAG", "task1_particle_ugradu_gino_pointwise")
BEST = OUT / f"{TAG}_best_model.pt"
LAST = OUT / f"{TAG}_last_model.pt"
HIST = OUT / f"{TAG}_history.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "epochs": int(os.environ.get("EPOCHS", "60")),
    "lr": float(os.environ.get("LR", "3e-4")),
    "min_lr": float(os.environ.get("MIN_LR", "5e-6")),
    "wd": float(os.environ.get("WD", "3e-5")),
    "accum": int(os.environ.get("ACCUM", "4")),
    "clip": float(os.environ.get("CLIP", "1.0")),
    "p_start": int(os.environ.get("P_START", "1024")),
    "p_final": int(os.environ.get("P_FINAL", "4096")),
    "eval_frames": int(os.environ.get("EVAL_FRAMES", "16")),
    "eval_every": int(os.environ.get("EVAL_EVERY", "2")),
    "skip_norm": float(os.environ.get("SKIP_NORM", "1e-6")),
    "drop_norm": float(os.environ.get("DROP_NORM", "1e-12")),
    "hidden": int(os.environ.get("HIDDEN", "128")),
    "layers": int(os.environ.get("LAYERS", "3")),
    "radius": float(os.environ.get("RADIUS", "0.12")),
    "dropout": float(os.environ.get("DROPOUT", "0.06")),
    "amp": os.environ.get("AMP", "1").lower() in {"1", "true", "yes"},
}

FEATURES = [
    "Gamma_x", "Gamma_y", "Gamma_z", "sigma",
    "geom_dist", "geom_nx", "geom_ny", "geom_nz", "geom_body_near",
    "angle_of_attack", "freestream_x", "freestream_z", "phase",
]

print("=" * 72)
print("DATA  :", DATA)
print("OUT   :", OUT)
print("HIST  :", HIST)
print("DEVICE:", DEVICE)
print("CFG   :", CFG)
print("=" * 72)


# ## Load Data And Print Split Health
# 
# This cell is the important debugging gate. If `val_angle` or `test` reports dropped zero/tiny frames, those frames would have caused the old `e11` relative-L2 values.

# In[ ]:


def load_npz(path: Path):
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as error:
        if "numpy._core" not in str(error):
            raise
        import numpy.core as numpy_core
        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        return np.load(path, allow_pickle=True)


def split_ids(ds, key: str) -> np.ndarray:
    return np.asarray(ds[key], dtype=np.int64) if key in ds.files else np.zeros(0, dtype=np.int64)


def choose(n: int, cap: Optional[int], seed: int) -> np.ndarray:
    if cap is None or cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, int(cap), replace=False)).astype(np.int64)


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def ctx(ds, fid: int) -> dict:
    if "frame_contexts" not in ds.files:
        return {}
    item = ds["frame_contexts"][int(fid)]
    if isinstance(item, dict):
        return item
    if hasattr(item, "item"):
        obj = item.item()
        if isinstance(obj, dict):
            return obj
    try:
        return dict(item)
    except Exception:
        return {}


ds = load_npz(DATA)
all_features = [str(x) for x in ds["feature_names"].tolist()]
target_names = [str(x) for x in ds["target_names"].tolist()]
features = [f for f in FEATURES if f in all_features]
missing_features = [f for f in FEATURES if f not in all_features]
feat_idx = [all_features.index(f) for f in features]
xyz_idx = [all_features.index(f) for f in ("x", "y", "z")]

if missing_features:
    print("[warn] missing optional features:", missing_features)

coord_min = np.asarray(ds["coord_min"], dtype=np.float32).reshape(3) if "coord_min" in ds.files else np.min(ds["inputs_t"][:, xyz_idx], axis=0)
coord_span = np.maximum(np.asarray(ds["coord_span"], dtype=np.float32).reshape(3) if "coord_span" in ds.files else np.ptp(ds["inputs_t"][:, xyz_idx], axis=0), 1e-8)


def norm_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.clip((xyz.astype(np.float32) - coord_min[None, :]) / coord_span[None, :], 0.0, 1.0).astype(np.float32)


def frame_target_norm(fid: int) -> float:
    y = np.asarray(ds["targets_by_frame"][int(fid)], dtype=np.float32)
    return float(np.linalg.norm(np.nan_to_num(y).reshape(-1)))


def clean_frames(ids: np.ndarray, name: str) -> np.ndarray:
    keep, bad = [], []
    norms = []
    for fid in np.asarray(ids, dtype=np.int64):
        nrm = frame_target_norm(int(fid))
        if np.isfinite(nrm) and nrm > CFG["drop_norm"]:
            keep.append(int(fid))
            norms.append(nrm)
        else:
            c = ctx(ds, int(fid))
            bad.append({"fid": int(fid), "case": c.get("case", "?"), "frame": c.get("frame", "?"), "norm": nrm})
    if norms:
        q = np.quantile(np.asarray(norms), [0.0, 0.01, 0.5, 0.95, 1.0])
        qtxt = f"norm[min,p01,p50,p95,max]=[{q[0]:.2e}, {q[1]:.2e}, {q[2]:.2e}, {q[3]:.2e}, {q[4]:.2e}]"
    else:
        qtxt = "no kept frames"
    print(f"{name:12s}: kept={len(keep):4d} dropped={len(bad):3d} {qtxt}")
    if bad:
        print("  dropped examples:", bad[:5])
    return np.asarray(keep, dtype=np.int64)

train_ids = clean_frames(split_ids(ds, "train_frame_ids"), "train")
val_ids = clean_frames(split_ids(ds, "val_id_frame_ids"), "val_id")
val_ang_ids = clean_frames(split_ids(ds, "validation_angle_frame_ids"), "val_angle")
test_ids = clean_frames(split_ids(ds, "test_normal_frame_ids"), "test")

print("features:", features)
print("targets :", target_names)


# ## Recompute Normalization From Clean Train Frames
# 
# This avoids using old normalization stats that may include zero-target startup frames.

# In[ ]:


def train_stats(frame_ids: np.ndarray):
    x_sum = np.zeros(len(feat_idx), dtype=np.float64)
    x_sq = np.zeros(len(feat_idx), dtype=np.float64)
    y_sum = np.zeros(len(target_names), dtype=np.float64)
    y_sq = np.zeros(len(target_names), dtype=np.float64)
    count = 0
    for fid in frame_ids:
        x_all = np.asarray(ds["inputs_by_frame"][int(fid)], dtype=np.float32)[:, feat_idx]
        y_all = np.asarray(ds["targets_by_frame"][int(fid)], dtype=np.float32)
        n = min(len(x_all), len(y_all))
        x = x_all[:n].astype(np.float64)
        y = y_all[:n].astype(np.float64)
        x_sum += x.sum(axis=0)
        x_sq += np.square(x).sum(axis=0)
        y_sum += y.sum(axis=0)
        y_sq += np.square(y).sum(axis=0)
        count += n
    x_mean = x_sum / max(count, 1)
    y_mean = y_sum / max(count, 1)
    x_std = np.sqrt(np.maximum(x_sq / max(count, 1) - x_mean**2, 1e-8))
    y_std = np.sqrt(np.maximum(y_sq / max(count, 1) - y_mean**2, 1e-8))
    return x_mean.astype(np.float32), x_std.astype(np.float32), y_mean.astype(np.float32), y_std.astype(np.float32)

in_mean, in_std, out_mean, out_std = train_stats(train_ids)

# scalar task scales for balanced physical loss
u_std = float(np.sqrt(np.mean(out_std[:3] ** 2)))
g_std = float(np.sqrt(np.mean(out_std[3:] ** 2)))
print(f"normalization recomputed from {len(train_ids)} clean train frames")
print(f"u_std={u_std:.4e} gradU_std={g_std:.4e}")
print("input mean/std examples:")
for name, mu, sd in list(zip(features, in_mean, in_std))[:8]:
    print(f"  {name:16s} mean={mu: .3e} std={sd: .3e}")


# ## Dataset

# In[ ]:


class Frames(Dataset):
    def __init__(self, ids: np.ndarray, cap: int):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.cap = int(cap)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = int(self.ids[int(i)])
        x_all = np.asarray(ds["inputs_by_frame"][fid], dtype=np.float32)
        y_all = np.asarray(ds["targets_by_frame"][fid], dtype=np.float32)
        n = min(len(x_all), len(y_all))
        idx = choose(n, self.cap, SEED + fid)
        x_raw = x_all[idx]
        y_raw = y_all[idx]
        x = (x_raw[:, feat_idx] - in_mean[None, :]) / in_std[None, :]
        y = (y_raw - out_mean[None, :]) / out_std[None, :]
        return {
            "pos": torch.from_numpy(norm_xyz(x_raw[:, xyz_idx])),
            "pos_raw": torch.from_numpy(x_raw[:, xyz_idx].astype(np.float32)),
            "x": torch.from_numpy(np.clip(np.nan_to_num(x), -8.0, 8.0).astype(np.float32)),
            "y": torch.from_numpy(np.nan_to_num(y).astype(np.float32)),
            "fid": torch.tensor(fid, dtype=torch.long),
        }


def collate_one(batch):
    return batch[0]

train_ds = Frames(train_ids, CFG["p_start"])
val_ds = Frames(val_ids, CFG["p_final"])
val_ang_ds = Frames(val_ang_ids, CFG["p_final"])
test_ds = Frames(test_ids, CFG["p_final"])
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_one)

sample = train_ds[min(5, len(train_ds)-1)]
print("sample x/y/pos:", tuple(sample["x"].shape), tuple(sample["y"].shape), tuple(sample["pos"].shape))
print("sample target norm:", float(torch.linalg.norm(sample["y"].reshape(-1))))


# ## Model
# 
# A compact vortex-inspired edge GNO. It uses normalized coordinates for neighbor search and raw coordinates for distance edge features.

# In[ ]:


class EdgeBlock(nn.Module):
    def __init__(self, hidden: int, radius: float, dropout: float):
        super().__init__()
        self.radius = float(radius)
        self.search = NeighborSearch(use_open3d=False, return_norm=False) if NeighborSearch is not None else None
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden + 14, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def neighbors(self, pos: torch.Tensor):
        if self.search is not None:
            nb = self.search(data=pos, queries=pos, radius=self.radius)
            src = nb["neighbors_index"].long()
            splits = nb["neighbors_row_splits"].long()
            counts = splits[1:] - splits[:-1]
            dst = torch.repeat_interleave(torch.arange(pos.shape[0], device=pos.device), counts)
            return src, dst, counts
        dist = torch.cdist(pos, pos)
        mask = dist <= self.radius
        dst, src = torch.where(mask)
        counts = mask.sum(dim=1).long()
        return src.long(), dst.long(), counts

    def forward(self, h, pos, raw_pos, gamma, sigma):
        src, dst, counts = self.neighbors(pos)
        if src.numel() == 0:
            return h
        dr = raw_pos[src] - raw_pos[dst]
        r = torch.linalg.norm(dr, dim=-1, keepdim=True)
        sig_s = sigma[src].abs().clamp_min(1e-8)
        sig_d = sigma[dst].abs().clamp_min(1e-8)
        edge = torch.cat([h[src], h[dst], dr, r, r / sig_s, r / sig_d, gamma[src], gamma[dst], sig_s, sig_d], dim=-1)
        msg = self.mlp(edge)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, msg)
        out = out / counts.to(h.dtype).clamp_min(1).unsqueeze(-1)
        return self.norm(h + out)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = CFG["hidden"]
        self.sigma_i = features.index("sigma")
        self.gamma_i = [features.index(k) for k in ("Gamma_x", "Gamma_y", "Gamma_z")]
        self.register_buffer("mu", torch.tensor(in_mean).view(1, -1))
        self.register_buffer("sd", torch.tensor(in_std).view(1, -1))
        self.lift = nn.Sequential(
            nn.Linear(len(features), hidden), nn.GELU(), nn.Dropout(CFG["dropout"]),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([EdgeBlock(hidden, CFG["radius"], CFG["dropout"]) for _ in range(CFG["layers"])])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(CFG["dropout"]), nn.Linear(hidden, len(target_names)))

    def forward(self, batch):
        x = batch["x"]
        physical_x = x * self.sd.to(x.device) + self.mu.to(x.device)
        gamma = physical_x[:, self.gamma_i]
        sigma = physical_x[:, self.sigma_i:self.sigma_i + 1]
        h = self.lift(x)
        for block in self.blocks:
            h = block(h, batch["pos"], batch["pos_raw"], gamma, sigma)
        return self.head(h)

model = Net().to(DEVICE)
print("params:", f"{count_model_params(model):,}")


# ## Train And Evaluate

# In[ ]:


out_mean_t = torch.tensor(out_mean, dtype=torch.float32, device=DEVICE).view(1, -1)
out_std_t = torch.tensor(out_std, dtype=torch.float32, device=DEVICE).view(1, -1)
u_std_t = torch.tensor(u_std, dtype=torch.float32, device=DEVICE)
g_std_t = torch.tensor(g_std, dtype=torch.float32, device=DEVICE)


def to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def denorm(y):
    return y * out_std_t + out_mean_t


def loss_fn(pred, y):
    pred_p = denorm(pred)
    y_p = denorm(y)
    u_loss = torch.mean((pred_p[:, :3] - y_p[:, :3]) ** 2) / u_std_t.clamp_min(1e-12).pow(2)
    g_loss = torch.mean((pred_p[:, 3:] - y_p[:, 3:]) ** 2) / g_std_t.clamp_min(1e-12).pow(2)
    return u_loss + g_loss, u_loss.detach(), g_loss.detach()


def rel_rmse(pred, y):
    pred_p = denorm(pred)
    y_p = denorm(y)
    target_norm = torch.linalg.norm(y_p.reshape(-1))
    rmse = torch.sqrt(torch.mean((pred_p - y_p) ** 2)).item()
    if target_norm.item() <= CFG["skip_norm"]:
        return np.nan, rmse, True
    rel = (torch.linalg.norm((pred_p - y_p).reshape(-1)) / target_norm).item()
    return rel, rmse, False


@torch.no_grad()
def eval_split(dataset, name: str):
    if len(dataset) == 0:
        return {"rel": np.nan, "rmse": np.nan, "n": 0, "skip": 0}
    model.eval()
    ids = np.arange(len(dataset))
    if len(ids) > CFG["eval_frames"]:
        ids = np.sort(np.random.default_rng(SEED + 99).choice(ids, CFG["eval_frames"], replace=False))
    rels, rmses, skips = [], [], 0
    for i in ids:
        batch = to_device(dataset[int(i)])
        pred = model(batch)
        rel, rmse, skipped = rel_rmse(pred, batch["y"])
        rels.append(rel)
        rmses.append(rmse)
        skips += int(skipped)
    out = {"rel": finite_mean(rels), "rmse": finite_mean(rmses), "n": int(len(ids)), "skip": int(skips)}
    print(f"eval {name:10s}: rel={out['rel']:.4g} rmse={out['rmse']:.4g} frames={out['n']} skipped={out['skip']}")
    return out

opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=CFG["min_lr"])
use_amp = DEVICE.type == "cuda" and CFG["amp"]
dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

history = defaultdict(list)
best = float("inf")
for epoch in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    cap = round(CFG["p_start"] + (CFG["p_final"] - CFG["p_start"]) * (epoch - 1) / max(CFG["epochs"] - 1, 1))
    train_ds.cap = int(cap)
    model.train()
    opt.zero_grad(set_to_none=True)
    losses, u_losses, g_losses = [], [], []
    bad = 0
    steps = 0

    for step, batch0 in enumerate(train_loader, 1):
        batch = to_device(batch0)
        with torch.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
            pred = model(batch)
            loss, u_loss, g_loss = loss_fn(pred, batch["y"])
            loss_back = loss / CFG["accum"]
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            continue
        if scaler.is_enabled():
            scaler.scale(loss_back).backward()
        else:
            loss_back.backward()
        if step % CFG["accum"] == 0 or step == len(train_loader):
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
        losses.append(float(loss.item()))
        u_losses.append(float(u_loss.item()))
        g_losses.append(float(g_loss.item()))

    sched.step()
    if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
        val = eval_split(val_ds, "val_id")
        val_angle = eval_split(val_ang_ds, "val_angle")
        test = eval_split(test_ds, "test")
        history["epoch"].append(epoch)
        history["loss"].append(finite_mean(losses))
        history["u_loss"].append(finite_mean(u_losses))
        history["g_loss"].append(finite_mean(g_losses))
        history["val_rel"].append(val["rel"])
        history["val_angle_rel"].append(val_angle["rel"])
        history["test_rel"].append(test["rel"])
        history["val_angle_skip"].append(val_angle["skip"])
        history["test_skip"].append(test["skip"])
        history["cap"].append(int(cap))
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        HIST.write_text(json.dumps(history, indent=2))

        payload = {
            "cfg": CFG,
            "features": features,
            "targets": target_names,
            "model": model.state_dict(),
            "in_mean": in_mean,
            "in_std": in_std,
            "out_mean": out_mean,
            "out_std": out_std,
            "coord_min": coord_min,
            "coord_span": coord_span,
            "u_std": u_std,
            "g_std": g_std,
            "history": dict(history),
        }
        torch.save(payload, LAST)
        if np.isfinite(val["rel"]) and val["rel"] < best:
            best = val["rel"]
            torch.save(payload, BEST)
        print(
            f"epoch {epoch:03d}: loss={history['loss'][-1]:.4g} "
            f"u={history['u_loss'][-1]:.4g} g={history['g_loss'][-1]:.4g} "
            f"cap={cap} bad={bad} steps={steps} lr={history['lr'][-1]:.2e} "
            f"time={time.time()-t0:.1f}s"
        )

print("best val_rel:", best)
print("history:", HIST)
print("best:", BEST)

