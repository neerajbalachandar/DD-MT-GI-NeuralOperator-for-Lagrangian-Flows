#!/usr/bin/env python
# coding: utf-8

# # Task2 v2 GINO Analysis
# 
# Loads the trained `results/task2_gino_best_model.pt` checkpoint, reconstructs the `neuralop.models.GINO` model, and reproduces the Task1-v2 style diagnostics: learning curves, split metrics, parity/error plots, qualitative Eulerian slices, timeline panels, data profile, data-hunger curve, and compact hyperparameter sweeps.

# In[ ]:


from pathlib import Path
import json
import math
import os
import random
import inspect
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

try:
    from neuralop.models import GINO
except Exception as error:
    raise RuntimeError('This notebook requires NeuralOperator in the active kernel.') from error

try:
    from neuralop.utils import count_model_params
except Exception:
    def count_model_params(model):
        return sum(p.numel() for p in model.parameters())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

current_directory = Path.cwd().resolve()
if (current_directory / 'final-2').is_dir():
    REPO_ROOT = current_directory
    FINAL_DIR = current_directory / 'final-2'
elif current_directory.name == 'final-2':
    FINAL_DIR = current_directory
    REPO_ROOT = current_directory.parent
elif current_directory.parent.name == 'final-2':
    FINAL_DIR = current_directory.parent
    REPO_ROOT = current_directory.parent.parent
else:
    REPO_ROOT = current_directory
    FINAL_DIR = current_directory / 'final-2'

RESULTS_DIR = FINAL_DIR / 'result' / 'task2'
CKPT_PATH = RESULTS_DIR / 'task2_gino_best_model.pt'
if not CKPT_PATH.exists():
    fallback = RESULTS_DIR / 'task2_gino_last_model.pt'
    if fallback.exists():
        CKPT_PATH = fallback
    else:
        raise FileNotFoundError(f'Missing checkpoint in {RESULTS_DIR}. Run task2_v2_GINO.ipynb training first.')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

CFG = dict(checkpoint['config'])

def resolve_dataset_path(raw_value):
    raw = Path(str(raw_value))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(FINAL_DIR / raw)
        candidates.append(FINAL_DIR / 'processed_data_task2' / raw.name)
        candidates.append(FINAL_DIR / 'processed_data' / 'task2' / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Could not resolve dataset path from checkpoint: {raw}')

DATASET_PATH = resolve_dataset_path(checkpoint['dataset_path'])
print('Checkpoint :', CKPT_PATH)
print('Dataset    :', DATASET_PATH)
print('Results dir:', RESULTS_DIR)
print('Device     :', DEVICE)


# In[ ]:


dataset_file = np.load(DATASET_PATH, allow_pickle=True)


def resolve_sample_path(path_like):
    raw = Path(str(path_like))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
        if 'task2_gino_frames' in raw.parts:
            candidates.append(DATASET_PATH.parent / 'task2_gino_frames' / raw.name)
    else:
        candidates.append(DATASET_PATH.parent / raw)
        candidates.append(DATASET_PATH.parent / 'task2_gino_frames' / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Sample path in manifest is missing: {raw}')

sample_paths = [resolve_sample_path(p) for p in dataset_file['sample_paths'].tolist()]
frame_contexts = list(dataset_file['frame_contexts'])
feature_names_all = [str(x) for x in dataset_file['feature_names'].tolist()]
feature_names = [str(x) for x in checkpoint['feature_names']]
target_names = [str(x) for x in checkpoint['target_names']]
active_input_feature_indices = [int(i) for i in checkpoint['active_input_feature_indices']]
removed_input_features = [str(x) for x in checkpoint.get('removed_input_features', [])]


def manifest_ids(key):
    if key in dataset_file.files:
        return dataset_file[key].astype(np.int64)
    print(f'[info] manifest key {key!r} is missing; using an empty split.')
    return np.zeros((0,), dtype=np.int64)

train_frame_ids = manifest_ids('train_frame_ids')
val_id_frame_ids = manifest_ids('val_id_frame_ids')
val_angle_frame_ids = manifest_ids('val_angle_frame_ids')
test_normal_frame_ids = manifest_ids('test_normal_frame_ids')
test_spatial_sr_frame_ids = manifest_ids('test_spatial_sr_frame_ids')
test_temporal_sr_frame_ids = manifest_ids('test_temporal_sr_frame_ids')
test_unseen_angle_frame_ids = manifest_ids('test_unseen_angle_frame_ids')
testing_frame_ids = test_normal_frame_ids if len(test_normal_frame_ids) > 0 else val_angle_frame_ids

input_mean = np.asarray(checkpoint['input_mean'], dtype=np.float32).reshape(-1)
input_std = np.maximum(np.asarray(checkpoint['input_std'], dtype=np.float32).reshape(-1), 1e-8)
target_mean = np.asarray(checkpoint['target_mean'], dtype=np.float32).reshape(-1)
target_std = np.asarray(checkpoint['target_std'], dtype=np.float32).reshape(-1)
target_mean = np.nan_to_num(target_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
target_std = np.maximum(np.nan_to_num(target_std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32), 1e-8)
coord_min = np.asarray(checkpoint['coord_min'], dtype=np.float32).reshape(3)
coord_span = np.maximum(np.asarray(checkpoint['coord_span'], dtype=np.float32).reshape(3), 1e-8)
grid_resolution = tuple(int(x) for x in checkpoint['grid_resolution'])
history = checkpoint.get('history', {})

print('Samples:', len(sample_paths))
print('Grid resolution:', grid_resolution)
print('Model input features:', feature_names)
print('Removed input features:', removed_input_features if removed_input_features else 'none')
print('Targets:', target_names)
print('Split counts:', {
    'train': len(train_frame_ids),
    'val_id': len(val_id_frame_ids),
    'val_angle': len(val_angle_frame_ids),
    'test': len(testing_frame_ids),
    'test_spatial_sr': len(test_spatial_sr_frame_ids),
    'test_temporal_sr': len(test_temporal_sr_frame_ids),
    'test_unseen': len(test_unseen_angle_frame_ids),
})


# In[ ]:


def save_fig(fig, name, dpi=220):
    out = RESULTS_DIR / f"{CFG.get('file_tag', 'task2_gino')}_{name}"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    print('[saved]', out)


def sample_indices(n, cap, seed):
    if cap is None or cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(cap), replace=False)).astype(np.int64)


def normalize_xyz(xyz):
    return np.clip((xyz.astype(np.float32) - coord_min[None, :]) / coord_span[None, :], 0.0, 1.0).astype(np.float32)


class Task2GINODataset(Dataset):
    def __init__(self, frame_ids, split_name, max_input_particles=None, max_output_points=None):
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.split_name = str(split_name)
        self.max_input_particles = max_input_particles
        self.max_output_points = max_output_points

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = int(self.frame_ids[int(idx)])
        with np.load(sample_paths[frame_id], allow_pickle=True) as d:
            input_geom_raw = np.asarray(d['input_geom'], dtype=np.float32)
            features_raw = np.asarray(d['input_features'], dtype=np.float32)[:, active_input_feature_indices]
            output_queries_raw = np.asarray(d['output_queries'], dtype=np.float32)
            targets_raw = np.nan_to_num(np.asarray(d['targets'], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        in_idx = sample_indices(input_geom_raw.shape[0], self.max_input_particles, CFG.get('seed', SEED) + frame_id)
        out_idx = sample_indices(output_queries_raw.shape[0], self.max_output_points, CFG.get('seed', SEED) + 100000 + frame_id)
        x = (features_raw[in_idx] - input_mean[None, :]) / input_std[None, :]
        y = (targets_raw[out_idx] - target_mean[None, :]) / target_std[None, :]
        return {
            'input_geom': torch.from_numpy(normalize_xyz(input_geom_raw[in_idx])),
            'x': torch.from_numpy(np.clip(x, -8.0, 8.0).astype(np.float32)),
            'output_queries': torch.from_numpy(normalize_xyz(output_queries_raw[out_idx])),
            'y': torch.from_numpy(y.astype(np.float32)),
            'frame_id': torch.tensor(frame_id, dtype=torch.long),
        }

def make_loader(frame_ids, split_name, shuffle=False, max_output_points=None):
    ds = Task2GINODataset(
        frame_ids,
        split_name,
        max_input_particles=CFG.get('maximum_input_particles', None),
        max_output_points=max_output_points,
    )
    return ds, DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0)

train_ds, train_loader = make_loader(train_frame_ids, 'train', shuffle=False, max_output_points=CFG.get('maximum_eval_output_points', 32768))
val_id_ds, val_id_loader = make_loader(val_id_frame_ids, 'val_id', shuffle=False, max_output_points=CFG.get('maximum_eval_output_points', 32768))
val_angle_ds, val_angle_loader = make_loader(val_angle_frame_ids, 'val_angle', shuffle=False, max_output_points=CFG.get('maximum_eval_output_points', 32768))
test_ds, test_loader = make_loader(testing_frame_ids, 'test', shuffle=False, max_output_points=CFG.get('maximum_eval_output_points', 32768))


# In[ ]:


def make_latent_queries(res, device):
    line = torch.linspace(0.0, 1.0, int(res), dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(line, line, line, indexing='ij')
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)

LATENT_QUERIES = make_latent_queries(CFG['latent_res'], DEVICE)


def build_model(cfg):
    kwargs = dict(
        in_channels=len(feature_names),
        out_channels=len(target_names),
        gno_coord_dim=3,
        in_gno_radius=cfg['in_gno_radius'],
        out_gno_radius=cfg['out_gno_radius'],
        in_gno_transform_type=cfg['in_gno_transform_type'],
        out_gno_transform_type=cfg['out_gno_transform_type'],
        gno_embed_channels=cfg['gno_embed_channels'],
        gno_use_open3d=cfg.get('gno_use_open3d', False),
        gno_use_torch_scatter=cfg.get('gno_use_torch_scatter', False),
        fno_n_modes=tuple(cfg['fno_n_modes']),
        fno_hidden_channels=cfg['fno_hidden_channels'],
        fno_n_layers=cfg['fno_n_layers'],
        projection_channel_ratio=cfg['projection_channel_ratio'],
    )
    signature = inspect.signature(GINO)
    accepted = set(signature.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print('[info] Dropping GINO kwargs not accepted by this neuralop version:', dropped)
    return GINO(**filtered).to(DEVICE)

MODEL = build_model(CFG)
MODEL.load_state_dict(checkpoint['model_state_dict'])
MODEL.eval()
TOTAL_PARAMS = count_model_params(MODEL)
print('Params:', f'{TOTAL_PARAMS:,}')


# In[ ]:


target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
target_std_t = torch.tensor(target_std, dtype=torch.float32, device=DEVICE).view(1, 1, -1)


def move_batch(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model, batch):
    return model(
        input_geom=batch['input_geom'],
        latent_queries=LATENT_QUERIES,
        output_queries=batch['output_queries'],
        x=batch['x'],
    )


def denorm(y):
    return y * target_std_t + target_mean_t


def relative_l2(pred, target, eps=1e-12):
    diff = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    return torch.linalg.norm(diff, dim=1) / torch.linalg.norm(ref, dim=1).clamp_min(eps)


@torch.no_grad()
def evaluate(loader, collect=False):
    if len(loader.dataset) == 0:
        return {'rel_l2_phys': math.nan, 'rmse_phys': math.nan, 'mae_phys': math.nan, 'rel_per_sample': np.asarray([], dtype=np.float32)}
    rels, rmses, maes = [], [], []
    # Add tqdm here
    for batch in tqdm(loader, desc=f"Evaluating {loader.dataset.split_name}"):
        batch = move_batch(batch)
        pred = denorm(predict(MODEL, batch))
        y = denorm(batch['y'])
        rels.extend(relative_l2(pred, y).detach().cpu().numpy().tolist())
        rmses.append(float(torch.sqrt(torch.mean((pred - y) ** 2)).item()))
        maes.append(float(torch.mean(torch.abs(pred - y)).item()))
    out = {'rel_l2_phys': float(np.mean(rels)), 'rmse_phys': float(np.mean(rmses)), 'mae_phys': float(np.mean(maes))}
    if collect:
        out['rel_per_sample'] = np.asarray(rels, dtype=np.float32)
    return out

FINAL_METRICS = {
    'train_sampled': evaluate(train_loader, collect=True),
    'val_id': evaluate(val_id_loader, collect=True),
    'val_angle': evaluate(val_angle_loader, collect=True),
    'test': evaluate(test_loader, collect=True),
}
for split, m in FINAL_METRICS.items():
    print(f"{split:13s} rel-L2={m['rel_l2_phys']:.4e} RMSE={m['rmse_phys']:.4e} MAE={m['mae_phys']:.4e}")


# In[ ]:


# Parameters + convergence curves.
module_param_counts = defaultdict(int)
for name, p in MODEL.named_parameters():
    module_param_counts[name.split('.')[0]] += p.numel()
mods = sorted(module_param_counts, key=lambda k: module_param_counts[k], reverse=True)
vals = [module_param_counts[m] for m in mods]

fig_p, ax_p = plt.subplots(figsize=(9, 4))
ax_p.bar(mods, vals)
ax_p.set_title(f'Parameter distribution (total={TOTAL_PARAMS:,})')
ax_p.set_ylabel('parameters')
ax_p.tick_params(axis='x', rotation=45)
fig_p.tight_layout()
save_fig(fig_p, 'params_by_module.png')
plt.show()

if history and len(history.get('epoch', [])) > 0:
    fig_c, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].plot(history['epoch'], history['train_loss'], label='train normalized loss')
    axs[0].set_yscale('log'); axs[0].set_xlabel('epoch'); axs[0].set_title('Training loss'); axs[0].legend()
    axs[1].plot(history['epoch'], history['val_id_rel_l2_phys'], label='val id')
    axs[1].plot(history['epoch'], history['val_angle_rel_l2_phys'], label='val angle')
    axs[1].plot(history['epoch'], history['test_rel_l2_phys'], label='test')
    axs[1].set_yscale('log'); axs[1].set_xlabel('epoch'); axs[1].set_title('Physical relative error'); axs[1].legend()
    axs[2].plot(history['epoch'], history['lr'], label='lr')
    axs[2].set_xlabel('epoch'); axs[2].set_title('Learning rate'); axs[2].legend()
    fig_c.tight_layout()
    save_fig(fig_c, 'convergence_curves.png')
    plt.show()
else:
    print('[info] No training history found in checkpoint.')


# In[ ]:


# Error distribution and parity plot.
rel_vals = FINAL_METRICS['test']['rel_per_sample']
if rel_vals.size == 0:
    rel_vals = FINAL_METRICS['val_angle']['rel_per_sample']
plot_loader = test_loader if len(test_loader.dataset) else val_angle_loader
sample = move_batch(next(iter(plot_loader)))
with torch.no_grad():
    pred = denorm(predict(MODEL, sample))[0].detach().cpu().numpy()
true = denorm(sample['y'])[0].detach().cpu().numpy()
true_speed = np.linalg.norm(true[:, :3], axis=1)
pred_speed = np.linalg.norm(pred[:, :3], axis=1)
idx = sample_indices(true_speed.size, 12000, SEED + 99)

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
axs[0].hist(rel_vals, bins=30, color='tab:blue', alpha=0.85)
axs[0].set_title('Split rel-L2 histogram'); axs[0].set_xlabel('rel-L2'); axs[0].set_ylabel('count')
if rel_vals.size:
    sorted_rel = np.sort(rel_vals); cdf = np.linspace(0, 1, len(sorted_rel), endpoint=True)
    axs[1].plot(sorted_rel, cdf, color='tab:green')
axs[1].set_title('rel-L2 CDF'); axs[1].set_xlabel('rel-L2'); axs[1].set_ylabel('cdf'); axs[1].grid(True, alpha=0.3)
vmin = min(float(true_speed[idx].min()), float(pred_speed[idx].min()))
vmax = max(float(true_speed[idx].max()), float(pred_speed[idx].max()))
axs[2].scatter(true_speed[idx], pred_speed[idx], s=5, alpha=0.35)
axs[2].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.2)
axs[2].set_title('Parity: velocity magnitude'); axs[2].set_xlabel('true |U|'); axs[2].set_ylabel('pred |U|')
fig.tight_layout()
save_fig(fig, 'error_distribution_and_parity.png')
plt.show()


# In[ ]:


# Qualitative mid-plane slice: true, prediction, error for |U| and |W|.
def load_full_prediction(frame_id, max_output_points=None):
    ds = Task2GINODataset([frame_id], 'plot', max_input_particles=CFG.get('maximum_input_particles', None), max_output_points=max_output_points)
    batch = move_batch(next(iter(DataLoader(ds, batch_size=1, shuffle=False))))
    with torch.no_grad():
        pred = denorm(predict(MODEL, batch))[0].detach().cpu().numpy()
        true = denorm(batch['y'])[0].detach().cpu().numpy()
    pts = batch['output_queries'][0].detach().cpu().numpy() * coord_span[None, :] + coord_min[None, :]
    return pts, true, pred


def midplane(points, values, axis=2, band_frac=0.025):
    coord = points[:, axis]
    mid = 0.5 * (float(coord.min()) + float(coord.max()))
    band = max((float(coord.max()) - float(coord.min())) * band_frac, 1e-8)
    mask = np.abs(coord - mid) <= band
    if mask.sum() == 0:
        idx = np.argsort(np.abs(coord - mid))[:max(1, len(coord)//64)]
        mask = np.zeros(len(coord), dtype=bool); mask[idx] = True
    return points[mask], values[mask]

plot_ids = testing_frame_ids if len(testing_frame_ids) else val_angle_frame_ids
frame_id = int(plot_ids[len(plot_ids)//2])
pts, true, pred = load_full_prediction(frame_id, max_output_points=CFG.get('maximum_eval_output_points', 32768))
true_u = np.linalg.norm(true[:, :3], axis=1); pred_u = np.linalg.norm(pred[:, :3], axis=1); err_u = np.abs(pred_u - true_u)
true_w = np.linalg.norm(true[:, 3:], axis=1); pred_w = np.linalg.norm(pred[:, 3:], axis=1); err_w = np.abs(pred_w - true_w)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for row, (name, fields, cmap) in enumerate([('|U|', [true_u, pred_u, err_u], 'turbo'), ('|W|', [true_w, pred_w, err_w], 'magma')]):
    for col, (label, vals) in enumerate(zip(['true', 'pred', 'abs error'], fields)):
        p2, v2 = midplane(pts, vals)
        sc = axs[row, col].scatter(p2[:, 0], p2[:, 1], c=v2, s=6, cmap=cmap)
        axs[row, col].set_title(f'{name} {label}')
        axs[row, col].set_aspect('equal')
        plt.colorbar(sc, ax=axs[row, col], fraction=0.046, pad=0.04)
fig.suptitle(f'Mid-plane qualitative frame_id={frame_id}', y=0.995)
fig.tight_layout()
save_fig(fig, 'qualitative_midplane_velocity_vorticity.png')
plt.show()


# In[ ]:


# Timeline panel like Task1 analysis.
def plot_timeline(frame_ids, name, n_times=4):
    if len(frame_ids) == 0:
        print(f'[info] {name}: empty split')
        return
    frame_ids = np.asarray(sorted(frame_ids, key=lambda i: int((frame_contexts[int(i)] if isinstance(frame_contexts[int(i)], dict) else dict(frame_contexts[int(i)].item()))['frame'])), dtype=np.int64)
    idxs = np.linspace(0, len(frame_ids) - 1, min(n_times, len(frame_ids)), dtype=int)
    chosen = [int(frame_ids[i]) for i in idxs]
    fig, axs = plt.subplots(3, len(chosen), figsize=(4 * len(chosen), 9))
    if len(chosen) == 1:
        axs = np.asarray(axs).reshape(3, 1)
    for j, fid in enumerate(chosen):
        pts, true, pred = load_full_prediction(fid, max_output_points=CFG.get('maximum_eval_output_points', 32768))
        ts = np.linalg.norm(true[:, :3], axis=1); ps = np.linalg.norm(pred[:, :3], axis=1); es = np.abs(ps - ts)
        for row, vals in enumerate([ts, ps, es]):
            p2, v2 = midplane(pts, vals)
            sc = axs[row, j].scatter(p2[:, 0], p2[:, 1], c=v2, s=5, cmap='turbo' if row < 2 else 'magma')
            axs[row, j].set_aspect('equal')
            axs[row, j].set_title(['true |U|', 'pred |U|', 'abs error'][row])
            plt.colorbar(sc, ax=axs[row, j], fraction=0.046, pad=0.04)
    fig.suptitle(f'{name}: chronological snapshots', y=0.995)
    fig.tight_layout()
    save_fig(fig, f'timeline_{name.lower()}.png')
    plt.show()

plot_timeline(testing_frame_ids if len(testing_frame_ids) else val_angle_frame_ids, 'TEST')


# In[ ]:


# Data profile: feature distributions and frame-level split summary.
def context_for_id(i):
    c = frame_contexts[int(i)]
    return c if isinstance(c, dict) else dict(c.item())

rows = []
for split_name, ids in [('train', train_frame_ids), ('val_id', val_id_frame_ids), ('val_angle', val_angle_frame_ids), ('test', testing_frame_ids)]:
    for fid in ids[: min(len(ids), 300)]:
        c = context_for_id(fid)
        rows.append((split_name, float(c.get('aoa_deg', np.nan)), float(c.get('phase', np.nan)), int(c.get('n_particles', 0))))

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for split_name in sorted(set(r[0] for r in rows)):
    arr = np.asarray([[r[1], r[2], r[3]] for r in rows if r[0] == split_name], dtype=np.float64)
    axs[0].hist(arr[:, 0], bins=16, alpha=0.45, label=split_name)
    axs[1].hist(arr[:, 1], bins=16, alpha=0.45, label=split_name)
    axs[2].hist(arr[:, 2], bins=16, alpha=0.45, label=split_name)
axs[0].set_title('AoA distribution'); axs[1].set_title('Phase distribution'); axs[2].set_title('Particle count')
for ax in axs: ax.legend(); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, 'data_profile_split_distributions.png')
plt.show()

# 3D pair profile for one sample.
with np.load(sample_paths[int(train_frame_ids[0])], allow_pickle=True) as d:
    in_xyz = np.asarray(d['input_geom'], dtype=np.float32)
    out_xyz = np.asarray(d['output_queries'], dtype=np.float32)
    y = np.nan_to_num(np.asarray(d['targets'], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
idx_in = sample_indices(len(in_xyz), 7000, SEED)
idx_out = sample_indices(len(out_xyz), 7000, SEED + 1)
fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(in_xyz[idx_in, 0], in_xyz[idx_in, 1], in_xyz[idx_in, 2], c=in_xyz[idx_in, 2], s=3, cmap='viridis', alpha=0.65)
ax1.set_title('Input particles in 3D'); plt.colorbar(sc1, ax=ax1, fraction=0.04, pad=0.03)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(out_xyz[idx_out, 0], out_xyz[idx_out, 1], out_xyz[idx_out, 2], c=np.linalg.norm(y[idx_out, :3], axis=1), s=3, cmap='turbo', alpha=0.65)
ax2.set_title('Output Eulerian |U| in 3D'); plt.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.03)
fig.tight_layout()
save_fig(fig, 'data_profile_3d_pairs.png')
plt.show()


# In[ ]:


# Optional quick diagnostics: data-hunger curve and hyperparameter sweeps.
# Keep these small; they are meant to reveal gross sensitivity, not replace full training.
def train_eval_subset(train_ids_subset, quick_cfg, quick_epochs=2):
    if len(train_ids_subset) == 0 or len(test_loader.dataset) == 0:
        return math.nan
    ds = Task2GINODataset(train_ids_subset, 'quick_train', max_input_particles=quick_cfg.get('maximum_input_particles', 4000), max_output_points=min(quick_cfg.get('maximum_train_output_points', 4096), 4096))
    ld = DataLoader(ds, batch_size=1, shuffle=True)
    m = build_model(quick_cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=quick_cfg['lr'], weight_decay=quick_cfg['weight_decay'])
    for _ in range(quick_epochs):
        m.train()
        for batch in ld:
            batch = move_batch(batch)
            opt.zero_grad(set_to_none=True)
            pred = m(input_geom=batch['input_geom'], latent_queries=LATENT_QUERIES, output_queries=batch['output_queries'], x=batch['x'])
            loss = torch.mean((pred - batch['y']) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), quick_cfg.get('grad_clip_norm', 1.0))
            opt.step()
    old_model = globals()['MODEL']
    globals()['MODEL'] = m.eval()
    rel = evaluate(test_loader)['rel_l2_phys']
    globals()['MODEL'] = old_model
    return rel

RUN_OPTIONAL_QUICK_STUDIES = False
if RUN_OPTIONAL_QUICK_STUDIES:
    rng = np.random.default_rng(SEED)
    curve = []
    for frac in [0.2, 0.5, 1.0]:
        n = max(1, int(frac * len(train_frame_ids)))
        ids = np.sort(rng.choice(train_frame_ids, size=n, replace=False))
        rel = train_eval_subset(ids, dict(CFG), quick_epochs=2)
        curve.append((n, rel))
        print('data hunger', n, rel)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c[0] for c in curve], [c[1] for c in curve], marker='o')
    ax.set_xlabel('training samples'); ax.set_ylabel('test rel-L2'); ax.set_title('Data-hunger quick curve')
    ax.grid(True, alpha=0.3); fig.tight_layout(); save_fig(fig, 'data_hunger_curve.png'); plt.show()
else:
    print('Optional quick studies are disabled. Set RUN_OPTIONAL_QUICK_STUDIES=True to run data hunger/sweeps.')

