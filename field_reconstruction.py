# Field Reconstruction GINO (corrected)

import os, json, math, random, inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from neuralop.layers.gno_block import GNOBlock
from neuralop.models import FNO            # no fallback, must be present

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', DEVICE)

# -----------------------------------------------------------------------------
# Helpers (identical to particle notebook)
# -----------------------------------------------------------------------------
def env_int(name, default):
    return int(os.environ.get(name, default))

def env_float(name, default):
    return float(os.environ.get(name, default))

def env_list(name, default):
    v = os.environ.get(name)
    if not v:
        return list(default)
    return [x.strip() for x in v.split(',') if x.strip()]

def find_dataset(candidates):
    for p in candidates:
        p = Path(p).expanduser()
        if p.exists():
            return p
    raise FileNotFoundError('None of these dataset paths exist: ' + ', '.join(map(str, candidates)))

def as_list(arr):
    if arr is None:
        return []
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    return [str(x) for x in arr]

def load_contexts(npz, n):
    if 'pair_contexts' in npz:
        raw = npz['pair_contexts']
    elif 'frame_contexts' in npz:
        raw = npz['frame_contexts']
    else:
        return [{} for _ in range(n)]
    out = []
    for item in raw.tolist():
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            try:
                out.append(json.loads(item))
            except Exception:
                out.append({'raw': item})
        else:
            out.append({})
    return out

def get_pair_ranges(npz):
    for key in ('pair_ranges', 'frame_ranges', 'ranges'):
        if key in npz:
            return npz[key]   # ← NO .astype, NO np.asarray with dtype
    raise KeyError('Dataset needs pair_ranges or frame_ranges.')

def stat(npz, key, size=None, fill=0.0):
    if key in npz:
        return np.asarray(npz[key], dtype=np.float32)
    if size is None:
        raise KeyError(key)
    return np.full((size,), fill, dtype=np.float32)

def normalize_xyz(xyz, coord_min, coord_span):
    return np.clip((xyz - coord_min) / np.maximum(coord_span, 1e-8), 0.0, 1.0).astype(np.float32)

def context_global_params(ctx):
    aoa = float(ctx.get('angle_of_attack', ctx.get('aoa', 0.0))) / 45.0
    uinf = float(ctx.get('freestream_magnitude', ctx.get('Uinf', ctx.get('u_inf', 10.0)))) / 10.0
    phase = float(ctx.get('phase', 0.0))
    return np.asarray([aoa, uinf, phase], dtype=np.float32)

def make_gnoblock(in_channels, out_channels, radius, reduction='mean'):
    sig = inspect.signature(GNOBlock)
    kwargs = {}
    for name in sig.parameters:
        if name in {'in_channels','in_chan','in_features'}:
            kwargs[name] = in_channels
        elif name in {'out_channels','out_chan','out_features'}:
            kwargs[name] = out_channels
        elif name in {'radius','radius_cutoff'}:
            kwargs[name] = radius
        elif name == 'coord_dim':
            kwargs[name] = 3
        elif name == 'reduction':
            kwargs[name] = reduction
        elif name == 'pos_embedding_type':
            kwargs[name] = 'transformer'
        elif name == 'channel_mlp_layers':
            kwargs[name] = [in_channels, out_channels, out_channels]
        elif name == 'use_open3d_neighbor_search':
            kwargs[name] = False
        elif name == 'use_torch_scatter_reduce':
            kwargs[name] = False
    try:
        return GNOBlock(**kwargs)
    except TypeError:
        return GNOBlock(in_channels=in_channels, out_channels=out_channels, coord_dim=3, radius=radius)

def call_gno(block, x, y, f_y):
    try:
        return block(y=x, x=y, f_y=f_y)
    except TypeError:
        return block(x=y, y=x, f_y=f_y)

def rel_l2(pred, true, eps=1e-12):
    pred = torch.nan_to_num(pred)
    true = torch.nan_to_num(true)
    num = torch.linalg.vector_norm((pred - true).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(true.reshape(true.shape[0], -1), dim=1).clamp_min(eps)
    return (num / den).mean()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CFG = {
    'run_tag': os.environ.get('FIELD_GINO_RUN_TAG', 'field_reconstruction_gino'),
    'dataset_candidates': env_list('FIELD_RECONSTRUCTION_DATASETS', [
        'processed_data/particle_evolution_dataset.npz',   # reuses the same dataset
        'process_data_evolution/particle_evolution_dataset.npz',
    ]),
    'result_dir': os.environ.get('FIELD_GINO_RESULT_DIR', 'GINO_GNO/result/field_reconstruction_gino'),
    'input_channels': env_list('EVOLUTION_SR_INPUT_CHANNELS', ['Gamma_x','Gamma_y','Gamma_z','sigma','u_x','u_y','u_z','geom_dist']),
    'hidden_channels': env_int('FIELD_GINO_HIDDEN', 128),
    'latent_res': env_int('FIELD_GINO_LATENT_RES', 24),
    'fno_layers': env_int('FIELD_GINO_FNO_LAYERS', 6),
    'fno_modes': env_int('FIELD_GINO_FNO_MODES', 6),
    'encoder_radius': env_float('FIELD_GINO_ENCODER_RADIUS', 0.07),
    'max_train_query_points': env_int('FIELD_GINO_MAX_QUERY_POINTS', 4096),
    'max_eval_query_points': env_int('FIELD_GINO_MAX_EVAL_QUERY_POINTS', 12000),
    'batch_size': env_int('FIELD_GINO_BATCH_SIZE', 1),
    'epochs': env_int('FIELD_GINO_EPOCHS', 120),
    'lr': env_float('FIELD_GINO_LR', 3e-4),
    'weight_decay': env_float('FIELD_GINO_WEIGHT_DECAY', 1e-4),
    'warmup_epochs': env_int('FIELD_GINO_WARMUP_EPOCHS', 5),
    'plateau_patience': env_int('FIELD_GINO_PLATEAU_PATIENCE', 8),
    'plateau_factor': env_float('FIELD_GINO_PLATEAU_FACTOR', 0.5),
    'grad_clip_norm': env_float('FIELD_GINO_GRAD_CLIP', 1.0),
    'num_workers': env_int('FIELD_GINO_NUM_WORKERS', 0),
}
RESULT_DIR = Path(CFG['result_dir'])
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = find_dataset(CFG['dataset_candidates'])
print('Dataset:', DATASET_PATH)
print(json.dumps(CFG, indent=2))

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
npz = np.load(DATASET_PATH, allow_pickle=True)
inputs_t = np.asarray(npz['inputs_t'], dtype=np.float32)
pair_ranges = get_pair_ranges(npz)
contexts = load_contexts(npz, len(pair_ranges))
feature_names = as_list(npz['feature_names'])
coord_min = stat(npz, 'coord_min', 3, 0.0)
coord_span = stat(npz, 'coord_span', 3, 1.0)
# AFTER (added .reshape(-1))
in_mean = stat(npz, 'in_mean', inputs_t.shape[1], 0.0).reshape(-1)
in_std = np.maximum(stat(npz, 'in_std', inputs_t.shape[1], 1.0).reshape(-1), 1e-8)


# Field targets
for qkey in ('query_coords', 'field_query_coords', 'output_queries'):
    if qkey in npz:
        query_coords_all = np.asarray(npz[qkey], dtype=np.float32)
        break
else:
    raise KeyError('Field dataset needs query_coords / field_query_coords / output_queries.')
for ykey in ('targets_velocity_field_norm', 'targets_field_norm', 'y_field_norm'):
    if ykey in npz:
        y_field_all = np.asarray(npz[ykey], dtype=np.float32)
        break
else:
    raise KeyError('Field dataset needs normalized field velocity targets.')
mask_all = np.asarray(npz['field_query_mask'], dtype=bool) if 'field_query_mask' in npz else np.isfinite(y_field_all).all(axis=-1)
field_mean = stat(npz, 'field_mean', y_field_all.shape[-1], 0.0).reshape(-1)
field_std = np.maximum(stat(npz, 'field_std', y_field_all.shape[-1], 1.0).reshape(-1), 1e-8)
field_target_names = as_list(npz['field_target_names']) if 'field_target_names' in npz else ['u_x','u_y','u_z']

# Input feature selection
feature_to_idx = {name: i for i, name in enumerate(feature_names)}
missing = [name for name in CFG['input_channels'] if name not in feature_to_idx]
if missing:
    raise KeyError(f'Missing input feature channels: {missing}. Available: {feature_names}')
active_feature_idx = np.asarray([feature_to_idx[name] for name in CFG['input_channels']], dtype=np.int64)

coord_idx = np.asarray([feature_to_idx.get(k, -1) for k in ('x','y','z')], dtype=np.int64)
if (coord_idx < 0).any():
    coord_idx = np.asarray([0,1,2], dtype=np.int64)

# Splits
split_keys = ('train_pair_ids','val_pair_ids','test_pair_ids')
if all(k in npz for k in split_keys):
    split_indices = {'train': np.asarray(npz['train_pair_ids'], dtype=np.int64),
                     'val':   np.asarray(npz['val_pair_ids'], dtype=np.int64),
                     'test':  np.asarray(npz['test_pair_ids'], dtype=np.int64)}
else:
    n = len(pair_ranges)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train, n_val = int(0.8*n), int(0.1*n)
    split_indices = {'train': perm[:n_train], 'val': perm[n_train:n_train+n_val], 'test': perm[n_train+n_val:]}

class FieldReconstructionDataset(Dataset):
    def __init__(self, pair_ids, split):
        self.pair_ids = np.asarray(pair_ids, dtype=np.int64)
        self.split = split
    def __len__(self):
        return len(self.pair_ids)
    def __getitem__(self, i):
        pair_id = int(self.pair_ids[i])
        start = int(pair_ranges[pair_id][3])
        end   = int(pair_ranges[pair_id][4])
        rows = inputs_t[start:end]
        input_geom = normalize_xyz(rows[:, coord_idx], coord_min, coord_span)
        x = ((rows[:, active_feature_idx] - in_mean[active_feature_idx]) / in_std[active_feature_idx]).astype(np.float32)
        mask = mask_all[pair_id].astype(bool)
        valid = np.flatnonzero(mask)
        # Cap number of query points to keep memory in check
        cap = CFG['max_train_query_points'] if self.split == 'train' else CFG['max_eval_query_points']
        if len(valid) > cap:
            rng = np.random.default_rng(SEED + pair_id)
            valid = rng.choice(valid, size=cap, replace=False)
        q = normalize_xyz(query_coords_all[pair_id, valid], coord_min, coord_span)
        y = y_field_all[pair_id, valid, :len(field_target_names)].astype(np.float32)
        return {
            'pair_id': torch.tensor(pair_id),
            'input_geom': torch.from_numpy(input_geom),
            'x': torch.from_numpy(x),
            'output_queries': torch.from_numpy(q),
            'y_field': torch.from_numpy(y),
            'global_params': torch.from_numpy(context_global_params(contexts[pair_id]))
        }

def collate_one(batch):
    assert len(batch) == 1
    b = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) and k not in {'pair_id'} else v) for k, v in b.items()}

loaders = {}
for split, ids in split_indices.items():
    loaders[split] = DataLoader(
        FieldReconstructionDataset(ids, split),
        batch_size=CFG['batch_size'],
        shuffle=(split=='train'),
        num_workers=CFG['num_workers'],
        collate_fn=collate_one,
        pin_memory=torch.cuda.is_available()
    )
print('features:', CFG['input_channels'])
print('targets:', field_target_names)
print({k: len(v.dataset) for k, v in loaders.items()})

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def positional_encoding(x, num_freqs=4):
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    xb = x.unsqueeze(-2) * freqs.view(1,1,-1,1)
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)

def make_latent_grid(res, device):
    lin = torch.linspace(0, 1, res, device=device)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

class FieldReconstructionGINO(nn.Module):
    def __init__(self, in_channels, out_channels=3, hidden=128, latent_res=24,
                 fno_layers=6, fno_modes=6, radius=0.15, global_dim=3):
        super().__init__()
        self.latent_res = latent_res
        self.hidden = hidden
        self.lift = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.encoder = make_gnoblock(hidden, hidden, radius)
        self.global_condition_mlp = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.fno = FNO(
            n_modes=(fno_modes, fno_modes, fno_modes),
            hidden_channels=hidden,
            in_channels=hidden,
            out_channels=hidden,
            n_layers=fno_layers
        )
        pe_dim = 3 + 2 * 4 * 3
        self.decoder = nn.Sequential(
            nn.Linear(hidden + pe_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_channels)
        )

    def sample_grid(self, grid, queries):
        q = queries.clamp(0,1) * 2 - 1
        q = q[..., [2,1,0]].view(q.shape[0], q.shape[1], 1, 1, 3)
        vals = F.grid_sample(grid, q, mode='bilinear', align_corners=True)
        return vals.squeeze(-1).squeeze(-1).transpose(1,2)

    def forward(self, input_geom, x, output_queries, global_params):
        if input_geom.ndim == 2:
            input_geom = input_geom.unsqueeze(0); x = x.unsqueeze(0); output_queries = output_queries.unsqueeze(0); global_params = global_params.unsqueeze(0)
        base_grid = make_latent_grid(self.latent_res, input_geom.device)
        h = self.lift(x)
        latent_batches = []
        for b in range(h.shape[0]):
            lat = call_gno(self.encoder, x=input_geom[b], y=base_grid, f_y=h[b])
            if lat.ndim == 3:
                lat = lat.squeeze(0)
            latent_batches.append(lat)
        latent = torch.stack(latent_batches, dim=0)
        latent = latent + self.global_condition_mlp(global_params).unsqueeze(1)
        grid = latent.transpose(1,2).reshape(latent.shape[0], self.hidden, self.latent_res, self.latent_res, self.latent_res)
        grid = self.fno(grid)
        qfeat = self.sample_grid(grid, output_queries)
        return self.decoder(torch.cat([qfeat, positional_encoding(output_queries)], dim=-1))

model = FieldReconstructionGINO(
    len(active_feature_idx), len(field_target_names),
    CFG['hidden_channels'], CFG['latent_res'], CFG['fno_layers'], CFG['fno_modes'], CFG['encoder_radius']
).to(DEVICE)
print(model.__class__.__name__, f'{sum(p.numel() for p in model.parameters()):,} parameters')

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=CFG['plateau_factor'], patience=CFG['plateau_patience']
)
field_mean_t = torch.tensor(field_mean[:len(field_target_names)], device=DEVICE).view(1,1,-1)
field_std_t = torch.tensor(field_std[:len(field_target_names)], device=DEVICE).view(1,1,-1)

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def denormalize_field(y):
    return y * field_std_t + field_mean_t

def train_epoch(epoch):
    model.train()
    meters = {'loss':0.0, 'n':0}
    for batch in loaders['train']:
        batch = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch['input_geom'], batch['x'], batch['output_queries'], batch['global_params'])
        loss = F.mse_loss(pred, torch.nan_to_num(batch['y_field']))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip_norm'])
        optimizer.step()
        meters['loss'] += float(loss.detach()); meters['n'] += 1
    return {'loss': meters['loss']/max(meters['n'],1)}

@torch.no_grad()
def evaluate(split='val'):
    model.eval()
    meters = {'loss':0.0, 'rel_l2':0.0, 'n':0}
    for batch in loaders[split]:
        batch = to_device(batch)
        pred = model(batch['input_geom'], batch['x'], batch['output_queries'], batch['global_params'])
        loss = F.mse_loss(pred, torch.nan_to_num(batch['y_field']))
        meters['loss'] += float(loss)
        meters['rel_l2'] += float(rel_l2(denormalize_field(pred), denormalize_field(batch['y_field'])))
        meters['n'] += 1
    return {k: v/max(meters['n'],1) for k,v in meters.items() if k!='n'}

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
history, best = [], float('inf')
for epoch in range(1, CFG['epochs'] + 1):
    if epoch <= CFG['warmup_epochs']:
        lr_scale = epoch / max(CFG['warmup_epochs'], 1)
        for g in optimizer.param_groups:
            g['lr'] = CFG['lr'] * lr_scale
    train = train_epoch(epoch)
    val = evaluate('val') if len(loaders['val'].dataset) else {'loss': train['loss']}
    if epoch > CFG['warmup_epochs']:
        plateau.step(val['loss'])
    rec = {'epoch': epoch, 'train': train, 'val': val, 'lr': optimizer.param_groups[0]['lr']}
    history.append(rec)
    print(json.dumps(rec, indent=2))
    if val['loss'] < best:
        best = val['loss']
        ckpt = {
            'model_state_dict': model.state_dict(), 'cfg': CFG, 'feature_names': CFG['input_channels'],
            'field_target_names': field_target_names, 'coord_min': coord_min, 'coord_span': coord_span,
            'in_mean_active': in_mean[active_feature_idx], 'in_std_active': in_std[active_feature_idx],
            'field_mean': field_mean, 'field_std': field_std, 'saved_at_utc': datetime.utcnow().isoformat() + 'Z'
        }
        torch.save(ckpt, RESULT_DIR / f"{CFG['run_tag']}_best_model.pt")
        with open(RESULT_DIR / f"{CFG['run_tag']}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
print('best val:', best)
if len(loaders['test'].dataset):
    print('test:', json.dumps(evaluate('test'), indent=2))