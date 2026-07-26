# Particle Evolution GNO (corrected)

import os, json, math, random, inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from neuralop.layers.gno_block import GNOBlock   # must be available

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
# Helpers
# -----------------------------------------------------------------------------
def env_int(name, default):
    return int(os.environ.get(name, default))

def env_float(name, default):
    return float(os.environ.get(name, default))

def env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in {'1','true','yes','y','on'}

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
            return npz[key]   # <-- no .astype, no np.asarray(..., dtype=...)
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
    # x: data coords, y: query coords. This installed GNOBlock uses forward(y=data_coords, x=query_coords, f_y=data_values)
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
    'run_tag': os.environ.get('PARTICLE_GNO_RUN_TAG', 'particle_evolution_gno'),
    'dataset_candidates': env_list('PARTICLE_EVOLUTION_DATASETS', [
        'processed_data/particle_evolution_dataset.npz'
    ]),
    'result_dir': os.environ.get('PARTICLE_GNO_RESULT_DIR', 'GINO_GNO/result/particle_evolution_gno'),
    # Input channels: include coordinates and Gamma because the GNO needs them in the feature vector
    'input_channels': env_list('EVOLUTION_SR_INPUT_CHANNELS', ['x','y','z','Gamma_x','Gamma_y','Gamma_z','sigma','geom_dist']),
    'predict_delta_u': env_bool('PARTICLE_GNO_PREDICT_DELTA_U', False),
    'hidden_channels': env_int('PARTICLE_GNO_HIDDEN', 128),
    'num_gno_layers': env_int('PARTICLE_GNO_LAYERS', 4),
    'gno_radius': env_float('PARTICLE_GNO_RADIUS', 0.30),
    'batch_size': env_int('PARTICLE_GNO_BATCH_SIZE', 1),
    'epochs': env_int('PARTICLE_GNO_EPOCHS', 120),
    'lr': env_float('PARTICLE_GNO_LR', 3e-4),
    'weight_decay': env_float('PARTICLE_GNO_WEIGHT_DECAY', 1e-4),
    'warmup_epochs': env_int('PARTICLE_GNO_WARMUP_EPOCHS', 5),
    'plateau_patience': env_int('PARTICLE_GNO_PLATEAU_PATIENCE', 8),
    'plateau_factor': env_float('PARTICLE_GNO_PLATEAU_FACTOR', 0.5),
    'grad_clip_norm': env_float('PARTICLE_GNO_GRAD_CLIP', 1.0),
    'num_workers': env_int('PARTICLE_GNO_NUM_WORKERS', 0),
    'max_input_particles': env_int('PARTICLE_GNO_MAX_PARTICLES', 0),
    'rollout_steps_max': env_int('PARTICLE_GNO_ROLLOUT_STEPS_MAX', 1),
    'rollout_weight_max': env_float('PARTICLE_GNO_ROLLOUT_WEIGHT_MAX', 0.2),
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
target_names_all = as_list(npz['target_names']) if 'target_names' in npz else ['dx','dy','dz','dGamma_x','dGamma_y','dGamma_z','dsigma']
coord_min = stat(npz, 'coord_min', 3, 0.0)
coord_span = stat(npz, 'coord_span', 3, 1.0)
# AFTER (added .reshape(-1))
in_mean = stat(npz, 'in_mean', inputs_t.shape[1], 0.0).reshape(-1)
in_std = np.maximum(stat(npz, 'in_std', inputs_t.shape[1], 1.0).reshape(-1), 1e-8)
raw_target_mean = stat(npz, 'out_mean', len(target_names_all), 0.0).reshape(-1)
raw_target_std = np.maximum(stat(npz, 'out_std', len(target_names_all), 1.0).reshape(-1), 1e-8)

target_key = 'targets_delta_norm_all' if 'targets_delta_norm_all' in npz else 'targets_delta_norm'
targets_delta = np.asarray(npz[target_key], dtype=np.float32)
if targets_delta.shape[1] > len(target_names_all):
    target_names_all += [f'target_{i}' for i in range(len(target_names_all), targets_delta.shape[1])]

out_channels = 10 if CFG['predict_delta_u'] else 7
target_names = target_names_all[:out_channels]
target_mean = raw_target_mean[:out_channels]
target_std = raw_target_std[:out_channels]

feature_to_idx = {name: i for i, name in enumerate(feature_names)}
missing = [name for name in CFG['input_channels'] if name not in feature_to_idx]
if missing:
    raise KeyError(f'Missing input feature channels: {missing}. Available: {feature_names}')
active_feature_idx = np.asarray([feature_to_idx[name] for name in CFG['input_channels']], dtype=np.int64)

# Coordinates are now part of the features; we still need a separate geometry for the GNO?
# The GNO block in this notebook expects the coordinates as a separate argument `input_geom`,
# so we still extract them from the features.
coord_idx = np.asarray([feature_to_idx.get(k, -1) for k in ('x','y','z')], dtype=np.int64)
if (coord_idx < 0).any():
    # fallback to first three columns
    coord_idx = np.asarray([0,1,2], dtype=np.int64)
state_idx = np.asarray([feature_to_idx.get(k, -1) for k in ('x','y','z','Gamma_x','Gamma_y','Gamma_z','sigma')], dtype=np.int64)
if (state_idx < 0).any():
    raise KeyError('Need x,y,z,Gamma_x,Gamma_y,Gamma_z,sigma in inputs_t for rollout/state tracking.')
velocity_idx = np.asarray([feature_to_idx.get(k, -1) for k in ('u_x','u_y','u_z')], dtype=np.int64)

# Splits – use the exact keys from the dataset
split_keys = ('train_pair_ids','val_pair_ids','test_pair_ids')
if all(k in npz for k in split_keys):
    split_indices = {'train': np.asarray(npz['train_pair_ids'], dtype=np.int64),
                     'val':   np.asarray(npz['val_pair_ids'], dtype=np.int64),
                     'test':  np.asarray(npz['test_pair_ids'], dtype=np.int64)}
else:
    # fallback random split (not recommended, but kept for safety)
    n = len(pair_ranges)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train, n_val = int(0.8*n), int(0.1*n)
    split_indices = {'train': perm[:n_train], 'val': perm[n_train:n_train+n_val], 'test': perm[n_train+n_val:]}

rollout_cases = as_list(npz['rollout_cases']) if 'rollout_cases' in npz else []
rollout_true_states = np.asarray(npz['rollout_true_states'], dtype=np.float32) if 'rollout_true_states' in npz else None
case_to_rollout = {case: i for i, case in enumerate(rollout_cases)}
case_to_nmin = {case: int(rollout_true_states[i].shape[1]) for case, i in case_to_rollout.items()} if rollout_true_states is not None else {}
case_to_frame_index = {}
# Build frame index from rollout (or from contexts)
if 'rollout_frame_ids' in npz:
    for case, frames in zip(rollout_cases, npz['rollout_frame_ids'].tolist()):
        case_to_frame_index[case] = {str(f).zfill(6): i for i, f in enumerate(frames)}
else:
    for pid, ctx in enumerate(contexts):
        case = str(ctx.get('case', ''))
        frame = str(ctx.get('frame_t', ctx.get('frame_id', ''))).zfill(6)
        case_to_frame_index.setdefault(case, {})[frame] = len(case_to_frame_index.setdefault(case, {}))

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ParticleEvolutionDataset(Dataset):
    def __init__(self, pair_ids):
        self.pair_ids = np.asarray(pair_ids, dtype=np.int64)
    def __len__(self):
        return len(self.pair_ids)
    def __getitem__(self, i):
        pair_id = int(self.pair_ids[i])
        start = int(pair_ranges[pair_id][3])
        end   = int(pair_ranges[pair_id][4])
        n = end - start
        ctx = contexts[pair_id]
        case = str(ctx.get('case', ''))
        n_use = min(case_to_nmin.get(case, n), n)
        if CFG['max_input_particles'] > 0:
            n_use = min(n_use, int(CFG['max_input_particles']))
        idx = slice(start, start + n_use)
        rows = inputs_t[idx]
        coords = normalize_xyz(rows[:, coord_idx], coord_min, coord_span)
        x = ((rows[:, active_feature_idx] - in_mean[active_feature_idx]) / in_std[active_feature_idx]).astype(np.float32)
        return {
            'pair_id': torch.tensor(pair_id, dtype=torch.long),
            'input_geom': torch.from_numpy(coords),   # separate geometry for the GNO blocks
            'x': torch.from_numpy(x),
            'state_phys': torch.from_numpy(rows[:, state_idx].astype(np.float32)),
            'velocity_phys': torch.from_numpy(rows[:, velocity_idx].astype(np.float32)) if (velocity_idx >= 0).all() else torch.zeros((n_use,3)),
            'global_params': torch.from_numpy(context_global_params(ctx)),
            'y_delta': torch.from_numpy(targets_delta[idx, :out_channels].astype(np.float32)),
        }

def collate_one(batch):
    assert len(batch) == 1, 'Use batch_size=1 for variable-size particle sets.'
    b = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) and k not in {'pair_id'} else v) for k, v in b.items()}

loaders = {}
for split, ids in split_indices.items():
    loaders[split] = DataLoader(
        ParticleEvolutionDataset(ids),
        batch_size=CFG['batch_size'],
        shuffle=(split=='train'),
        num_workers=CFG['num_workers'],
        collate_fn=collate_one,
        pin_memory=torch.cuda.is_available()
    )
print('features:', CFG['input_channels'])
print('targets:', target_names)
print({k: len(v.dataset) for k, v in loaders.items()})

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class ParticleEvolutionGNO(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=128, layers=4, radius=0.30, global_dim=3):
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.global_condition_mlp = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([make_gnoblock(hidden, hidden, radius) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.particle_skip_proj = nn.Linear(hidden, hidden)
        self.delta_fusion = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.delta_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_channels))

    def forward(self, input_geom, x, global_params):
        if input_geom.ndim == 2:
            input_geom = input_geom.unsqueeze(0); x = x.unsqueeze(0); global_params = global_params.unsqueeze(0)
        h0 = self.lift(x)
        h = h0 + self.global_condition_mlp(global_params).unsqueeze(1)
        outs = []
        for b in range(h.shape[0]):
            hb = h[b]
            coords = input_geom[b]
            for block, norm in zip(self.blocks, self.norms):
                upd = call_gno(block, x=coords, y=coords, f_y=hb)
                if upd.ndim == 3:
                    upd = upd.squeeze(0)
                hb = norm(hb + upd)
            outs.append(hb)
        h = torch.stack(outs, dim=0)
        h = self.delta_fusion(torch.cat([h, self.particle_skip_proj(h0)], dim=-1))
        return self.delta_head(h)

model = ParticleEvolutionGNO(
    len(active_feature_idx), out_channels,
    CFG['hidden_channels'], CFG['num_gno_layers'], CFG['gno_radius']
).to(DEVICE)
print(model.__class__.__name__, f'{sum(p.numel() for p in model.parameters()):,} parameters')

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=CFG['plateau_factor'], patience=CFG['plateau_patience']
)

target_mean_t = torch.tensor(target_mean, device=DEVICE).view(1,1,-1)
target_std_t = torch.tensor(target_std, device=DEVICE).view(1,1,-1)
coord_min_t = torch.tensor(coord_min, device=DEVICE).view(1,1,3)
coord_span_t = torch.tensor(coord_span, device=DEVICE).view(1,1,3)

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def denormalize_delta(y):
    return y * target_std_t + target_mean_t

def make_features_from_state(old_x_norm, state_phys, velocity_phys):
    """
    Build a new normalized feature tensor from the updated state and velocity.
    NOTE: geometry features (geom_dist, etc.) are carried over from the original
    snapshot because recomputing them requires the wing surface VTK file.
    For short rollouts this is acceptable; for longer rollouts a live
    geometry recomputation should be added.
    """
    vals = []
    for j, name in enumerate(CFG['input_channels']):
        if name == 'x': raw = state_phys[...,0]
        elif name == 'y': raw = state_phys[...,1]
        elif name == 'z': raw = state_phys[...,2]
        elif name == 'Gamma_x': raw = state_phys[...,3]
        elif name == 'Gamma_y': raw = state_phys[...,4]
        elif name == 'Gamma_z': raw = state_phys[...,5]
        elif name == 'sigma': raw = state_phys[...,6]
        elif name == 'u_x': raw = velocity_phys[...,0]
        elif name == 'u_y': raw = velocity_phys[...,1]
        elif name == 'u_z': raw = velocity_phys[...,2]
        else:
            vals.append(old_x_norm[...,j])
            continue
        raw_mean = torch.tensor(in_mean[active_feature_idx[j]], device=DEVICE)
        raw_std = torch.tensor(in_std[active_feature_idx[j]], device=DEVICE).clamp_min(1e-8)
        vals.append((raw - raw_mean) / raw_std)
    return torch.stack(vals, dim=-1)

def rollout_targets_for_batch(batch, steps):
    if rollout_true_states is None or steps <= 1:
        return None
    pair_id = int(batch['pair_id'].reshape(-1)[0].item())
    ctx = contexts[pair_id]
    case = str(ctx.get('case', ''))
    rix = case_to_rollout.get(case)
    if rix is None:
        return None
    frame = str(ctx.get('frame_t', ctx.get('frame_id', ''))).zfill(6)
    start = case_to_frame_index.get(case, {}).get(frame)
    if start is None:
        return None
    seq = np.asarray(rollout_true_states[rix], dtype=np.float32)
    max_steps = min(int(steps), seq.shape[0] - start - 1)
    if max_steps <= 0:
        return None
    n = batch['state_phys'].shape[1]
    fut = seq[start+1:start+1+max_steps, :n, :7]
    return torch.from_numpy(fut).to(DEVICE).unsqueeze(1)

def rollout_schedule(epoch):
    max_steps = int(CFG['rollout_steps_max'])
    if max_steps <= 1:
        return 1, 0.0
    frac = min(max(epoch / max(CFG['epochs'], 1), 0.0), 1.0)
    steps = 1 + int(round(frac * (max_steps - 1)))
    weight = frac * float(CFG['rollout_weight_max'])
    return max(2, steps), weight

def rollout_loss(batch, steps):
    targets = rollout_targets_for_batch(batch, steps)
    if targets is None:
        return torch.zeros((), device=DEVICE)
    state = batch['state_phys'].clone()
    vel = batch['velocity_phys'].clone()
    xnorm = batch['x'].clone()
    total = torch.zeros((), device=DEVICE)
    weights = []
    for k in range(targets.shape[0]):
        geom = torch.clamp((state[...,:3] - coord_min_t) / coord_span_t.clamp_min(1e-8), 0.0, 1.0)
        pred_norm = model(geom, xnorm, batch['global_params'])
        pred_phys = denormalize_delta(pred_norm)
        next_state = state.clone()
        next_state[..., :7] = state[..., :7] + pred_phys[..., :7]
        if pred_phys.shape[-1] >= 10:
            vel = vel + pred_phys[..., 7:10]
        w = 1.0 + k / max(targets.shape[0] - 1, 1)
        total = total + w * F.mse_loss(next_state[..., :7], torch.nan_to_num(targets[k]))
        weights.append(w)
        state = next_state
        xnorm = make_features_from_state(xnorm, state, vel)
    return total / max(sum(weights), 1e-8)

def train_epoch(epoch):
    model.train()
    meters = {'loss':0.0, 'delta_loss':0.0, 'rollout_loss':0.0, 'n':0}
    steps, rw = rollout_schedule(epoch)
    for batch in loaders['train']:
        batch = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch['input_geom'], batch['x'], batch['global_params'])
        delta_loss = F.mse_loss(pred, torch.nan_to_num(batch['y_delta']))
        rloss = rollout_loss(batch, steps) if rw > 0 else torch.zeros((), device=DEVICE)
        loss = delta_loss + rw * rloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip_norm'])
        optimizer.step()
        meters['loss'] += float(loss.detach()); meters['delta_loss'] += float(delta_loss.detach()); meters['rollout_loss'] += float(rloss.detach()); meters['n'] += 1
    return {k: v/max(meters['n'],1) for k,v in meters.items() if k!='n'} | {'rollout_steps': steps, 'rollout_weight': rw}

@torch.no_grad()
def evaluate(split='val'):
    model.eval()
    meters = {'loss':0.0, 'delta_loss':0.0, 'rel_l2':0.0, 'rollout_loss':0.0, 'n':0}
    steps = max(2, int(CFG['rollout_steps_max'])) if int(CFG['rollout_steps_max']) > 1 else 1
    for batch in loaders[split]:
        batch = to_device(batch)
        pred = model(batch['input_geom'], batch['x'], batch['global_params'])
        loss = F.mse_loss(pred, torch.nan_to_num(batch['y_delta']))
        meters['loss'] += float(loss); meters['delta_loss'] += float(loss)
        meters['rel_l2'] += float(rel_l2(denormalize_delta(pred), denormalize_delta(batch['y_delta'])))
        meters['rollout_loss'] += float(rollout_loss(batch, steps)) if steps > 1 else 0.0
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
            'target_names': target_names, 'coord_min': coord_min, 'coord_span': coord_span,
            'in_mean_active': in_mean[active_feature_idx], 'in_std_active': in_std[active_feature_idx],
            'target_mean': target_mean, 'target_std': target_std, 'saved_at_utc': datetime.utcnow().isoformat() + 'Z'
        }
        torch.save(ckpt, RESULT_DIR / f"{CFG['run_tag']}_best_model.pt")
        with open(RESULT_DIR / f"{CFG['run_tag']}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
print('best val:', best)
if len(loaders['test'].dataset):
    print('test:', json.dumps(evaluate('test'), indent=2))