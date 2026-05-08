"""Simplified manual preprocessing (single file).

Goal:
- Merge split INPUT/OUTPUT H5 files into per-frame NPZ files.
- Build one field dataset NPZ for Task-2 (FNO: particle->velocity/vorticity field).
- Build one particle dataset NPZ for Task-1 (GNO: particle features->U, gradU).

Edit only the USER SETTINGS block and run:
    python3 final-2/preprocess_data.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import h5py

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ==========================
# USER SETTINGS
# ==========================
RAW_ROOT = Path('/media/dysco/New Volume/Neeraj/neuralop/data/raw data')
DATASET_IDS = ['1', '2', '7', '8', '9']

INPUT_H5_PATTERN = 'input/wing-example_pfield.*.h5'
OUTPUT_H5_PATTERN = 'output/wing-example_fdom.*.h5'
VTK_PATTERN = 'vtk/wing-example_Wing_vlm.*.vtk'  # optional sidecar for path tracking

OUT_ROOT = Path(__file__).resolve().parent / 'output'
MERGED_ROOT = OUT_ROOT / 'merged_frames'

GRID_RESOLUTION = (32, 32, 32)
OUTPUT_MODE = 'UW'  # 'U', 'W', or 'UW'
RANDOM_SEED = 42

FIELD_INPUT_CHANNELS = [
    'Gamma_x', 'Gamma_y', 'Gamma_z', 'sigma', 'density',
    'X', 'Y', 'Z',
]

PARTICLE_INPUT_FEATURES = [
    'x', 'y', 'z',
    'Gamma_x', 'Gamma_y', 'Gamma_z',
    'sigma', 'vol', 'circulation', 'static',
    'phase', 'angle_of_attack', 'freestream_x', 'freestream_y', 'freestream_z',
]

# ==========================
# Helpers
# ==========================
FRAME_RE = re.compile(r'(\d+)(?!.*\d)')


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def frame_id(path: Path) -> str:
    m = FRAME_RE.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


def as_xyz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f'Cannot parse xyz from shape={a.shape}')


def as_vec_field(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 4 and a.shape[-1] == 3:
        return a.astype(np.float64)
    if a.ndim == 4 and a.shape[0] == 3:
        return np.moveaxis(a, 0, -1).astype(np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f'Cannot parse vector field from shape={a.shape}')


def read_h5_selected(path: Path, key_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, 'r') as f:
        for out_key, in_key in key_map.items():
            if in_key not in f:
                raise KeyError(f'{path.name}: missing key {in_key}')
            out[out_key] = np.asarray(f[in_key])
    return out


def split_idx(n: int, seed: int = 42, train: float = 0.7, val: float = 0.15):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    nt = int(round(train * n))
    nv = int(round(val * n))
    tr = np.sort(p[:nt])
    va = np.sort(p[nt:nt + nv])
    te = np.sort(p[nt + nv:])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def channel_norm(x: np.ndarray, train_idx: np.ndarray):
    axes = tuple(i for i in range(x.ndim) if i != 1)
    mean = np.mean(x[train_idx], axis=axes, keepdims=True)
    std = np.std(x[train_idx], axis=axes, keepdims=True)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float32), std.astype(np.float32), ((x - mean) / std).astype(np.float32)


def build_grid(bounds, res):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = res
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([xg, yg, zg], axis=-1)


def project_nearest(pxyz: np.ndarray, values: np.ndarray, grid_xyz: np.ndarray):
    # Simple robust projection for manual workflow: nearest-cell accumulation.
    flat = grid_xyz.reshape(-1, 3)
    if SCIPY_AVAILABLE:
        tree = cKDTree(flat)
        _, j = tree.query(pxyz, k=1)
    else:
        diff = pxyz[:, None, :] - flat[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        j = np.argmin(d2, axis=1)

    m = flat.shape[0]
    c = values.shape[1]
    acc = np.zeros((m, c), dtype=np.float64)
    cnt = np.zeros((m, 1), dtype=np.float64)
    for i, k in enumerate(j):
        acc[int(k)] += values[i]
        cnt[int(k)] += 1.0
    out = acc / np.maximum(cnt, 1.0)
    nx, ny, nz, _ = grid_xyz.shape
    return out.reshape(nx, ny, nz, c).astype(np.float32)


INPUT_KEYS = {
    'particle_xyz': 'X',
    'Gamma_vec': 'Gamma',
    'velocity': 'velocity',
    'velocity_gradient_x': 'velocity_gradient_x',
    'velocity_gradient_y': 'velocity_gradient_y',
    'velocity_gradient_z': 'velocity_gradient_z',
    'sigma': 'sigma',
    'circulation': 'circulation',
    'vol': 'vol',
    'static': 'static',
}
OUTPUT_KEYS = {'points': 'nodes', 'U': 'U', 'W': 'W'}


def merge_frames() -> List[Path]:
    ensure_dir(MERGED_ROOT)
    merged: List[Path] = []

    for ds in DATASET_IDS:
        root = RAW_ROOT / ds
        in_h5 = sorted(root.glob(INPUT_H5_PATTERN))
        out_h5 = sorted(root.glob(OUTPUT_H5_PATTERN))
        vtk = {frame_id(p): p for p in sorted(root.glob(VTK_PATTERN))}

        in_map = {frame_id(p): p for p in in_h5}
        out_map = {frame_id(p): p for p in out_h5}
        common = sorted(set(in_map).intersection(out_map))
        print(f'[merge] {ds}: input={len(in_h5)} output={len(out_h5)} paired={len(common)}')

        out_dir = ensure_dir(MERGED_ROOT / ds)
        for fr in common:
            pin = in_map[fr]
            pout = out_map[fr]
            payload = {}
            payload.update(read_h5_selected(pin, INPUT_KEYS))
            payload.update(read_h5_selected(pout, OUTPUT_KEYS))
            payload['source_dataset'] = np.asarray(ds, dtype=object)
            payload['frame_id'] = np.asarray(fr, dtype=object)
            payload['source_vtk_path'] = np.asarray(str(vtk.get(fr, '')), dtype=object)
            payload['source_input_h5_path'] = np.asarray(str(pin), dtype=object)
            payload['source_output_h5_path'] = np.asarray(str(pout), dtype=object)

            op = out_dir / f'{ds}__frame_{fr}.npz'
            np.savez_compressed(op, **payload)
            merged.append(op)

    merged = sorted(merged)
    print('[merge] total merged:', len(merged))
    return merged


def compute_bounds(merged: List[Path]):
    mn = np.array([np.inf, np.inf, np.inf])
    mx = np.array([-np.inf, -np.inf, -np.inf])
    for p in merged:
        with np.load(p, allow_pickle=True) as d:
            xyz = as_xyz(d['particle_xyz'])
        mn = np.minimum(mn, xyz.min(axis=0))
        mx = np.maximum(mx, xyz.max(axis=0))
    span = np.maximum(mx - mn, 1e-9)
    pad = 0.05 * span
    mn -= pad
    mx += pad
    return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2])


def build_field_dataset(merged: List[Path]) -> Path:
    bounds = compute_bounds(merged)
    grid = build_grid(bounds, GRID_RESOLUTION)

    Xs, Ys = [], []
    cases, frames = [], []

    for p in merged:
        with np.load(p, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}

        xyz = as_xyz(data['particle_xyz'])
        gamma = as_xyz(data['Gamma_vec'])
        n = xyz.shape[0]
        sigma = np.asarray(data['sigma']).reshape(-1)
        if sigma.size < n:
            s = np.zeros(n, dtype=np.float64)
            s[:sigma.size] = sigma
            sigma = s
        sigma = sigma[:n]

        vol = np.asarray(data['vol']).reshape(-1)
        if vol.size == 0:
            vol = np.ones(n, dtype=np.float64)
        if vol.size < n:
            vv = np.ones(n, dtype=np.float64)
            vv[:vol.size] = vol
            vol = vv
        vol = vol[:n]
        density = vol / max(float(np.mean(vol)), 1e-12)

        chans = {
            'Gamma_x': gamma[:, 0],
            'Gamma_y': gamma[:, 1],
            'Gamma_z': gamma[:, 2],
            'sigma': sigma,
            'density': density,
        }
        coord = {'X': grid[..., 0], 'Y': grid[..., 1], 'Z': grid[..., 2]}

        projected = {}
        for k, v in chans.items():
            g = project_nearest(xyz, v.reshape(-1, 1), grid)
            projected[k] = g[..., 0]

        merged_channels = {**projected, **coord}
        channel_order = [c for c in FIELD_INPUT_CHANNELS if c in merged_channels]
        x = np.stack([merged_channels[c] for c in channel_order], axis=0).astype(np.float32)

        src_pts = as_xyz(data['points']) if 'points' in data else None
        yparts = []
        if OUTPUT_MODE in ('U', 'UW'):
            U = as_vec_field(data['U'])
            if U.ndim == 4 and U.shape[:3] == GRID_RESOLUTION:
                yparts.append(np.moveaxis(U, -1, 0))
            elif U.ndim == 2 and src_pts is not None and U.shape[0] == src_pts.shape[0]:
                Ug = project_nearest(src_pts, U, grid)
                yparts.append(np.moveaxis(Ug, -1, 0))
            else:
                raise RuntimeError(f'Cannot map U shape {U.shape}')

        if OUTPUT_MODE in ('W', 'UW'):
            W = as_vec_field(data['W'])
            if W.ndim == 4 and W.shape[:3] == GRID_RESOLUTION:
                yparts.append(np.moveaxis(W, -1, 0))
            elif W.ndim == 2 and src_pts is not None and W.shape[0] == src_pts.shape[0]:
                Wg = project_nearest(src_pts, W, grid)
                yparts.append(np.moveaxis(Wg, -1, 0))
            else:
                raise RuntimeError(f'Cannot map W shape {W.shape}')

        y = np.concatenate(yparts, axis=0).astype(np.float32)
        Xs.append(x)
        Ys.append(y)

        cases.append(str(np.asarray(data['source_dataset']).reshape(-1)[0]))
        frames.append(str(np.asarray(data['frame_id']).reshape(-1)[0]))

    X = np.stack(Xs).astype(np.float32)
    Y = np.stack(Ys).astype(np.float32)

    tr, va, te = split_idx(len(X), RANDOM_SEED)
    in_mean, in_std, Xn = channel_norm(X, tr)
    out_mean, out_std, Yn = channel_norm(Y, tr)

    target_channels = ['U_x', 'U_y', 'U_z'] if OUTPUT_MODE == 'U' else ['W_x', 'W_y', 'W_z']
    if OUTPUT_MODE == 'UW':
        target_channels = ['U_x', 'U_y', 'U_z', 'W_x', 'W_y', 'W_z']

    out_path = OUT_ROOT / 'field_dataset.npz'
    np.savez_compressed(
        out_path,
        inputs=X, targets=Y,
        inputs_norm=Xn, targets_norm=Yn,
        input_channels=np.asarray(FIELD_INPUT_CHANNELS, dtype=object),
        target_channels=np.asarray(target_channels, dtype=object),
        grid_xyz=grid.astype(np.float32),
        bounds=np.asarray(bounds, dtype=np.float32),
        case_names=np.asarray(cases, dtype=object),
        frame_ids=np.asarray(frames, dtype=object),
        train_idx=tr, val_idx=va, test_idx=te,
        in_mean=in_mean, in_std=in_std,
        out_mean=out_mean, out_std=out_std,
    )
    print('[field] wrote', out_path, 'shape', X.shape, Y.shape)
    return out_path


def build_particle_dataset(merged: List[Path]) -> Path:
    by_case = {}
    for p in merged:
        by_case.setdefault(p.parent.name, []).append(p)
    for c in by_case:
        by_case[c] = sorted(by_case[c], key=lambda x: frame_id(x))

    rows_x, rows_y = [], []
    frame_ranges = []
    frame_contexts = []
    start = 0

    for case, plist in sorted(by_case.items()):
        ncase = len(plist)
        for i, p in enumerate(plist):
            with np.load(p, allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}

            xyz = as_xyz(data['particle_xyz'])
            gamma = as_xyz(data['Gamma_vec'])
            vel = as_xyz(data['velocity'])
            gx = as_xyz(data['velocity_gradient_x'])
            gy = as_xyz(data['velocity_gradient_y'])
            gz = as_xyz(data['velocity_gradient_z'])
            grad = np.concatenate([gx, gy, gz], axis=1)

            n = xyz.shape[0]
            sigma = np.asarray(data['sigma']).reshape(-1)
            vol = np.asarray(data['vol']).reshape(-1)
            circ = np.asarray(data['circulation']).reshape(-1)
            stat = np.asarray(data['static']).reshape(-1)

            def fit(arr, default):
                if arr.size == 0:
                    return np.full(n, default, dtype=np.float64)
                if arr.size < n:
                    x = np.full(n, default, dtype=np.float64)
                    x[:arr.size] = arr
                    return x
                return arr[:n].astype(np.float64)

            sigma = fit(sigma, 1e-2)
            vol = fit(vol, 1.0)
            circ = fit(circ, float(np.mean(np.linalg.norm(gamma, axis=1))))
            stat = fit(stat, 0.0)

            phase = 0.0 if ncase <= 1 else float(i) / float(ncase - 1)
            aoa = 0.0
            fs = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            feat = {
                'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
                'Gamma_x': gamma[:, 0], 'Gamma_y': gamma[:, 1], 'Gamma_z': gamma[:, 2],
                'sigma': sigma, 'vol': vol, 'circulation': circ, 'static': stat,
                'phase': np.full(n, phase), 'angle_of_attack': np.full(n, aoa),
                'freestream_x': np.full(n, fs[0]), 'freestream_y': np.full(n, fs[1]), 'freestream_z': np.full(n, fs[2]),
            }

            x = np.stack([np.asarray(feat[k], dtype=np.float32) for k in PARTICLE_INPUT_FEATURES], axis=1)
            y = np.concatenate([vel, grad], axis=1).astype(np.float32)

            end = start + n
            fr = str(np.asarray(data['frame_id']).reshape(-1)[0])
            frame_ranges.append((case, fr, start, end))
            frame_contexts.append({'case': case, 'frame': fr, 'start': start, 'end': end, 'phase': phase, 'dt': None})
            rows_x.append(x)
            rows_y.append(y)
            start = end

    X = np.concatenate(rows_x, axis=0).astype(np.float32)
    Y = np.concatenate(rows_y, axis=0).astype(np.float32)

    nf = len(frame_ranges)
    trf, vaf, tef = split_idx(nf, RANDOM_SEED)

    def rows_from_frames(fid):
        chunks = []
        for i in fid:
            _, _, s, e = frame_ranges[int(i)]
            chunks.append(np.arange(int(s), int(e), dtype=np.int64))
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)

    tr = rows_from_frames(trf)
    va = rows_from_frames(vaf)
    te = rows_from_frames(tef)

    in_mean = np.mean(X[tr], axis=0, keepdims=True)
    in_std = np.maximum(np.std(X[tr], axis=0, keepdims=True), 1e-8)
    out_mean = np.mean(Y[tr], axis=0, keepdims=True)
    out_std = np.maximum(np.std(Y[tr], axis=0, keepdims=True), 1e-8)

    Xn = ((X - in_mean) / in_std).astype(np.float32)
    Yn = ((Y - out_mean) / out_std).astype(np.float32)

    target_names = [
        'velocity_x','velocity_y','velocity_z',
        'gradUx_x','gradUx_y','gradUx_z',
        'gradUy_x','gradUy_y','gradUy_z',
        'gradUz_x','gradUz_y','gradUz_z',
    ]

    out_path = OUT_ROOT / 'particle_dataset.npz'
    np.savez_compressed(
        out_path,
        inputs_particle=X,
        targets_particle=Y,
        inputs_particle_norm=Xn,
        targets_particle_norm=Yn,
        feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        target_names=np.asarray(target_names, dtype=object),
        frame_ranges=np.asarray(frame_ranges, dtype=object),
        frame_contexts=np.asarray(frame_contexts, dtype=object),
        train_rows=tr, val_rows=va, test_rows=te,
        train_frames=trf, val_frames=vaf, test_frames=tef,
        in_mean=in_mean.astype(np.float32), in_std=in_std.astype(np.float32),
        out_mean=out_mean.astype(np.float32), out_std=out_std.astype(np.float32),
    )
    print('[particle] wrote', out_path, 'rows', X.shape, Y.shape, 'frames', nf)
    return out_path


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(MERGED_ROOT)

    merged = merge_frames()
    fpath = build_field_dataset(merged)
    ppath = build_particle_dataset(merged)

    summary = {
        'raw_root': str(RAW_ROOT),
        'datasets': DATASET_IDS,
        'n_merged_frames': len(merged),
        'field_dataset': str(fpath),
        'particle_dataset': str(ppath),
        'grid_resolution': GRID_RESOLUTION,
        'output_mode': OUTPUT_MODE,
    }
    (OUT_ROOT / 'preprocess_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
