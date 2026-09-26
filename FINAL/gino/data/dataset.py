from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_processed_dataset(path):
    with np.load(Path(path), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def sequence_level_split(data, seed=42):
    cases = sorted({str(row[0]) for row in data["pair_ranges"]})
    fixed_test = {str(x) for x in data.get("test_cases", [])}
    remaining = [case for case in cases if case not in fixed_test]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    n_val = max(1, int(round(0.2 * len(remaining)))) if len(remaining) > 1 else 0
    val_cases, train_cases = set(remaining[:n_val]), set(remaining[n_val:])
    out = {"train_pair_ids": [], "val_pair_ids": [], "test_pair_ids": [],
           "train_cases": sorted(train_cases), "val_cases": sorted(val_cases), "test_cases": sorted(fixed_test)}
    for i, row in enumerate(data["pair_ranges"]):
        case = str(row[0])
        key = "test_pair_ids" if case in fixed_test else ("val_pair_ids" if case in val_cases else "train_pair_ids")
        out[key].append(i)
    for key in ("train_pair_ids", "val_pair_ids", "test_pair_ids"):
        out[key] = np.asarray(out[key], dtype=np.int64)
    return out


def assert_no_sequence_leakage(data, splits):
    cases = {}
    for split in ("train", "val", "test"):
        cases[split] = {str(data["pair_ranges"][int(i)][0]) for i in splits[f"{split}_pair_ids"]}
    if cases["train"] & cases["val"] or cases["train"] & cases["test"] or cases["val"] & cases["test"]:
        raise ValueError("A case appears in more than one split")


class EvolutionDataset(Dataset):
    def __init__(self, data, pair_ids, input_features, global_features, max_particles=4096,
                 max_queries=2048, rollout_horizon=16, normalization=None, residual_channels=None):
        self.data = data
        self.pair_ids = np.asarray(pair_ids, dtype=np.int64)
        self.features = [str(x) for x in input_features]
        self.global_features = [str(x) for x in global_features]
        names = [str(x) for x in data["feature_names"]]
        self.feature_indices = [names.index(name) for name in self.features]
        self.max_particles, self.max_queries = int(max_particles), int(max_queries)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.normalization = normalization
        self.residual_channels = residual_channels
        self.coord_min = np.asarray(data["coord_min"], dtype=np.float32).reshape(3) if normalization is None else np.asarray(normalization.coord_min).reshape(3)
        self.coord_span = np.maximum(np.asarray(data["coord_span"], dtype=np.float32).reshape(3), 1e-8) if normalization is None else np.maximum(np.asarray(normalization.coord_span).reshape(3), 1e-8)
        self.input_mean = (np.asarray(data["in_mean"], dtype=np.float32).reshape(-1)[self.feature_indices]
                           if normalization is None else np.asarray(normalization.input_mean).reshape(-1))
        self.input_std = (np.maximum(np.asarray(data["in_std"], dtype=np.float32).reshape(-1)[self.feature_indices], 1e-8)
                          if normalization is None else np.maximum(np.asarray(normalization.input_std).reshape(-1), 1e-8))
        ranges = list(data["pair_ranges"])
        self.next_pair = {}
        starts = {(str(row[0]), str(row[1])): i for i, row in enumerate(ranges)}
        for i, row in enumerate(ranges):
            self.next_pair[i] = starts.get((str(row[0]), str(row[2])))

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, index):
        pid = int(self.pair_ids[index])
        row = self.data["pair_ranges"][pid]
        _, _, _, start, end, count = row
        count = min(int(count), self.max_particles)
        start, end = int(start), int(start) + count
        full = np.asarray(self.data["inputs_t"][start:end], dtype=np.float32)
        x = full[:, self.feature_indices]
        mean = np.asarray(self.data["in_mean"], dtype=np.float32).reshape(-1)[self.feature_indices]
        std = np.maximum(np.asarray(self.data["in_std"], dtype=np.float32).reshape(-1)[self.feature_indices], 1e-8)
        x = (x - mean) / std
        raw_names = [str(n) for n in self.data["feature_names"]]
        xyz_idx = [raw_names.index(k) for k in ("x", "y", "z")]
        xyz = full[:, xyz_idx]
        coord_min, coord_span = self.coord_min, self.coord_span
        geom = np.clip((xyz - coord_min) / coord_span, 0.0, 1.0)
        contexts = self.data.get("pair_contexts")
        context = contexts[pid] if contexts is not None else {}
        if isinstance(context, np.ndarray):
            context = context.item()
        global_values = []
        for name in self.global_features:
            if name == "phase": value = context.get("phase_t", 0.0)
            elif name == "angle_of_attack": value = context.get("aoa_deg", 0.0) / 45.0
            elif name == "freestream_magnitude": value = np.linalg.norm(context.get("freestream", [0, 0, 0])) / 10.0
            elif name in ("freestream_x", "freestream_y", "freestream_z"):
                value = context.get("freestream", [0, 0, 0])["xyz".index(name[-1])] / 10.0
            else: value = 0.0
            global_values.append(value)
        state_idx = [raw_names.index(k) for k in ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")]
        next_state = np.asarray(self.data["targets_next_state"][start:end], dtype=np.float32)
        residuals = np.asarray(self.data.get("targets_residual", self.data["targets_delta"])[start:end], dtype=np.float32)
        residual_mean = np.asarray(self.data.get("residual_mean", self.data["out_mean"]), dtype=np.float32).reshape(-1) if self.normalization is None else self.normalization.residual_mean
        residual_std = np.maximum(np.asarray(self.data.get("residual_std", self.data["out_std"]), dtype=np.float32).reshape(-1), 1e-8) if self.normalization is None else np.maximum(self.normalization.residual_std, 1e-8)
        residuals = residuals[:, :int(self.residual_channels or len(residual_mean))]
        qmask = np.flatnonzero(np.asarray(self.data["field_query_mask"][pid], dtype=bool))
        if len(qmask) > self.max_queries:
            qmask = qmask[np.linspace(0, len(qmask)-1, self.max_queries, dtype=np.int64)]
        coords = np.asarray(self.data["query_coords"][pid, qmask], dtype=np.float32)
        queries = np.clip((coords - coord_min) / coord_span, 0.0, 1.0)
        fmean = np.asarray(self.data["field_mean"], dtype=np.float32).reshape(-1) if self.normalization is None else self.normalization.field_mean
        fstd = np.maximum(np.asarray(self.data["field_std"], dtype=np.float32).reshape(-1), 1e-8) if self.normalization is None else np.maximum(self.normalization.field_std, 1e-8)
        field = (np.asarray(self.data["targets_velocity_field"][pid, qmask], dtype=np.float32) - fmean) / fstd
        future = []
        future_fields, future_queries, future_contexts, future_pair_ids = [], [], [], []
        next_id = self.next_pair.get(pid)
        while next_id is not None and len(future) < self.rollout_horizon - 1:
            next_row = self.data["pair_ranges"][next_id]
            ns, ne = int(next_row[3]), int(next_row[4])
            n_future = min(count, int(next_row[5]))
            future.append(np.asarray(self.data["targets_next_state"][ns:ns + n_future], dtype=np.float32))
            future_mask = np.flatnonzero(np.asarray(self.data["field_query_mask"][next_id], dtype=bool))
            if len(future_mask) > self.max_queries:
                future_mask = future_mask[np.linspace(0, len(future_mask) - 1, self.max_queries, dtype=np.int64)]
            future_xyz = np.asarray(self.data["query_coords"][next_id, future_mask], dtype=np.float32)
            future_queries.append(np.clip((future_xyz - coord_min) / coord_span, 0.0, 1.0))
            future_fields.append((np.asarray(self.data["targets_velocity_field"][next_id, future_mask], dtype=np.float32) - fmean) / fstd)
            next_context = self.data.get("pair_contexts", [])[next_id]
            if isinstance(next_context, np.ndarray):
                next_context = next_context.item()
            future_contexts.append(dict(next_context))
            future_pair_ids.append(int(next_id))
            next_id = self.next_pair.get(next_id)
        if future:
            n_common = min([count] + [len(x) for x in future])
            future = np.stack([x[:n_common, :7] for x in future], axis=0)
            rollout_states = np.concatenate((next_state[None, :n_common, :7], future[:, :n_common, :7]), axis=0)
            q_common = min([len(queries)] + [len(x) for x in future_queries])
            queries, coords, field = queries[:q_common], coords[:q_common], field[:q_common]
            future_queries = np.stack([x[:q_common] for x in future_queries], axis=0)
            future_fields = np.stack([x[:q_common] for x in future_fields], axis=0)
            rollout_phases = np.asarray([context.get("phase_tp1", context.get("phase_t", 0.0))] +
                                        [item.get("phase_tp1", item.get("phase_t", 0.0)) for item in future_contexts],
                                        dtype=np.float32)
        else:
            future = np.zeros((0, count, 7), dtype=np.float32)
            rollout_states = next_state[None, :, :7]
            future_queries = np.zeros((0, len(queries), 3), dtype=np.float32)
            future_fields = np.zeros((0, len(queries), len(fmean)), dtype=np.float32)
            rollout_phases = np.asarray([context.get("phase_tp1", context.get("phase_t", 0.0))], dtype=np.float32)
        rollout_field_targets = np.concatenate((field[None], future_fields), axis=0)
        return {"pair_id": pid, "input_geom": geom, "output_queries": queries, "x": x,
                "global_params": np.asarray(global_values, dtype=np.float32), "state_phys": full[:, state_idx],
                "delta_target": (residuals - residual_mean) / residual_std,
                "field_target": field, "dt": float(context.get("dt", 0.0034)),
                "next_state_phys": next_state[:, :7], "one_step_target": next_state[:, :7],
                "rollout_state_targets": rollout_states,
                "rollout_queries": future_queries, "rollout_field_targets": rollout_field_targets,
                "rollout_contexts": future_contexts, "rollout_pair_ids": future_pair_ids,
                "rollout_phases": rollout_phases,
                "feature_names": self.features, "all_feature_names": raw_names,
                "global_feature_names": self.global_features, "pair_context": context,
                "input_mean": self.input_mean, "input_std": self.input_std,
                "particle_queries": geom.copy(),
                "query_xyz_phys": coords, "particle_features_phys": full,
                "phase_next": float(context.get("phase_tp1", context.get("phase_t", 0.0))),
                "phase_delta": float(context.get("phase_delta", 0.0)),
                "particle_correspondence": "canonical row ordering; shared leading rows as used by process_data.py"}


def collate_one(items):
    item = items[0]
    batch = {k: (torch.as_tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else v)
             for k, v in item.items()}
    batch["pair_id"] = torch.tensor([int(item["pair_id"])], dtype=torch.long)
    return batch
