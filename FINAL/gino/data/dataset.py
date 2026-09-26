from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .context import RolloutContext


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
                 max_queries=2048, rollout_horizon=16, normalization=None, residual_channels=None,
                 include_teacher_inputs=False):
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
        self.include_teacher_inputs = bool(include_teacher_inputs)
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

    def _pair_particle_ids(self, pair_id, kind="t"):
        row = self.data["pair_ranges"][int(pair_id)]
        start, end = int(row[3]), int(row[4])
        key = f"particle_ids_{kind}"
        if key in self.data:
            ids = np.asarray([str(value) for value in np.asarray(self.data[key][start:end]).reshape(-1)])
        else:
            # Legacy archives can only rely on the original canonical row order.
            ids = np.asarray([f"legacy-row:{i}" for i in range(end - start)])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Pair {pair_id} has duplicate particle identities in {key}")
        return ids

    def __getitem__(self, index):
        pid = int(self.pair_ids[index])
        row = self.data["pair_ranges"][pid]
        _, _, _, start, end, count = row
        start, end, count = int(start), int(end), int(count)
        contexts = self.data.get("pair_contexts")
        context = contexts[pid] if contexts is not None else {}
        if isinstance(context, np.ndarray):
            context = context.item()

        future_pair_ids = []
        future_id_sets = []
        rollout_time_indices = []
        next_id = self.next_pair.get(pid)
        expected_frame = str(row[2])
        while next_id is not None and len(future_pair_ids) < self.rollout_horizon - 1:
            next_row = self.data["pair_ranges"][next_id]
            if str(next_row[0]) != str(row[0]) or str(next_row[1]) != expected_frame:
                raise ValueError(f"Broken adjacent pair chain after pair {pid}: expected frame {expected_frame}")
            future_pair_ids.append(int(next_id))
            future_id_sets.append(self._pair_particle_ids(next_id, "t"))
            rollout_time_indices.append(len(future_pair_ids) + 1)
            expected_frame = str(next_row[2])
            next_id = self.next_pair.get(next_id)

        current_ids = self._pair_particle_ids(pid, "t")
        next_ids = self._pair_particle_ids(pid, "tp1")
        if not np.array_equal(current_ids, next_ids):
            raise ValueError(f"Pair {pid} preprocessing did not align current/next particle identities")
        future_id_sets = [set(ids.tolist()) for ids in future_id_sets]
        common_ids = [identity for identity in current_ids
                      if all(identity in available for available in future_id_sets)]
        common_ids = common_ids[:self.max_particles]
        if not common_ids:
            raise ValueError(f"Pair {pid} has no particle identities valid across the requested rollout")
        current_lookup = {identity: i for i, identity in enumerate(current_ids.tolist())}
        selected_indices = np.asarray([current_lookup[identity] for identity in common_ids], dtype=np.int64)
        count = len(selected_indices)
        selected_rows = start + selected_indices
        future_indices = []
        for future_pid in future_pair_ids:
            future_ids = self._pair_particle_ids(future_pid, "t")
            future_target_ids = self._pair_particle_ids(future_pid, "tp1")
            if not np.array_equal(future_ids, future_target_ids):
                raise ValueError(f"Pair {future_pid} preprocessing did not align current/next particle identities")
            lookup = {identity: i for i, identity in enumerate(future_ids.tolist())}
            future_indices.append(np.asarray([lookup[identity] for identity in common_ids], dtype=np.int64))
        full = np.asarray(self.data["inputs_t"][selected_rows], dtype=np.float32)
        x = full[:, self.feature_indices]
        mean, std = self.input_mean, self.input_std
        x = (x - mean) / std
        raw_names = [str(n) for n in self.data["feature_names"]]
        xyz_idx = [raw_names.index(k) for k in ("x", "y", "z")]
        xyz = full[:, xyz_idx]
        coord_min, coord_span = self.coord_min, self.coord_span
        geom = np.clip((xyz - coord_min) / coord_span, 0.0, 1.0)
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
        next_state = np.asarray(self.data["targets_next_state"][selected_rows], dtype=np.float32)
        residuals = np.asarray(self.data.get("targets_residual", self.data["targets_delta"])[selected_rows], dtype=np.float32)
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
        future_fields, future_queries, future_contexts, future_teacher_inputs = [], [], [], []
        for future_pid, row_indices in zip(future_pair_ids, future_indices):
            next_id = future_pid
            next_row = self.data["pair_ranges"][next_id]
            ns = int(next_row[3])
            future_rows = ns + row_indices
            future.append(np.asarray(self.data["targets_next_state"][future_rows, :7], dtype=np.float32))
            if self.include_teacher_inputs:
                next_raw = np.asarray(self.data["inputs_t"][ns + row_indices], dtype=np.float32)[:, self.feature_indices]
                future_teacher_inputs.append((next_raw - self.input_mean) / self.input_std)
            future_mask = np.flatnonzero(np.asarray(self.data["field_query_mask"][next_id], dtype=bool))
            if len(future_mask) > self.max_queries:
                future_mask = future_mask[np.linspace(0, len(future_mask) - 1, self.max_queries, dtype=np.int64)]
            future_xyz = np.asarray(self.data["query_coords"][next_id, future_mask], dtype=np.float32)
            future_queries.append(np.clip((future_xyz - coord_min) / coord_span, 0.0, 1.0))
            future_fields.append((np.asarray(self.data["targets_velocity_field"][next_id, future_mask], dtype=np.float32) - fmean) / fstd)
            next_context = self.data.get("pair_contexts", [])[next_id]
            if isinstance(next_context, np.ndarray):
                next_context = next_context.item()
            future_contexts.append(RolloutContext.from_mapping(next_context, pair_id=next_id))
        if future:
            future = np.stack(future, axis=0)
            rollout_states = np.concatenate((next_state[None, :, :7], future), axis=0)
            q_common = min([len(queries)] + [len(x) for x in future_queries])
            queries, coords, field = queries[:q_common], coords[:q_common], field[:q_common]
            future_queries = np.stack([x[:q_common] for x in future_queries], axis=0)
            future_fields = np.stack([x[:q_common] for x in future_fields], axis=0)
            if self.include_teacher_inputs:
                future_teacher_inputs = np.stack(future_teacher_inputs, axis=0)
            rollout_phases = np.asarray([context.get("phase_tp1", context.get("phase_t", 0.0))] +
                                        [item.get("phase_tp1", item.get("phase_t", 0.0)) for item in future_contexts],
                                        dtype=np.float32)
        else:
            future = np.zeros((0, count, 7), dtype=np.float32)
            rollout_states = next_state[None, :, :7]
            future_queries = np.zeros((0, len(queries), 3), dtype=np.float32)
            future_fields = np.zeros((0, len(queries), len(fmean)), dtype=np.float32)
            if self.include_teacher_inputs:
                future_teacher_inputs = np.zeros((0, count, len(self.features)), dtype=np.float32)
            rollout_phases = np.asarray([context.get("phase_tp1", context.get("phase_t", 0.0))], dtype=np.float32)
        rollout_field_targets = np.concatenate((field[None], future_fields), axis=0)
        return {"pair_id": pid, "input_geom": geom, "output_queries": queries, "x": x,
                "global_params": np.asarray(global_values, dtype=np.float32), "state_phys": full[:, state_idx],
                "delta_target": (residuals - residual_mean) / residual_std,
                "field_target": field, "dt": float(context.get("dt", 0.0034)),
                "next_state_phys": next_state[:, :7], "one_step_target": next_state[:, :7],
                "rollout_state_targets": rollout_states,
                "rollout_queries": future_queries, "rollout_field_targets": rollout_field_targets,
                "rollout_contexts": future_contexts,
                "rollout_time_indices": np.arange(1, len(rollout_states) + 1, dtype=np.int64),
                "rollout_phases": rollout_phases,
                "feature_names": self.features, "all_feature_names": raw_names,
                "global_feature_names": self.global_features,
                "pair_context": RolloutContext.from_mapping(context, pair_id=pid),
                "input_mean": self.input_mean, "input_std": self.input_std,
                "particle_queries": geom.copy(),
                "query_xyz_phys": coords, "particle_features_phys": full,
                "phase_next": float(context.get("phase_tp1", context.get("phase_t", 0.0))),
                "phase_delta": float(context.get("phase_delta", 0.0)),
                "particle_ids": np.asarray(common_ids, dtype=str),
                "particle_correspondence": str(context.get("correspondence_source", "legacy_row_index_assumption_unverified")),
                "rollout_teacher_inputs": (np.stack(future_teacher_inputs, axis=0) if self.include_teacher_inputs and future_teacher_inputs else
                                           np.zeros((0, count, len(self.features)), dtype=np.float32) if self.include_teacher_inputs else None)}



def collate_one(items):
    item = items[0]
    batch = {k: (torch.as_tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else v)
             for k, v in item.items()}
    batch["pair_id"] = torch.tensor([int(item["pair_id"])], dtype=torch.long)
    return batch
