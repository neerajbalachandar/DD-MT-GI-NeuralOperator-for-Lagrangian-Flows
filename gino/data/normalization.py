from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray
    field_mean: np.ndarray
    field_std: np.ndarray
    coord_min: np.ndarray
    coord_span: np.ndarray
    state_mean: np.ndarray = None
    state_std: np.ndarray = None

    @classmethod
    def from_checkpoint(cls, checkpoint):
        nested = checkpoint.get("normalization", {})
        def required(primary, fallback=None):
            value = checkpoint.get(primary, nested.get(fallback or primary))
            if value is None:
                raise ValueError(f"Checkpoint is missing required normalization statistic: {primary}")
            return np.asarray(value, dtype=np.float32).reshape(-1)
        return cls(
            required("input_mean"), required("input_std"),
            required("target_mean", "residual_mean"), required("target_std", "residual_std"),
            required("field_mean"), required("field_std"),
            required("coord_min"), required("coord_span"),
            required("state_mean")[:7], required("state_std")[:7],
        )

    @classmethod
    def fit_train_sequences(cls, data, pair_ids, feature_indices):
        pair_ids = np.asarray(pair_ids, dtype=np.int64)
        if pair_ids.size == 0:
            raise ValueError("Cannot fit normalization from an empty training split")
        ranges = list(data["pair_ranges"])
        rows = np.concatenate([np.arange(int(ranges[int(i)][3]), int(ranges[int(i)][4])) for i in pair_ids])
        x = np.asarray(data["inputs_t"], dtype=np.float32)[rows][:, feature_indices]
        residual_key = "targets_residual" if "targets_residual" in data else "targets_delta"
        residual = np.asarray(data[residual_key], dtype=np.float32)
        residual_rows = residual[rows]
        qmask = np.asarray(data["field_query_mask"], dtype=bool)[pair_ids]
        field = np.asarray(data["targets_velocity_field"], dtype=np.float32)[pair_ids][qmask]
        coords = np.asarray(data["query_coords"], dtype=np.float32)[pair_ids][qmask]
        pos_idx = [list(data["feature_names"]).index(n) for n in ("x", "y", "z")]
        particle_coords = np.asarray(data["inputs_t"], dtype=np.float32)[rows][:, pos_idx]
        combined = np.concatenate((particle_coords, coords), axis=0) if len(coords) else particle_coords
        if "targets_next_state" in data:
            next_states = np.asarray(data["targets_next_state"], dtype=np.float32)[rows]
            state_mean, state_std = next_states.mean(0)[:7], np.maximum(next_states.std(0)[:7], 1e-8)
        else:
            feature_names = [str(name) for name in data["feature_names"]]
            state_mean, state_std = np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)
            state_features = ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
            input_rows = np.asarray(data["inputs_t"], dtype=np.float32)[rows]
            for channel, name in enumerate(state_features):
                if name in feature_names:
                    values = input_rows[:, feature_names.index(name)]
                    state_mean[channel], state_std[channel] = values.mean(), max(float(values.std()), 1e-8)
        return cls(x.mean(0), np.maximum(x.std(0), 1e-8), residual_rows.mean(0),
                   np.maximum(residual_rows.std(0), 1e-8), field.mean(0),
                   np.maximum(field.std(0), 1e-8), combined.min(0),
                   np.maximum(np.ptp(combined, axis=0), 1e-8), state_mean, state_std)

    def _tensor(self, value, ref):
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def denormalize_residual(self, value):
        channels = value.shape[-1]
        mean = self._tensor(self.residual_mean[:channels], value)
        std = self._tensor(self.residual_std[:channels], value)
        return value * std + mean

    def denormalize_field(self, value):
        return value * self._tensor(self.field_std, value) + self._tensor(self.field_mean, value)

    def normalize_input(self, value):
        return (value - self._tensor(self.input_mean, value)) / self._tensor(self.input_std, value).clamp_min(1e-8)

    def normalize_positions(self, value):
        return (value - self._tensor(self.coord_min, value)) / self._tensor(self.coord_span, value).clamp_min(1e-8)


def _array(x):
    return x.item() if isinstance(x, np.ndarray) and x.shape == () else x
