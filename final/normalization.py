"""Dataset-level normalization utilities (train split only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


@dataclass
class ChannelStats:
    """Per-channel normalization statistics (mean, std, and total observed sample count)."""
    mean: np.ndarray
    std: np.ndarray
    count: int


class RunningChannelStats:
    """Streaming per-channel mean/std over tensors shaped (C, ...)."""

    def __init__(self, n_channels: int):
        """Initialize running sums for streaming mean/std estimation across channels."""
        self.n_channels = int(n_channels)
        self.count = 0
        self.sum = np.zeros(self.n_channels, dtype=np.float64)
        self.sumsq = np.zeros(self.n_channels, dtype=np.float64)

    def update(self, tensor: np.ndarray) -> None:
        """Accumulate per-channel first and second moments from one `(C, ...)` tensor sample."""
        x = np.asarray(tensor, dtype=np.float64)
        if x.ndim < 1:
            raise ValueError("tensor must have at least 1 dimension")
        if x.shape[0] != self.n_channels:
            raise ValueError(
                f"channel mismatch: expected {self.n_channels}, got {x.shape[0]}"
            )

        flat = x.reshape(self.n_channels, -1)
        self.sum += flat.sum(axis=1)
        self.sumsq += (flat ** 2).sum(axis=1)
        self.count += int(flat.shape[1])

    def finalize(self, eps: float = 1e-8) -> ChannelStats:
        """Finalize running moments into stable mean/std channel statistics."""
        if self.count == 0:
            raise RuntimeError("No data observed while fitting normalization stats")

        mean = self.sum / float(self.count)
        var = self.sumsq / float(self.count) - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std = np.maximum(std, eps)
        return ChannelStats(mean=mean.astype(np.float32), std=std.astype(np.float32), count=self.count)


def fit_stats_from_npz(
    sample_paths: Iterable[Path],
    key: str,
) -> ChannelStats:
    """Fit streaming per-channel normalization statistics over a list of sample NPZ files."""
    paths = list(sample_paths)
    if not paths:
        raise RuntimeError("No sample paths passed to fit_stats_from_npz")

    first = np.load(paths[0], allow_pickle=False)[key]
    n_channels = int(first.shape[0])
    running = RunningChannelStats(n_channels)

    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            running.update(np.asarray(d[key]))

    return running.finalize()


def apply_channel_norm(tensor: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Apply channel-wise normalization to a tensor of shape `(C, ...)`."""
    x = np.asarray(tensor, dtype=np.float32)
    mean = stats.mean.reshape(-1, *([1] * (x.ndim - 1)))
    std = stats.std.reshape(-1, *([1] * (x.ndim - 1)))
    return (x - mean) / std


def invert_channel_norm(tensor: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Undo channel-wise normalization and map normalized values back to physical scale."""
    x = np.asarray(tensor, dtype=np.float32)
    mean = stats.mean.reshape(-1, *([1] * (x.ndim - 1)))
    std = stats.std.reshape(-1, *([1] * (x.ndim - 1)))
    return x * std + mean


def save_stats(path: Path | str, input_stats: ChannelStats, target_stats: ChannelStats) -> None:
    """Save input and target normalization statistics to a compressed NPZ file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        input_mean=input_stats.mean,
        input_std=input_stats.std,
        input_count=np.int64(input_stats.count),
        target_mean=target_stats.mean,
        target_std=target_stats.std,
        target_count=np.int64(target_stats.count),
    )


def load_stats(path: Path | str) -> Dict[str, ChannelStats]:
    """Load saved normalization statistics and return typed `ChannelStats` objects."""
    with np.load(path, allow_pickle=False) as d:
        input_stats = ChannelStats(
            mean=np.asarray(d["input_mean"], dtype=np.float32),
            std=np.asarray(d["input_std"], dtype=np.float32),
            count=int(d["input_count"]),
        )
        target_stats = ChannelStats(
            mean=np.asarray(d["target_mean"], dtype=np.float32),
            std=np.asarray(d["target_std"], dtype=np.float32),
            count=int(d["target_count"]),
        )
    return {"input": input_stats, "target": target_stats}
