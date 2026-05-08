"""I/O utilities for the geometry-aware preprocessing pipeline.

This module is intentionally self-contained and does not mutate existing project files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FrameEntry:
    """One simulation frame pairing particle data and optional geometry."""

    frame_id: str
    case_name: str
    npz_path: Path
    vtk_path: Optional[Path] = None
    time: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


FRAME_ID_REGEX = re.compile(r"(\d+)(?!.*\d)")


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (including parents) if it does not exist and return it as `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load JSON or YAML config.

    YAML support is optional and only activated when PyYAML is installed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".json"}:
        return json.loads(p.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install pyyaml or use JSON config."
            ) from exc
        return yaml.safe_load(p.read_text())

    raise ValueError(f"Unsupported config extension: {p.suffix}")


def save_json(path: Path | str, payload: Dict[str, Any]) -> None:
    """Write a JSON file with indentation and ensure the parent folder exists first."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2))


def parse_frame_id(path: Path) -> str:
    """Extract frame id from filename using trailing integer when available."""
    stem = path.stem
    match = FRAME_ID_REGEX.search(stem)
    if match:
        return match.group(1).zfill(6)
    return stem


def discover_files(glob_patterns: Sequence[str], root: Path | str = ".") -> List[Path]:
    """Resolve one or more glob patterns relative to `root` and return a sorted unique file list."""
    root_path = Path(root)
    found: List[Path] = []
    for pattern in glob_patterns:
        found.extend(sorted(root_path.glob(pattern)))
    return sorted(set(found))


def _index_by_frame(files: Iterable[Path]) -> Dict[str, Path]:
    """Index files by parsed frame id so frame-based pairing can be done quickly."""
    out: Dict[str, Path] = {}
    for f in sorted(files):
        frame_id = parse_frame_id(f)
        out[frame_id] = f
    return out


def pair_npz_vtk(npz_files: Sequence[Path], vtk_files: Sequence[Path]) -> List[Tuple[Path, Optional[Path], str]]:
    """Pair each NPZ frame with an optional VTK frame by matching extracted frame ids."""
    vtk_map = _index_by_frame(vtk_files)
    pairs: List[Tuple[Path, Optional[Path], str]] = []
    for npz in sorted(npz_files):
        frame_id = parse_frame_id(npz)
        pairs.append((npz, vtk_map.get(frame_id), frame_id))
    return pairs


def discover_case_entries(case_cfg: Dict[str, Any], root: Path | str = ".") -> List[FrameEntry]:
    """Build `FrameEntry` list for one case.

    Required keys in case config:
      - name
      - npz_glob (str or list[str])

    Optional:
      - vtk_glob (str or list[str])
      - time_step
      - any additional metadata keys (copied into entry.meta)
    """
    root_path = Path(root)
    case_name = str(case_cfg["name"])

    npz_glob = case_cfg["npz_glob"]
    npz_patterns = [npz_glob] if isinstance(npz_glob, str) else list(npz_glob)
    npz_files = discover_files(npz_patterns, root=root_path)

    vtk_files: List[Path] = []
    if case_cfg.get("vtk_glob"):
        vtk_glob = case_cfg["vtk_glob"]
        vtk_patterns = [vtk_glob] if isinstance(vtk_glob, str) else list(vtk_glob)
        vtk_files = discover_files(vtk_patterns, root=root_path)

    pairs = pair_npz_vtk(npz_files, vtk_files)

    extra_meta = {
        k: v
        for k, v in case_cfg.items()
        if k
        not in {
            "name",
            "npz_glob",
            "vtk_glob",
            "time_step",
        }
    }
    dt = case_cfg.get("time_step")

    entries: List[FrameEntry] = []
    for idx, (npz_path, vtk_path, frame_id) in enumerate(pairs):
        t = None if dt is None else float(dt) * idx
        entries.append(
            FrameEntry(
                frame_id=frame_id,
                case_name=case_name,
                npz_path=npz_path,
                vtk_path=vtk_path,
                time=t,
                meta=dict(extra_meta),
            )
        )

    return entries


def split_entries(
    entries: Sequence[FrameEntry],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    strategy: str = "random",
) -> Dict[str, List[FrameEntry]]:
    """Split by frame with reproducible random or chronological ordering."""
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    ordered = list(entries)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ordered)).tolist()
        ordered = [ordered[i] for i in perm]
    elif strategy == "chronological":
        ordered = sorted(ordered, key=lambda e: (e.case_name, e.frame_id))
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    n = len(ordered)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    n_test = n - n_train - n_val

    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val : n_train + n_val + n_test]

    return {"train": train, "val": val, "test": test}


def entries_to_records(entries: Sequence[FrameEntry]) -> List[Dict[str, Any]]:
    """Convert `FrameEntry` objects into plain dictionaries for logging and JSON export."""
    recs: List[Dict[str, Any]] = []
    for e in entries:
        recs.append(
            {
                "frame_id": e.frame_id,
                "case_name": e.case_name,
                "npz_path": str(e.npz_path),
                "vtk_path": None if e.vtk_path is None else str(e.vtk_path),
                "time": e.time,
                "meta": e.meta,
            }
        )
    return recs


def save_split_index(path: Path | str, split: Dict[str, List[FrameEntry]]) -> None:
    """Persist train/val/test split metadata to disk as JSON for reproducible training."""
    payload = {k: entries_to_records(v) for k, v in split.items()}
    save_json(path, payload)


def npz_to_dict(path: Path | str) -> Dict[str, np.ndarray]:
    """Load an `.npz` archive into a normal Python dict of numpy arrays."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def pretty_shape(a: np.ndarray) -> str:
    """Format a numpy shape tuple into a compact `a x b x c` style string."""
    return "x".join(str(int(s)) for s in a.shape)
