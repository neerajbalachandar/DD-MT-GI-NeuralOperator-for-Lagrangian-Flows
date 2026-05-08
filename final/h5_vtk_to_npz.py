"""Convert split input/output H5 streams into merged per-frame NPZ files.

Why this module exists:
- Your raw simulation pipeline stores input and output in separate folders.
- Training/preprocessing is easier when each frame lives in one NPZ payload.
- This converter performs that merge while preserving traceability metadata.

This file intentionally supports the project's current workflow only:
- explicit `datasets:` config entries
- separate `input_h5_glob` and `output_h5_glob`
- optional sidecar `input_xmf_glob`, `output_xmf_glob`, and `vtk_glob`
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .io import ensure_dir, load_config, save_json


DEFAULT_FRAME_ID_REGEX = r"(\d+)(?!.*\d)"


@dataclass
class PairRecord:
    """Container for one paired input/output frame and its optional sidecar files."""

    in_frame: str
    out_frame: str
    in_h5: Path
    out_h5: Path
    in_xmf: Optional[Path]
    out_xmf: Optional[Path]
    vtk: Optional[Path]


def parse_frame_id(path: Path, regex: re.Pattern[str]) -> str:
    """Extract normalized frame id from filename using configured regex."""

    m = regex.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


def discover_many(patterns: Sequence[str], root: Path) -> List[Path]:
    """Resolve many glob patterns relative to `root` and return unique sorted paths."""

    out: List[Path] = []
    for pat in patterns:
        if not pat:
            continue
        out.extend(sorted(root.glob(pat)))
    return sorted(set(out))


def _h5_read_dataset(h5f, dataset_path: str) -> np.ndarray:
    """Read one dataset path from an open H5 handle."""

    if dataset_path not in h5f:
        raise KeyError(f"Dataset path not found in H5: {dataset_path}")
    return np.asarray(h5f[dataset_path])


def _load_h5_map(h5_path: Path, key_map: Mapping[str, str]) -> Dict[str, np.ndarray]:
    """Load selected arrays from H5 using mapping `output_key -> h5_dataset_path`."""

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for H5->NPZ conversion") from exc

    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5f:
        for out_key, in_path in key_map.items():
            if not in_path:
                continue
            out[out_key] = _h5_read_dataset(h5f, in_path)
    return out


def _index_by_frame(files: Sequence[Path], regex: re.Pattern[str]) -> Dict[str, Path]:
    """Index files by parsed frame id for fast temporal pairing."""

    idx: Dict[str, Path] = {}
    for f in sorted(files):
        idx[parse_frame_id(f, regex)] = f
    return idx


def _pair_input_output(
    input_h5: Sequence[Path],
    output_h5: Sequence[Path],
    regex: re.Pattern[str],
    pair_mode: str,
) -> List[Tuple[str, str, Path, Path]]:
    """Create input/output frame pairs.

    Supported pair modes:
    - `intersection`: use frames present in both streams
    - `nearest_prev`: for each input frame, pick latest output frame <= input frame
    """

    in_map = _index_by_frame(input_h5, regex)
    out_map = _index_by_frame(output_h5, regex)

    in_frames = sorted(in_map.keys())
    out_frames = sorted(out_map.keys())

    if pair_mode == "intersection":
        common = sorted(set(in_frames).intersection(out_frames))
        return [(fr, fr, in_map[fr], out_map[fr]) for fr in common]

    if pair_mode == "nearest_prev":
        out_int = [int(f) for f in out_frames]
        pairs: List[Tuple[str, str, Path, Path]] = []
        for in_fr in in_frames:
            target = int(in_fr)
            j = bisect.bisect_right(out_int, target) - 1
            if j < 0:
                continue
            out_fr = out_frames[j]
            pairs.append((in_fr, out_fr, in_map[in_fr], out_map[out_fr]))
        return pairs

    raise ValueError(f"Unknown pair_mode={pair_mode}. Use 'intersection' or 'nearest_prev'.")


def _optional_for_pair(by_frame: Mapping[str, Path], in_frame: str, out_frame: str) -> Optional[Path]:
    """Resolve optional sidecar for a pair (prefer input-frame match, then output-frame)."""

    return by_frame.get(in_frame) or by_frame.get(out_frame)


def _merge_payloads(
    in_payload: Mapping[str, np.ndarray],
    out_payload: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Merge input/output payload dictionaries and guard against key collisions."""

    overlap = set(in_payload.keys()).intersection(out_payload.keys())
    if overlap:
        raise KeyError(
            "Input/output key maps overlap. Use unique output keys. "
            f"Overlapping keys: {sorted(overlap)}"
        )

    payload = dict(in_payload)
    payload.update(out_payload)
    return payload


def _as_obj_scalar(v: str) -> np.ndarray:
    """Store string metadata as numpy object scalar for robust NPZ round-trip."""

    return np.asarray(v, dtype=object)


def convert_one_dataset(
    ds_cfg: Mapping[str, Any],
    global_cfg: Mapping[str, Any],
    root: Path,
    out_root: Path,
) -> Dict[str, Any]:
    """Convert one dataset config entry into merged per-frame NPZ outputs."""

    name = str(ds_cfg.get("name", "dataset"))
    output_subdir = str(ds_cfg.get("output_subdir", name))
    out_dir = ensure_dir(out_root / output_subdir)

    frame_regex = re.compile(
        str(ds_cfg.get("frame_id_regex", global_cfg.get("frame_id_regex", DEFAULT_FRAME_ID_REGEX)))
    )
    pair_mode = str(ds_cfg.get("pair_mode", global_cfg.get("pair_mode", "intersection"))).lower()

    input_h5_glob = list(ds_cfg.get("input_h5_glob", []))
    output_h5_glob = list(ds_cfg.get("output_h5_glob", []))
    input_xmf_glob = list(ds_cfg.get("input_xmf_glob", []))
    output_xmf_glob = list(ds_cfg.get("output_xmf_glob", []))
    vtk_glob = list(ds_cfg.get("vtk_glob", []))

    if not input_h5_glob or not output_h5_glob:
        raise RuntimeError(f"Dataset `{name}` requires both input_h5_glob and output_h5_glob.")

    input_h5 = discover_many(input_h5_glob, root)
    output_h5 = discover_many(output_h5_glob, root)
    input_xmf = discover_many(input_xmf_glob, root)
    output_xmf = discover_many(output_xmf_glob, root)
    vtk_files = discover_many(vtk_glob, root)

    input_key_map = dict(ds_cfg.get("input_key_map", global_cfg.get("input_key_map", {})))
    output_key_map = dict(ds_cfg.get("output_key_map", global_cfg.get("output_key_map", {})))

    if not input_key_map and not output_key_map:
        raise RuntimeError(f"Dataset `{name}` has no input_key_map/output_key_map.")

    pairs = _pair_input_output(input_h5, output_h5, frame_regex, pair_mode=pair_mode)

    in_xmf_map = _index_by_frame(input_xmf, frame_regex)
    out_xmf_map = _index_by_frame(output_xmf, frame_regex)
    vtk_map = _index_by_frame(vtk_files, frame_regex)

    written = 0
    records: List[Dict[str, Any]] = []

    for in_fr, out_fr, in_h5, out_h5 in pairs:
        try:
            in_payload = _load_h5_map(in_h5, input_key_map) if input_key_map else {}
            out_payload = _load_h5_map(out_h5, output_key_map) if output_key_map else {}
            payload = _merge_payloads(in_payload, out_payload)
        except Exception as exc:
            print(f"[warn][{name}] skip in={in_h5.name} out={out_h5.name}: {exc}")
            continue

        in_xmf_path = _optional_for_pair(in_xmf_map, in_fr, out_fr)
        out_xmf_path = _optional_for_pair(out_xmf_map, in_fr, out_fr)
        vtk_path = _optional_for_pair(vtk_map, in_fr, out_fr)

        payload["source_dataset"] = _as_obj_scalar(name)
        payload["source_input_h5_path"] = _as_obj_scalar(str(in_h5))
        payload["source_output_h5_path"] = _as_obj_scalar(str(out_h5))
        payload["source_input_xmf_path"] = _as_obj_scalar("" if in_xmf_path is None else str(in_xmf_path))
        payload["source_output_xmf_path"] = _as_obj_scalar("" if out_xmf_path is None else str(out_xmf_path))
        payload["source_vtk_path"] = _as_obj_scalar("" if vtk_path is None else str(vtk_path))
        payload["input_frame_id"] = _as_obj_scalar(in_fr)
        payload["output_frame_id"] = _as_obj_scalar(out_fr)

        fname = (
            f"{name}__frame_{in_fr}.npz"
            if in_fr == out_fr
            else f"{name}__in_{in_fr}__out_{out_fr}.npz"
        )
        out_path = out_dir / fname
        np.savez_compressed(out_path, **payload)
        written += 1

        records.append(
            {
                "dataset": name,
                "input_frame": in_fr,
                "output_frame": out_fr,
                "input_h5": str(in_h5),
                "output_h5": str(out_h5),
                "input_xmf": "" if in_xmf_path is None else str(in_xmf_path),
                "output_xmf": "" if out_xmf_path is None else str(out_xmf_path),
                "vtk": "" if vtk_path is None else str(vtk_path),
                "npz_path": str(out_path),
                "keys": sorted(payload.keys()),
            }
        )

    summary = {
        "dataset": name,
        "pair_mode": pair_mode,
        "n_input_h5": len(input_h5),
        "n_output_h5": len(output_h5),
        "n_pairs": len(pairs),
        "n_written": written,
        "n_input_xmf": len(input_xmf),
        "n_output_xmf": len(output_xmf),
        "n_vtk": len(vtk_files),
        "output_dir": str(out_dir),
    }

    save_json(out_dir / "conversion_summary.json", summary)
    save_json(out_dir / "conversion_index.json", {"frames": records})
    return summary


def convert_h5_vtk_to_npz(cfg: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    """Convert all configured datasets from split H5/XMF/VTK to merged NPZ outputs."""

    out_root = ensure_dir(root / cfg.get("output_dir", "final/output/merged_npz"))

    datasets = cfg.get("datasets", None)
    if not datasets:
        raise RuntimeError(
            "Expected `datasets:` list in conversion config. "
            "Use final/configs/h5_vtk_to_npz_template.yaml format."
        )

    ds_summaries = []
    for ds_cfg in datasets:
        ds_summaries.append(convert_one_dataset(ds_cfg, cfg, root=root, out_root=out_root))

    summary = {
        "mode": "multi_dataset_split_io",
        "n_datasets": len(ds_summaries),
        "datasets": ds_summaries,
        "output_root": str(out_root),
    }
    save_json(out_root / "conversion_summary.json", summary)
    return summary


def main() -> None:
    """CLI wrapper for H5/XMF/VTK to NPZ conversion."""

    parser = argparse.ArgumentParser(
        description="Convert split input/output H5 (+ optional XMF/VTK) into unified NPZ frames"
    )
    parser.add_argument("--config", type=str, default="final/configs/h5_vtk_to_npz_template.yaml")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = convert_h5_vtk_to_npz(cfg, root=Path(args.root))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
