#!/usr/bin/env python3
"""
Benchmark Task-1 FLOWUnsteady runtime with FMM vs ML U/gradU surrogate and
measure vortex-particle state error over simulation time.

Typical usage from the repository root:

  python3 /home/dysco/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/benchmark_comparison_task1.py --run

Or analyze existing runtime snapshots:

  python3 /home/dysco/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/benchmark_comparison_task1.py \
    --baseline-dir output/runtime_baseline_fmm \
    --surrogate-dir output/runtime_surrogate_ml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
REPO_DIR = FINAL_DIR.parent
FLOWUNSTEADY_PROJECT = Path(os.environ.get("FLOWUNSTEADY_PROJECT", "/home/dysco/FLOWUnsteady")).expanduser()
if not FLOWUNSTEADY_PROJECT.is_dir():
    FLOWUNSTEADY_PROJECT = REPO_DIR


def _default_model() -> Path:
    candidates = [
        THIS_DIR / "result" / "task1" / "good result" / "final" / "best_task1_v3_model.pt",
        FINAL_DIR / "result" / "task1" / "task1_particle_ugradu_gino_pointwise_best_model.pt",
        FINAL_DIR / "result" / "task1_particle_ugradu_gino_pointwise_best_model.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> dict[str, Any]:
    print("Running:", " ".join(cmd), flush=True)
    start = time.perf_counter()
    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            log.write("Running: " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "log_path": None if log_path is None else str(log_path),
    }


def resolve_from_workflow(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    workflow_path = THIS_DIR / path
    if workflow_path.exists() or str(path).startswith("output"):
        return workflow_path.resolve()
    return path.resolve()


def run_flowunsteady_cases(args: argparse.Namespace) -> dict[str, Any]:
    runner = THIS_DIR / "run_task1_flowunsteady.jl"
    sim_file = resolve_from_workflow(args.sim_file)
    model = resolve_from_workflow(args.model)
    meta = resolve_from_workflow(args.meta)
    project = str(Path(args.project).expanduser().resolve()) if args.project else "."

    results: dict[str, Any] = {}
    common = [
        args.julia_bin,
        f"--project={project}",
        str(runner),
        "--sim-file",
        str(sim_file),
        "--model",
        str(model),
        "--meta",
        str(meta),
        "--device",
        args.device,
    ]

    baseline_dir = resolve_from_workflow(args.baseline_dir)
    surrogate_dir = resolve_from_workflow(args.surrogate_dir)
    out_dir = resolve_from_workflow(args.out_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results["baseline_fmm"] = _run_command(
        common + ["--mode", "baseline_fmm", "--save-dir", str(baseline_dir)],
        REPO_DIR,
        out_dir / "baseline_fmm.log",
    )
    if results["baseline_fmm"]["returncode"] != 0:
        raise SystemExit(f"Baseline FMM run failed; see {out_dir / 'baseline_fmm.log'}")
    _require_snapshots(baseline_dir, "baseline")

    results["surrogate_ml"] = _run_command(
        common + ["--mode", "surrogate_ml", "--save-dir", str(surrogate_dir)],
        REPO_DIR,
        out_dir / "surrogate_ml.log",
    )
    if results["surrogate_ml"]["returncode"] != 0:
        raise SystemExit(f"Surrogate ML run failed; see {out_dir / 'surrogate_ml.log'}")
    _require_snapshots(surrogate_dir, "surrogate")

    return results


def _require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Reading JLD2 snapshots from Python needs h5py. Install it or run the "
            "Julia pipeline with an NPZ export path."
        ) from exc
    return h5py


def _jld2_value(obj: Any) -> np.ndarray:
    if hasattr(obj, "shape"):
        return np.asarray(obj)
    if "data" in obj:
        return np.asarray(obj["data"])
    datasets = []
    obj.visititems(lambda _name, item: datasets.append(item) if hasattr(item, "shape") else None)
    if not datasets:
        raise KeyError("No dataset found inside JLD2 group")
    return np.asarray(datasets[0])


def read_snapshot(path: Path) -> dict[str, np.ndarray | float | int]:
    if path.suffix.lower() == ".csv":
        return read_csv_snapshot(path)

    h5py = _require_h5py()
    out: dict[str, np.ndarray | float | int] = {}
    with h5py.File(path, "r") as f:
        for name in ("X", "Gamma", "sigma", "U_ref", "U_ml", "t", "dt", "step"):
            if name in f:
                value = _jld2_value(f[name])
                if value.shape == ():
                    out[name] = value.item()
                else:
                    out[name] = np.asarray(value)
    for required in ("X", "Gamma", "sigma"):
        if required not in out:
            raise KeyError(f"{path} does not contain {required}; available keys may not match the JLD2 reader")
    return out


def read_csv_snapshot(path: Path) -> dict[str, np.ndarray | float | int]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = data.reshape(1)

    names = data.dtype.names or ()
    required = ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
    missing = [name for name in required if name not in names]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")

    out: dict[str, np.ndarray | float | int] = {
        "X": np.column_stack([data["x"], data["y"], data["z"]]),
        "Gamma": np.column_stack([data["Gamma_x"], data["Gamma_y"], data["Gamma_z"]]),
        "sigma": np.asarray(data["sigma"]).reshape(-1),
    }
    for scalar in ("t", "dt", "step"):
        if scalar in names:
            value = np.asarray(data[scalar]).reshape(-1)[0]
            out[scalar] = int(value) if scalar == "step" else float(value)
    for prefix in ("U_ref", "U_ml"):
        cols = [f"{prefix}_{axis}" for axis in ("x", "y", "z")]
        if all(col in names for col in cols):
            out[prefix] = np.column_stack([data[col] for col in cols])
    return out


def _snapshot_map(directory: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for suffix in ("csv", "jld2"):
        for path in sorted(directory.glob(f"{prefix}_*.{suffix}")):
            try:
                step = int(path.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            out[step] = path
    return out


def _require_snapshots(directory: Path, prefix: str) -> None:
    snapshots = _snapshot_map(directory, prefix)
    if not snapshots:
        raise SystemExit(
            f"Run completed but no {prefix}_*.csv snapshots were written in {directory}. "
            f"Check the run log and the Task1 runtime callback prints."
        )
    print(
        f"Found {len(snapshots)} {prefix} snapshots in {directory} "
        f"(steps {min(snapshots)}..{max(snapshots)})",
        flush=True,
    )


def _as_rows(arr: np.ndarray, width: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if width is not None and a.ndim == 2 and a.shape[0] == width and a.shape[1] != width:
        a = a.T
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _dist, idx = cKDTree(target).query(source, k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        idx = np.empty(source.shape[0], dtype=np.int64)
        chunk = 2048
        for start in range(0, source.shape[0], chunk):
            stop = min(start + chunk, source.shape[0])
            diff = source[start:stop, None, :] - target[None, :, :]
            idx[start:stop] = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return idx


def _metric(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(diff, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    l2 = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(ref))
    return {
        "rmse": float(math.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
        "mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "rel_l2": l2 / max(denom, 1.0e-12),
    }


def compare_histories(baseline_dir: Path, surrogate_dir: Path) -> list[dict[str, float | int]]:
    baseline = _snapshot_map(baseline_dir, "baseline")
    surrogate = _snapshot_map(surrogate_dir, "surrogate")
    common_steps = sorted(set(baseline) & set(surrogate))
    if not common_steps:
        raise SystemExit(
            f"No matching baseline_*.csv/.jld2 and surrogate_*.csv/.jld2 snapshots found in "
            f"{baseline_dir} and {surrogate_dir}."
        )

    rows: list[dict[str, float | int]] = []
    for step in common_steps:
        b = read_snapshot(baseline[step])
        s = read_snapshot(surrogate[step])

        xb = _as_rows(np.asarray(b["X"]), 3)
        xs = _as_rows(np.asarray(s["X"]), 3)
        gb = _as_rows(np.asarray(b["Gamma"]), 3)
        gs = _as_rows(np.asarray(s["Gamma"]), 3)
        sb = _as_rows(np.asarray(b["sigma"]))
        ss = _as_rows(np.asarray(s["sigma"]))

        if xb.shape[0] == xs.shape[0]:
            match = np.arange(xb.shape[0])
        else:
            match = _nearest_indices(xb, xs)

        pos = _metric(xs[match] - xb, xb)
        gam = _metric(gs[match] - gb, gb)
        sig = _metric(ss[match] - sb, sb)
        row: dict[str, float | int] = {
            "step": step,
            "time": float(b.get("t", step)),
            "baseline_particles": int(xb.shape[0]),
            "surrogate_particles": int(xs.shape[0]),
            "particle_count_delta": int(xs.shape[0] - xb.shape[0]),
            "position_rmse": pos["rmse"],
            "position_rel_l2": pos["rel_l2"],
            "position_max_abs": pos["max_abs"],
            "gamma_rmse": gam["rmse"],
            "gamma_rel_l2": gam["rel_l2"],
            "gamma_max_abs": gam["max_abs"],
            "sigma_rmse": sig["rmse"],
            "sigma_rel_l2": sig["rel_l2"],
            "sigma_max_abs": sig["max_abs"],
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int]], timings: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_common_steps": len(rows)}
    if timings:
        summary["timings"] = timings
        b = timings.get("baseline_fmm", {}).get("wall_seconds")
        s = timings.get("surrogate_ml", {}).get("wall_seconds")
        if b and s:
            summary["surrogate_speedup_vs_fmm"] = float(b) / max(float(s), 1.0e-12)
    for key in ("position_rel_l2", "gamma_rel_l2", "sigma_rel_l2"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        summary[f"{key}_final"] = float(vals[-1])
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_max"] = float(vals.max())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run baseline FMM and surrogate ML before analysis")
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument("--project", default=str(FLOWUNSTEADY_PROJECT), help="Julia project directory")
    parser.add_argument("--sim-file", default=str(THIS_DIR / "simulation.jl"))
    parser.add_argument("--model", default=str(_default_model()))
    parser.add_argument("--meta", default=str(FINAL_DIR / "processed_data_task1" / "particle_ugradu_dataset.npz"))
    parser.add_argument(
        "--device",
        default=os.environ.get("TASK1_DEVICE", "auto"),
        help="PyTorch device for the ML predictor: auto | cpu | cuda | cuda:<id>",
    )
    parser.add_argument("--baseline-dir", default=str(THIS_DIR / "output" / "runtime_baseline_fmm"))
    parser.add_argument("--surrogate-dir", default=str(THIS_DIR / "output" / "runtime_surrogate_ml"))
    parser.add_argument("--out-dir", default=str(THIS_DIR / "output" / "benchmark_task1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_from_workflow(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status_path = out_dir / "pipeline_status.json"
    status: dict[str, Any] = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with status_path.open("w") as f:
        json.dump(status, f, indent=2)

    try:
        timings = run_flowunsteady_cases(args) if args.run else None
        rows = compare_histories(resolve_from_workflow(args.baseline_dir), resolve_from_workflow(args.surrogate_dir))
        summary = summarize(rows, timings)

        write_csv(out_dir / "particle_state_error_timeseries.csv", rows)
        with (out_dir / "benchmark_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        status.update(
            {
                "status": "complete",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "summary_path": str(out_dir / "benchmark_summary.json"),
                "timeseries_csv": str(out_dir / "particle_state_error_timeseries.csv"),
            }
        )
        with status_path.open("w") as f:
            json.dump(status, f, indent=2)

        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_dir / 'particle_state_error_timeseries.csv'}")
        print(f"Wrote {out_dir / 'benchmark_summary.json'}")
    except Exception as exc:
        status.update({"status": "failed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "error": str(exc)})
        with status_path.open("w") as f:
            json.dump(status, f, indent=2)
        raise


if __name__ == "__main__":
    main()
