import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader
from gino.data.dataset import EvolutionDataset, load_processed_dataset, collate_one
from gino.data.normalization import NormalizationStats
from gino.evaluation.one_step import evaluate_one_step
from gino.data.reconstruction import rebuild_next_batch
from gino.data.hdf5 import find_task2_file, read_task2_field, native_field_plane
from gino.evaluation.metrics import mse, relative_l2, state_component_metrics, normalized_state_component_metrics
from gino.dynamics.rollout import inference_rollout
from gino.dynamics.state_transition import predict_next_state
from gino.model.gino import GINOSharedLatent, build_latent_grid
from gino.utils import load_config, write_json
from visualization.fields import plot_field_comparison
from visualization.temporal import plot_temporal_errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GINO checkpoint.")
    parser.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    checkpoint_path = Path(cfg["evaluation"].get("checkpoint", "")).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (HERE / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Set evaluation.checkpoint to an existing checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    data_path = (HERE / cfg["data"]["dataset"]).resolve()
    data = load_processed_dataset(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", cfg)["model"] if "model" in checkpoint.get("config", {}) else checkpoint.get("config", cfg)
    features = checkpoint.get("feature_names", cfg["data"]["input_features"])
    global_names = checkpoint.get("global_condition_channels", cfg["data"]["global_condition_channels"])
    field_names = checkpoint.get("field_target_names", data["field_target_names"])
    target_names = checkpoint.get("target_names", data["target_names"])
    state = checkpoint["model_state_dict"]
    model = GINOSharedLatent(len(features), len(target_names), len(field_names), len(global_names), model_cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatibility. Missing={missing}; unexpected={unexpected}")
    stats = NormalizationStats.from_dataset(data, [list(data["feature_names"]).index(x) for x in features])
    if "input_mean" in checkpoint:
        stats.input_mean = np.asarray(checkpoint["input_mean"]).reshape(-1)
        stats.input_std = np.asarray(checkpoint["input_std"]).reshape(-1)
        stats.residual_mean = np.asarray(checkpoint.get("target_mean", checkpoint.get("residual_mean"))).reshape(-1)
        stats.residual_std = np.asarray(checkpoint.get("target_std", checkpoint.get("residual_std"))).reshape(-1)
        stats.field_mean = np.asarray(checkpoint["field_mean"]).reshape(-1)
        stats.field_std = np.asarray(checkpoint["field_std"]).reshape(-1)
        stats.coord_min = np.asarray(checkpoint["coord_min"]).reshape(3)
        stats.coord_span = np.asarray(checkpoint["coord_span"]).reshape(3)
        if "state_mean" in checkpoint:
            stats.state_mean = np.asarray(checkpoint["state_mean"]).reshape(-1)[:7]
            stats.state_std = np.asarray(checkpoint["state_std"]).reshape(-1)[:7]
    split = cfg["evaluation"]["split"]
    pair_ids = data[f"{split}_pair_ids"]
    rollout_horizon = max(int(h) for h in cfg["evaluation"]["rollout_horizons"])
    ds = EvolutionDataset(data, pair_ids, features, global_names, cfg["data"]["max_particles"], cfg["data"]["max_queries"], rollout_horizon)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_one)
    max_batches = int(cfg["evaluation"].get("max_batches", 0))
    if max_batches:
        from itertools import islice
        loader = list(islice(loader, max_batches))
    latent = build_latent_grid(model_cfg["latent_res"], device)
    records = evaluate_one_step(model, loader, latent, stats, device)
    cases = sorted({record.get("case", "unknown") for record in records})
    for case in cases:
        case_records = [record for record in records if record.get("case", "unknown") == case]
        print(f"one-step {case}: {len(case_records)} pairs")
        for key in ("position_rmse", "circulation_rmse", "sigma_rmse", "field_mse", "field_relative_l2"):
            values = [float(record[key]) for record in case_records if key in record]
            if values:
                print(f"  {key}: {float(np.mean(values)):.6g}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rollout_records = []
    max_horizon = max(int(h) for h in cfg["evaluation"]["rollout_horizons"])
    rollout_ds = EvolutionDataset(data, pair_ids, features, global_names,
                                 cfg["data"]["max_particles"], cfg["data"]["max_queries"], rollout_horizon)
    rollout_loader = DataLoader(rollout_ds, batch_size=1, shuffle=False, collate_fn=collate_one)
    if max_batches:
        from itertools import islice
        rollout_loader = islice(rollout_loader, max_batches)
    model.eval()
    with torch.inference_mode():
        for batch in rollout_loader:
            targets = torch.cat((batch["rollout_initial_target"].unsqueeze(1), batch["rollout_targets"]), dim=1)
            steps = min(max_horizon, int(targets.shape[1]))
            if steps <= 0:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            targets = targets[:, :steps].to(device)
            rebuild = lambda old, state, field: rebuild_next_batch(old, state, field, stats)
            pred_states, pred_fields = inference_rollout(model, batch, latent, stats, steps, rebuild, store_on_cpu=True)
            context = batch["pair_context"]
            for horizon in cfg["evaluation"]["rollout_horizons"]:
                h = int(horizon)
                if h <= steps:
                    truth = targets[:, h - 1].cpu().numpy()
                    prediction = pred_states[:, h - 1, :truth.shape[1]].numpy()
                    field_target_norm = batch["field_target"] if h == 1 else batch["rollout_field_target"][:, h - 2]
                    field_target = stats.denormalize_field(field_target_norm).cpu().numpy()
                    field_prediction = pred_fields[:, h - 1].numpy()
                    rollout_records.append({"pair_id": int(batch["pair_id"][0]),
                        "case": context.get("case", "unknown"), "horizon": h,
                        "phase": float(batch.get("phase_next", 0.0)) + (h - 1) * float(batch.get("phase_delta", 0.0)),
                        "field_phase": float(context.get("phase_t", 0.0)) + (h - 1) * float(batch.get("phase_delta", 0.0)),
                        **state_component_metrics(prediction, truth),
                        **normalized_state_component_metrics(prediction, truth,
                                                             stats.state_mean, stats.state_std),
                        "field_mse": mse(field_prediction, field_target),
                        "field_relative_l2": relative_l2(field_prediction, field_target)})
            del pred_states, pred_fields, batch, targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for case in sorted({record.get("case", "unknown") for record in rollout_records}):
        case_records = [record for record in rollout_records if record.get("case", "unknown") == case]
        for horizon in sorted({int(record["horizon"]) for record in case_records}):
            rows = [record for record in case_records if int(record["horizon"]) == horizon]
            print(f"rollout {case} H={horizon}: {len(rows)} pairs | "
                  f"position RMSE={np.mean([row['position_rmse'] for row in rows]):.6g} | "
                  f"circulation RMSE={np.mean([row['circulation_rmse'] for row in rows]):.6g} | "
                  f"sigma RMSE={np.mean([row['sigma_rmse'] for row in rows]):.6g} | "
                  f"field MSE={np.mean([row['field_mse'] for row in rows]):.6g} | "
                  f"field relL2={np.mean([row['field_relative_l2'] for row in rows]):.6g}")
    native_records = []
    native_visual_candidates = {}
    native_root_value = str(cfg["data"].get("native_task2_root", "")).strip()
    native_root = str((HERE / native_root_value).resolve()) if native_root_value and not Path(native_root_value).is_absolute() else native_root_value
    if native_root:
        native_ds = EvolutionDataset(data, pair_ids, features, global_names,
                                     cfg["data"]["max_particles"], cfg["data"]["max_queries"], rollout_horizon)
        native_loader = DataLoader(native_ds, batch_size=1, shuffle=False, collate_fn=collate_one)
        if max_batches:
            from itertools import islice
            native_loader = islice(native_loader, max_batches)
        for batch in native_loader:
            context = batch["pair_context"]
            path = find_task2_file(native_root, context.get("case", ""), context.get("frame_t", ""))
            if path is None:
                continue
            native_xyz, native_truth = read_task2_field(path)
            take = np.arange(len(native_xyz))
            if len(take) > int(cfg["data"]["max_queries"]):
                take = take[np.linspace(0, len(take) - 1, int(cfg["data"]["max_queries"]), dtype=np.int64)]
            xyz, truth = native_xyz[take], native_truth[take]
            native_batch = dict(batch)
            native_batch["output_queries"] = stats.normalize_positions(torch.as_tensor(xyz, dtype=torch.float32).unsqueeze(0).to(device)).clamp(0.0, 1.0)
            native_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in native_batch.items()}
            with torch.inference_mode():
                result = predict_next_state(model, native_batch, latent, stats)
                predicted = result["field_phys"][0].cpu().numpy()
            native_records.append({"pair_id": int(batch["pair_id"][0]), "case": context.get("case", "unknown"),
                "frame": context.get("frame_t", ""), "phase": float(context.get("phase_t", 0.0)), "hdf5": str(path),
                "field_mse": mse(predicted, truth), "field_relative_l2": relative_l2(predicted, truth),
                "n_points": int(len(truth))})
            x_axis, z_axis, true_slice, plane_xyz, _ = native_field_plane(native_xyz, native_truth,
                                                                           y_plane=float(cfg["evaluation"].get("field_y_plane", 0.0)))
            plane_batch = dict(batch)
            plane_batch["output_queries"] = stats.normalize_positions(
                torch.as_tensor(plane_xyz, dtype=torch.float32).unsqueeze(0).to(device)).clamp(0.0, 1.0)
            plane_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in plane_batch.items()}
            with torch.inference_mode():
                plane_result = predict_next_state(model, plane_batch, latent, stats)
                predicted_slice = plane_result["field_phys"][0, :, 0].cpu().numpy().reshape(true_slice.shape)
            case_name = context.get("case", "unknown")
            native_visual_candidates.setdefault(case_name, []).append({
                "phase": float(context.get("phase_t", 0.0)), "x": x_axis, "z": z_axis,
                "true": true_slice, "pred": predicted_slice})
            del native_batch, plane_batch, result, plane_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for case in sorted({record.get("case", "unknown") for record in native_records}):
            rows = [record for record in native_records if record.get("case", "unknown") == case]
            print(f"native Task-2 {case}: {len(rows)} frames | "
                  f"field MSE={np.mean([row['field_mse'] for row in rows]):.6g} | "
                  f"field relL2={np.mean([row['field_relative_l2'] for row in rows]):.6g}")
    out_dir = checkpoint_path.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "one_step.json", {"checkpoint": str(checkpoint_path), "dataset": str(data_path),
        "architecture": model_cfg, "input_features": features, "target_features": target_names,
        "normalization_source": checkpoint.get("dataset_path", str(data_path)), "rollout_horizons": cfg["evaluation"]["rollout_horizons"], "records": records})
    write_json(out_dir / "autoregressive.json", {"checkpoint": str(checkpoint_path), "records": rollout_records})
    write_json(out_dir / "native_task2_field.json", {"root": native_root, "records": native_records})
    if cfg["evaluation"].get("save_plots", True) and native_visual_candidates:
        requested_phases = (0.3, 0.5, 0.7)
        for case_name, candidates in native_visual_candidates.items():
            if len(candidates) < 3:
                continue
            remaining = list(candidates)
            selected = []
            for phase in requested_phases:
                index = min(range(len(remaining)), key=lambda i: abs(remaining[i]["phase"] - phase))
                selected.append(remaining.pop(index))
            x_axis, z_axis = selected[0]["x"], selected[0]["z"]
            plot_field_comparison(x_axis, z_axis,
                np.stack([item["true"] for item in selected]),
                np.stack([item["pred"] for item in selected]),
                out_dir / f"native_field_{case_name.replace('/', '_')}.png",
                phases=[item["phase"] for item in selected], component_label="u_x")
    if cfg["evaluation"].get("save_plots", True):
        plot_temporal_errors(records, rollout_records, out_dir)
    if records:
        with open(out_dir / "one_step.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    if rollout_records:
        with open(out_dir / "autoregressive.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rollout_records[0].keys()); writer.writeheader(); writer.writerows(rollout_records)
    if native_records:
        with open(out_dir / "native_task2_field.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=native_records[0].keys()); writer.writeheader(); writer.writerows(native_records)
    print(f"Saved {len(records)} one-step and {len(rollout_records)} rollout records to {out_dir}")


if __name__ == "__main__":
    main()
