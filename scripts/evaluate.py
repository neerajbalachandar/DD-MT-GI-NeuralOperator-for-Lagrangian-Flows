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
from gino.data.reconstruction import rebuild_next_batch_inference
from gino.data.hdf5 import read_task2_velocity, native_field_plane, highest_energy_y_plane
from gino.data.geometry import estimate_chordwise_bounds
from gino.evaluation.metrics import mse, relative_l2, state_component_metrics, normalized_state_component_metrics
from gino.dynamics.rollout import inference_rollout
from gino.dynamics.state_transition import predict_next_state
from gino.model.gino import GINOSharedLatent, build_latent_grid
from gino.utils import load_config, write_json
from visualization.fields import (plot_native_ux_comparison, plot_normalized_ux_profiles,
                                  plot_velocity_magnitude_summary)
from visualization.temporal import (plot_temporal_errors, plot_rollout_snapshots,
                                    plot_rmse_vs_phase_grid)
from visualization.evaluation import plot_state_parity, plot_field_error_maps


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GINO checkpoint.")
    parser.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    configured_checkpoint = str(cfg["evaluation"].get("checkpoint", "")).strip()
    if configured_checkpoint:
        checkpoint_path = Path(configured_checkpoint).expanduser()
    else:
        checkpoint_path = Path(cfg["training"]["run_dir"]) / "best_model.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = (HERE / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Set evaluation.checkpoint to an existing checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    data_path = (HERE / cfg["data"]["dataset"]).resolve()
    data = load_processed_dataset(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_cfg = checkpoint.get("config", {})
    model_cfg = checkpoint_cfg.get("model", checkpoint_cfg or cfg)
    architecture_keys = ("latent_res", "hidden_channels", "fno_layers", "fno_modes", "gno_radius",
                         "mlp_layers", "mlp_hidden", "query_pos_encoding_frequencies",
                         "use_attention", "use_skip", "use_global_conditioning", "use_task_adapters")
    mismatches = {key: (model_cfg.get(key), cfg["model"].get(key)) for key in architecture_keys
                  if key in model_cfg and key in cfg["model"] and model_cfg[key] != cfg["model"][key]}
    if mismatches:
        print(f"Checkpoint/config architecture mismatch: {mismatches}")
    features = checkpoint.get("feature_names", cfg["data"]["input_features"])
    global_names = checkpoint.get("global_condition_channels", cfg["data"]["global_condition_channels"])
    field_names = checkpoint.get("field_target_names", data["field_target_names"])
    target_names = checkpoint.get("target_names", data["target_names"])
    state = {key: value for key, value in checkpoint["model_state_dict"].items() if key != "_metadata"}
    model = GINOSharedLatent(len(features), len(target_names), len(field_names), len(global_names), model_cfg).to(device)
    checkpoint_parameter_count = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    model_parameter_count = sum(value.numel() for value in model.state_dict().values() if torch.is_tensor(value))
    print(f"Checkpoint parameters: {checkpoint_parameter_count}; configured model parameters: {model_parameter_count}")
    model_state = {key: value for key, value in model.state_dict().items() if key != "_metadata"}
    model_keys, checkpoint_keys = set(model_state), set(state)
    missing_keys, unexpected_keys = sorted(model_keys - checkpoint_keys), sorted(checkpoint_keys - model_keys)
    shape_mismatches = {key: (tuple(state[key].shape), tuple(model.state_dict()[key].shape))
                        for key in model_keys & checkpoint_keys
                        if torch.is_tensor(state[key]) and state[key].shape != model_state[key].shape}
    print(f"Checkpoint missing keys: {missing_keys}")
    print(f"Checkpoint unexpected keys: {unexpected_keys}")
    print(f"Checkpoint tensor shape mismatches: {shape_mismatches}")
    if checkpoint_parameter_count != model_parameter_count:
        print("Checkpoint parameter-count mismatch")
    if not shape_mismatches:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            missing_keys, unexpected_keys = missing, unexpected
    if checkpoint_parameter_count != model_parameter_count or missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(f"Checkpoint incompatibility: parameter_count="
                           f"{checkpoint_parameter_count}/{model_parameter_count}, missing={missing_keys}, "
                           f"unexpected={unexpected_keys}, shapes={shape_mismatches}")
    print("Checkpoint architecture/configuration: compatible")
    stats = NormalizationStats.from_checkpoint(checkpoint)
    split = str(cfg["evaluation"].get("split", "test"))
    split_key = f"{split}_pair_ids"
    if split_key not in data:
        available = sorted(key.removesuffix("_pair_ids") for key in data if key.endswith("_pair_ids"))
        raise ValueError(f"Evaluation split {split!r} is unavailable in {data_path}; available splits: {available}")
    pair_ids = np.asarray(data[split_key], dtype=np.int64)
    if pair_ids.size == 0:
        raise ValueError(f"Evaluation split {split!r} is empty in {data_path}.")
    requested_horizons = [int(h) for h in cfg["evaluation"].get("rollout_horizons", [1])]
    if not requested_horizons or any(h < 1 for h in requested_horizons):
        raise ValueError("evaluation.rollout_horizons must be a non-empty list of positive integers.")
    rollout_horizon = max(requested_horizons)
    residual_channels = len(target_names)
    ds = EvolutionDataset(data, pair_ids, features, global_names, cfg["data"]["max_particles"], cfg["data"]["max_queries"], 1, stats, residual_channels)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_one)
    max_batches = int(cfg["evaluation"].get("max_batches", 0))
    if max_batches < 0:
        raise ValueError("evaluation.max_batches must be 0 (all batches) or a positive integer.")
    if max_batches:
        from itertools import islice
        loader = list(islice(loader, max_batches))
    latent = build_latent_grid(model_cfg["latent_res"], device)
    one_step_visuals = {}
    records = evaluate_one_step(model, loader, latent, stats, device, visual_samples=one_step_visuals)
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
    rollout_visuals = {}
    chain_diagnostic_ds = EvolutionDataset(data, pair_ids, features, global_names,
                                          cfg["data"]["max_particles"], cfg["data"]["max_queries"],
                                          rollout_horizon, stats, residual_channels)
    chain_lengths = {int(pair_id): chain_diagnostic_ds.chain_length(int(pair_id)) for pair_id in pair_ids}
    available_steps = max(chain_lengths.values(), default=0)
    if available_steps < 1:
        raise RuntimeError("Evaluation data has no available rollout steps.")
    horizons = sorted({min(h, available_steps) for h in requested_horizons})
    if horizons != sorted(set(requested_horizons)):
        print(f"[rollout] available chain supports {available_steps} steps; capping requested horizons "
              f"{sorted(set(requested_horizons))} to {horizons}")
    max_horizon = max(horizons)
    rollout_horizon = max_horizon
    for horizon in horizons:
        unavailable_pairs = sum(length < horizon for length in chain_lengths.values())
        if unavailable_pairs:
            print(f"[rollout] H={horizon} unavailable for {unavailable_pairs} split pairs: "
                  "their chain is shorter than the requested horizon")
    rollout_pair_ids = (np.asarray([pair_id for pair_id in pair_ids
                                    if chain_lengths[int(pair_id)] >= 2], dtype=np.int64)
                        if max_horizon > 1 else pair_ids)
    if max_horizon > 1 and not len(rollout_pair_ids):
        raise RuntimeError("No evaluation pair has a chainable future; inspect chain_length and "
                           "re-run scripts/preprocess.py after the identity-tracking fix.")
    rollout_ds = EvolutionDataset(data, rollout_pair_ids, features, global_names,
                                 cfg["data"]["max_particles"], cfg["data"]["max_queries"], rollout_horizon,
                                 stats, residual_channels, require_verified_correspondence=max_horizon > 1)
    # Starts are filtered for chainability, but their futures may pass through
    # terminal pairs that are not themselves valid rollout starts.
    rollout_ds.allowed_pair_ids = set(pair_ids.tolist())
    rollout_loader = DataLoader(rollout_ds, batch_size=1, shuffle=False, collate_fn=collate_one)
    rollout_total = min(len(rollout_ds), max_batches) if max_batches else len(rollout_ds)
    if max_batches:
        from itertools import islice
        rollout_loader = islice(rollout_loader, max_batches)
    model.eval()
    skipped_horizons = {horizon: 0 for horizon in horizons}
    with torch.inference_mode():
        for batch_index, batch in enumerate(rollout_loader, start=1):
            targets = batch["rollout_state_targets"]
            avail = targets.shape[1]
            if avail < min(horizons):
                pair_id = int(batch["pair_id"][0])
                raise RuntimeError(f"Pair {pair_id} provides {avail} rollout steps, fewer than the minimum "
                                   f"requested horizon {min(horizons)}. Inspect EvolutionDataset.chain_length.")
            steps = min(max_horizon, int(targets.shape[1]))
            if steps <= 0:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            targets = targets[:, :steps].to(device)
            rebuild = lambda old, state, field: rebuild_next_batch_inference(old, state, field, stats)
            pred_states, pred_fields = inference_rollout(model, batch, latent, stats, steps, rebuild, store_on_cpu=True)
            context = batch["pair_context"]
            for horizon in horizons:
                h = int(horizon)
                if h <= steps:
                    truth = targets[:, h - 1].cpu().numpy()
                    prediction = pred_states[:, h - 1, :truth.shape[1]].numpy()
                    field_context = context if h == 1 else batch["rollout_contexts"][h - 2]
                    state_frame = ((context.get("frame_tp1", ""),) +
                                   tuple(c.get("frame_tp1", "") for c in batch["rollout_contexts"]))[h - 1]
                    field_available = bool(batch["rollout_field_available"][0, h - 1])
                    if field_available:
                        field_target = stats.denormalize_field(batch["rollout_field_targets"][:, h - 1]).cpu().numpy()
                        field_prediction = pred_fields[:, h - 1].numpy()
                    case_name = context.get("case", "unknown")
                    visual_key = (case_name, h)
                    visual = rollout_visuals.setdefault(visual_key, {
                            "truth": truth[0],
                            "prediction": prediction[0],
                            "frame": state_frame,
                            "physical_time": field_context.get("physical_time_tp1"),
                            "phase": float(batch["rollout_phases"][0, h - 1]),
                        })
                    if field_available and "field_truth" not in visual:
                        if h == 1:
                            field_coords = batch["query_xyz_phys"][0].cpu().numpy()
                        else:
                            field_coords = batch["rollout_query_xyz_phys"][0, h - 2].cpu().numpy()
                        visual.update({"coords": field_coords, "field_truth": field_target[0],
                                       "field_prediction": field_prediction[0], "field_frame": state_frame})
                    record = {"pair_id": int(batch["pair_id"][0]),
                              "case": case_name, "horizon": h,
                              "phase": float(batch["rollout_phases"][0, h - 1]),
                              "field_phase": float(field_context.get("phase_t", 0.0)),
                              "physical_time": field_context.get("physical_time_tp1"),
                              "field_physical_time": field_context.get("task2_physical_time"),
                              "frame": state_frame, "field_frame": field_context.get("frame_t", ""),
                              **state_component_metrics(prediction, truth),
                              **normalized_state_component_metrics(prediction, truth,
                                                                   stats.state_mean, stats.state_std)}
                    if field_available:
                        record.update({"field_mse": mse(field_prediction[..., :3], field_target[..., :3]),
                                       "field_relative_l2": relative_l2(field_prediction[..., :3], field_target[..., :3]),
                                       "field_all_channel_mse": mse(field_prediction, field_target)})
                    rollout_records.append(record)
                else:
                    skipped_horizons[h] += 1
            del pred_states, pred_fields, batch, targets
            if batch_index % 50 == 0 or batch_index == rollout_total:
                print(f"[rollout] processed {batch_index}/{rollout_total} pairs", flush=True)
    for horizon, count in skipped_horizons.items():
        if count:
            print(f"[rollout] skipped H={horizon} for {count} evaluated pairs: "
                  "the batch contains fewer rollout targets than this horizon")
    rollout_cases = sorted({str(record.get("case", "unknown")) for record in records})
    for case in rollout_cases:
        case_records = [record for record in rollout_records if record.get("case", "unknown") == case]
        for horizon in horizons:
            rows = [record for record in case_records if int(record["horizon"]) == horizon]
            if not rows:
                print(f"rollout coverage {case} H={horizon}: n_pairs=0 | position_relative_l2=n/a | field_relative_l2=n/a")
                continue
            summary = (f"rollout coverage {case} H={horizon}: n_pairs={len(rows)} | "
                       f"position_relative_l2={np.mean([row['position_relative_l2'] for row in rows]):.6g}")
            field_rows = [row for row in rows if "field_mse" in row]
            if field_rows:
                summary += f" | field_relative_l2={np.mean([row['field_relative_l2'] for row in field_rows]):.6g}"
            else:
                summary += " | field_relative_l2=n/a"
            print(summary)
        if not case_records:
            continue
        final_horizon = max(int(row["horizon"]) for row in case_records)
        final_rows = [row for row in case_records if int(row["horizon"]) == final_horizon]
        time_average = float(np.mean([row["position_rmse"] for row in case_records]))
        final_error = float(np.mean([row["position_rmse"] for row in final_rows]))
        final_phase = float(np.mean([row["phase"] for row in final_rows]))
        print(f"rollout summary {case}: time-averaged position RMSE={time_average:.6g}; "
              f"final H={final_horizon} phase={final_phase:.6g} position RMSE={final_error:.6g}")
    native_records = []
    native_visual_candidates = {}
    native_root_value = str(cfg["data"].get("native_task2_root", "")).strip()
    native_root = str((HERE / native_root_value).resolve()) if native_root_value and not Path(native_root_value).is_absolute() else native_root_value
    allow_missing_native = cfg["evaluation"].get(
        "allow_missing_native_task2", cfg["data"].get("allow_missing_native_task2", False))
    stored_contexts = data.get("pair_contexts", [])
    field_superres_ids = np.asarray(data.get("field_superres_pair_ids", []), dtype=np.int64)
    native_candidates = np.unique(np.concatenate((pair_ids, field_superres_ids)))
    field_superres_set = set(field_superres_ids.tolist())
    native_paths = {}
    for pair_id in native_candidates:
        raw_context = stored_contexts[int(pair_id)] if len(stored_contexts) else {}
        if isinstance(raw_context, np.ndarray):
            raw_context = raw_context.item()
        matched_path = str(raw_context.get("task2_field_path", ""))
        path = Path(matched_path) if matched_path else None
        if path is not None and not path.is_file() and native_root:
            path = Path(native_root) / str(raw_context.get("case", "")) / path.name
        if path is not None and path.is_file():
            native_paths[int(pair_id)] = path
    if native_paths:
        native_pair_ids = np.asarray(sorted(native_paths), dtype=np.int64)
        print(f"[native Task-2] evaluating {len(native_pair_ids)} readable frames; "
              f"skipping {len(native_candidates) - len(native_pair_ids)} missing paths")
        native_ds = EvolutionDataset(data, native_pair_ids, features, global_names,
                                     cfg["data"]["max_particles"], cfg["data"]["max_queries"], 1, stats, residual_channels)
        native_loader = DataLoader(native_ds, batch_size=1, shuffle=False, collate_fn=collate_one)
        if max_batches:
            from itertools import islice
            native_loader = islice(native_loader, max_batches)
        for batch in native_loader:
            context = batch["pair_context"]
            pair_id = int(batch["pair_id"][0])
            raw_context = data.get("pair_contexts", [])[pair_id]
            if isinstance(raw_context, np.ndarray):
                raw_context = raw_context.item()
            path = native_paths[pair_id]
            # Native Task-2 U is the evaluation truth. Gradient channels elsewhere in
            # the pipeline are finite-difference derivatives, not stored measurements.
            native_xyz, native_truth = read_task2_velocity(path)
            take = np.arange(len(native_xyz))
            if len(take) > int(cfg["data"]["max_queries"]):
                take = take[np.linspace(0, len(take) - 1, int(cfg["data"]["max_queries"]), dtype=np.int64)]
            xyz, truth = native_xyz[take], native_truth[take]
            normalized_native_xyz = (native_xyz - stats.coord_min) / np.maximum(stats.coord_span, 1e-8)
            outside_fraction = float(np.mean(np.any((normalized_native_xyz < 0.0) |
                                                     (normalized_native_xyz > 1.0), axis=1)))
            native_batch = dict(batch)
            native_batch["output_queries"] = stats.normalize_positions(torch.as_tensor(xyz, dtype=torch.float32).unsqueeze(0).to(device)).clamp(0.0, 1.0)
            native_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in native_batch.items()}
            with torch.inference_mode():
                result = predict_next_state(model, native_batch, latent, stats)
                predicted = result["field_phys"][0].cpu().numpy()
            native_records.append({"pair_id": pair_id, "case": context.get("case", "unknown"),
                "evaluation_role": "field_superresolution" if pair_id in field_superres_set else split,
                "frame": context.get("frame_t", ""), "phase": float(context.get("phase_t", 0.0)), "hdf5": str(path),
                "task1_xmf_time": raw_context.get("xmf_time_t"),
                "task2_xmf_time": raw_context.get("task2_xmf_time"),
                "task2_frame_id": raw_context.get("task2_frame_id"),
                "velocity_mse": mse(predicted[:, :3], truth),
                "velocity_relative_l2": relative_l2(predicted[:, :3], truth),
                "field_mse": mse(predicted[:, :3], truth),
                "field_relative_l2": relative_l2(predicted[:, :3], truth),
                "outside_train_domain_fraction": outside_fraction,
                "n_points": int(len(truth))})
            plane_mode = str(cfg["evaluation"].get("field_plane_mode", "highest_energy"))
            requested_y = (highest_energy_y_plane(native_xyz, native_truth)
                           if plane_mode == "highest_energy"
                           else float(cfg["evaluation"].get("field_y_plane", 0.0)))
            if plane_mode not in ("highest_energy", "mid"):
                raise ValueError("evaluation.field_plane_mode must be 'highest_energy' or 'mid'")
            if plane_mode == "mid":
                requested_y = float(np.unique(native_xyz[:, 1])[len(np.unique(native_xyz[:, 1])) // 2])
            x_axis, z_axis, _, plane_xyz, _ = native_field_plane(
                native_xyz, native_truth, y_plane=requested_y, component=0)
            true_components = []
            for component in range(3):
                _, _, _, _, values = native_field_plane(native_xyz, native_truth,
                    y_plane=requested_y, component=component)
                true_components.append(values.reshape(len(z_axis), len(x_axis)))
            true_velocity_slice = np.stack(true_components, axis=-1)
            plane_batch = dict(batch)
            plane_batch["output_queries"] = stats.normalize_positions(
                torch.as_tensor(plane_xyz, dtype=torch.float32).unsqueeze(0).to(device)).clamp(0.0, 1.0)
            plane_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in plane_batch.items()}
            with torch.inference_mode():
                plane_result = predict_next_state(model, plane_batch, latent, stats)
                predicted_velocity_slice = plane_result["field_phys"][0, :, :3].cpu().numpy().reshape(
                    len(z_axis), len(x_axis), 3)
            case_name = context.get("case", "unknown")
            evaluation_role = "field_superresolution" if pair_id in field_superres_set else split
            native_visual_candidates.setdefault((case_name, evaluation_role), []).append({
                "phase": float(context.get("phase_t", 0.0)), "y_plane": float(plane_xyz[0, 1]),
                "evaluation_role": evaluation_role,
                "freestream_magnitude": float(np.linalg.norm(context.get("freestream", [1.0, 0.0, 0.0]))),
                "physical_time": raw_context.get("task2_physical_time"),
                "vtk_path": str(context.get("vtk_path", "")),
                "x": x_axis, "z": z_axis,
                "true_velocity": true_velocity_slice, "predicted_velocity": predicted_velocity_slice})
            del native_batch, plane_batch, result, plane_result
        for case, role in sorted({(record.get("case", "unknown"), record["evaluation_role"])
                                  for record in native_records}):
            rows = [record for record in native_records
                    if record.get("case", "unknown") == case and record["evaluation_role"] == role]
            print(f"native Task-2 {case} ({role}): {len(rows)} frames | "
                  f"velocity MSE={np.mean([row['field_mse'] for row in rows]):.6g} | "
                  f"field relL2={np.mean([row['field_relative_l2'] for row in rows]):.6g}")
    if not native_records and not allow_missing_native:
        raise RuntimeError("Native Task-2 evaluation produced no records. Check stored task2_field_path "
                           "values or set evaluation.allow_missing_native_task2=true to allow this.")
    if not native_records:
        print("[native Task-2] skipped: no readable source field files were found; "
              "one-step and rollout evaluation results will still be saved.")
    out_dir = checkpoint_path.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "one_step.json", {"checkpoint": str(checkpoint_path), "dataset": str(data_path),
        "split": split, "architecture": model_cfg, "input_features": features, "target_features": target_names,
        "normalization_source": checkpoint.get("dataset_path", str(data_path)), "rollout_horizons": horizons, "records": records})
    write_json(out_dir / "autoregressive.json", {"checkpoint": str(checkpoint_path), "split": split, "records": rollout_records})
    write_json(out_dir / "native_task2_field.json", {"root": native_root, "records": native_records})
    if cfg["evaluation"].get("save_plots", True):
        plot_temporal_errors(records, rollout_records, out_dir)
        snapshot_cases = sorted({case for case, _ in rollout_visuals})
        for case_name in snapshot_cases:
            snapshots = {horizon: item for (case, horizon), item in rollout_visuals.items()
                         if case == case_name}
            plot_rollout_snapshots(snapshots,
                out_dir / f"rollout_snapshots_{case_name.replace('/', '_')}.png",
                title=f"{case_name}: autoregressive particles colored by circulation/core-size proxy")
    if cfg["evaluation"].get("save_plots", True) and native_visual_candidates:
        requested_phase = float(cfg["evaluation"].get("field_visual_phase", 0.5))
        for (case_name, role), candidates in native_visual_candidates.items():
            sample = min(candidates, key=lambda item: abs(item["phase"] - requested_phase))
            x_axis, z_axis = sample["x"], sample["z"]
            true_ux = sample["true_velocity"][..., 0]
            predicted_ux = sample["predicted_velocity"][..., 0]
            safe_case = case_name.replace("/", "_")
            time_value = sample.get("physical_time")
            plot_native_ux_comparison(x_axis, z_axis, true_ux, predicted_ux,
                out_dir / f"native_ux_comparison_{safe_case}_{role}.png",
                phase=sample["phase"], physical_time=time_value)
            configured_chord = cfg["evaluation"].get("reference_chord", None)
            configured_leading_edge = cfg["evaluation"].get("reference_leading_edge_x", None)
            if configured_chord is None or configured_leading_edge is None:
                derived_leading_edge, derived_chord = estimate_chordwise_bounds(sample["vtk_path"])
            chord = float(configured_chord if configured_chord is not None else derived_chord)
            leading_edge = float(configured_leading_edge if configured_leading_edge is not None else derived_leading_edge)
            plot_normalized_ux_profiles(x_axis, z_axis, sample["true_velocity"],
                sample["predicted_velocity"],
                out_dir / f"normalized_ux_profiles_{safe_case}_{role}.png",
                freestream_speed=sample["freestream_magnitude"], chord=chord,
                leading_edge_x=leading_edge,
                x_stations_over_c=cfg["evaluation"].get("profile_x_over_c", [0.2, 0.4, 0.6]))
        roles_by_case = {}
        for case_name, role in native_visual_candidates:
            roles_by_case.setdefault(case_name, set()).add(role)
        requested_phases = (0.3, 0.5, 0.7)
        for case_name, roles in sorted(roles_by_case.items()):
            for role in ("field_superresolution", "test", "validation"):
                if role not in roles:
                    continue
                candidates = native_visual_candidates[(case_name, role)]
                samples = [min(candidates, key=lambda item: abs(item["phase"] - phase))
                           for phase in requested_phases]
                x_axis, z_axis = samples[0]["x"], samples[0]["z"]
                true_slices = np.stack([sample["true_velocity"] for sample in samples])
                predicted_slices = np.stack([sample["predicted_velocity"] for sample in samples])
                speeds = [sample["freestream_magnitude"] if sample["freestream_magnitude"] > 0 else 1.0
                          for sample in samples]
                safe_case = case_name.replace("/", "_")
                role_suffix = "" if role == "test" else f"_{role}"
                plot_velocity_magnitude_summary(
                    x_axis, z_axis, true_slices, predicted_slices, speeds,
                    out_dir / f"velocity_magnitude_summary_{safe_case}{role_suffix}.png",
                    phases=requested_phases,
                    x_stations=cfg["evaluation"].get("profile_x_over_c", [0.2, 0.4, 0.6]),
                    y_plane=float(samples[0]["y_plane"]))
    if cfg["evaluation"].get("save_parity_plots", True):
        test_cases = sorted({str(record.get("case", "unknown")) for record in records})
        for case_name in test_cases:
            sample = one_step_visuals.get(case_name)
            if sample is None:
                continue
            safe_case = case_name.replace("/", "_")
            plot_state_parity(sample["truth"], sample["prediction"],
                              out_dir / f"one_step_state_parity_{safe_case}.png",
                              title=f"{case_name}: one-step particle state")
            if "field_truth" in sample:
                plot_field_error_maps(sample["coords"], sample["field_truth"], sample["field_prediction"],
                                      out_dir / f"field_error_maps_{safe_case}.png",
                                      title=f"{case_name}: one-step field prediction error")
    if cfg["evaluation"].get("save_plots", True):
        plot_rmse_vs_phase_grid(records, out_dir)
    if records:
        with open(out_dir / "one_step.csv", "w", newline="", encoding="utf-8") as stream:
            columns = list(dict.fromkeys(key for record in records for key in record))
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(records)
    if rollout_records:
        with open(out_dir / "autoregressive.csv", "w", newline="", encoding="utf-8") as stream:
            columns = list(dict.fromkeys(key for record in rollout_records for key in record))
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rollout_records)
    if native_records:
        with open(out_dir / "native_task2_field.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=native_records[0].keys()); writer.writeheader(); writer.writerows(native_records)
    print(f"Saved {len(records)} one-step and {len(rollout_records)} rollout records to {out_dir}")


if __name__ == "__main__":
    main()
