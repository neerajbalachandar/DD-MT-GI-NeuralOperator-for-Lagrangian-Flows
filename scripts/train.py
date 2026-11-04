import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader
from gino.data.dataset import (
    EvolutionDataset,
    SUPPORTED_ROLLOUT_CORRESPONDENCE,
    ensure_legacy_input_feature_columns,
    load_processed_dataset,
    sequence_level_split,
)
from gino.data.normalization import NormalizationStats
from gino.model.gino import GINOSharedLatent, build_latent_grid
from gino.training.trainer import Trainer, set_deterministic_seed
from gino.utils import load_config


def maximum_available_horizon(data, pair_ids, limit):
    ranges = list(data["pair_ranges"])
    selected = {int(pair_id) for pair_id in pair_ids}
    starts = {(str(row[0]), str(row[1])): index for index, row in enumerate(ranges)}
    particle_ids = data.get("particle_ids_t")
    maximum = 1 if selected else 0
    for pair_id in selected:
        case, _, frame_tp1 = ranges[pair_id][:3]
        horizon = 1
        common_ids = None
        if particle_ids is not None:
            start, end = (int(value) for value in ranges[pair_id][3:5])
            common_ids = set(np.asarray(particle_ids[start:end]).astype(str).tolist())
        while horizon < limit:
            following = starts.get((str(case), str(frame_tp1)))
            if following is None or following not in selected:
                break
            if common_ids is not None:
                start, end = (int(value) for value in ranges[following][3:5])
                common_ids.intersection_update(np.asarray(particle_ids[start:end]).astype(str).tolist())
                if not common_ids:
                    break
            horizon += 1
            frame_tp1 = ranges[following][2]
        maximum = max(maximum, horizon)
    return maximum


def report_identity_retention(dataset, pair_ids, horizon, split_name):
    """Report rho_k: fraction of each start's particles still shared at step k."""
    counts = {}
    for pair_id in pair_ids:
        pair_id = int(pair_id)
        row = dataset.data["pair_ranges"][pair_id]
        case = str(row[0])
        initial = set(dataset._pair_particle_ids(pair_id, "t").tolist())
        shared = set(initial)
        expected_frame = str(row[2])
        next_id = dataset.next_pair.get(pair_id)
        for step in range(1, int(horizon)):
            if next_id is None or next_id not in dataset.allowed_pair_ids:
                break
            next_row = dataset.data["pair_ranges"][next_id]
            if str(next_row[0]) != case or str(next_row[1]) != expected_frame:
                break
            shared.intersection_update(dataset._pair_particle_ids(next_id, "t").tolist())
            entry = counts.setdefault((case, step + 1), [0, 0])
            entry[0] += len(shared)
            entry[1] += len(initial)
            if not shared:
                break
            expected_frame = str(next_row[2])
            next_id = dataset.next_pair.get(next_id)
    for case in sorted({key[0] for key in counts}):
        values = [f"rho_{step}={numerator / denominator:.3f}"
                  for (item_case, step), (numerator, denominator) in sorted(counts.items())
                  if item_case == case]
        if values:
            print(f"[identity] {split_name} {case}: " + ", ".join(values))


def main():
    parser = argparse.ArgumentParser(description="Train the shared latent VPM GINO.")
    parser.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    training_cfg = cfg["training"]
    schedule = training_cfg.get("rollout_horizon_schedule", [1])
    if not isinstance(schedule, (list, tuple)) or not schedule or any(int(h) < 1 for h in schedule):
        raise ValueError("training.rollout_horizon_schedule must be a non-empty list of positive integers.")
    if int(training_cfg.get("rollout_horizon_max", max(schedule))) < 1:
        raise ValueError("training.rollout_horizon_max must be at least 1.")
    if not training_cfg.get("use_state_loss", True) and not training_cfg.get("use_field_loss", True):
        raise ValueError("Enable at least one of training.use_state_loss or training.use_field_loss.")
    training_cfg["epochs"] = int(training_cfg["epochs"])
    if training_cfg["epochs"] < 1:
        raise ValueError("training.epochs must be at least 1.")
    set_deterministic_seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    if device.type == "cuda":
        print(f"[device] training on {torch.cuda.get_device_name(device)}")
    else:
        print(f"[device] training on {device}; CUDA available={torch.cuda.is_available()}")
    dataset_path = Path(cfg["data"]["dataset"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = HERE / dataset_path
    dataset_path = dataset_path.resolve()
    merged_frames_value = str(cfg["data"].get("merged_frames", "")).strip()
    if merged_frames_value:
        merged_root = Path(merged_frames_value).expanduser()
        if not merged_root.is_absolute():
            merged_root = HERE / merged_root
        merged_root = merged_root.resolve()
        if not merged_root.is_dir():
            raise FileNotFoundError(f"Configured VPM merged-frame directory does not exist: {merged_root}")
        expected_cases = {path.name for path in merged_root.iterdir() if path.is_dir()}
        rebuild_reason = None
        if not dataset_path.is_file():
            rebuild_reason = "final archive is missing"
        else:
            try:
                with np.load(dataset_path, allow_pickle=True) as archive:
                    present_cases = {str(value) for key in ("train_cases", "val_cases", "test_cases")
                                     for value in archive[key]} if all(
                                         key in archive for key in ("train_cases", "val_cases", "test_cases")) else set()
                    field_names = {str(value) for value in archive["field_target_names"]
                                   } if "field_target_names" in archive else set()
                    schema_version = int(np.asarray(archive["preprocessing_schema_version"]).reshape(-1)[0]
                                         if "preprocessing_schema_version" in archive else 0)
                if expected_cases != present_cases:
                    rebuild_reason = "archive case IDs do not match merged-frame folders"
                elif not {"omega_x", "omega_y", "omega_z"}.issubset(field_names):
                    rebuild_reason = "archive predates the velocity/vorticity target update"
                elif schema_version < 3:
                    rebuild_reason = "archive predates the memory-bounded preprocessing format"
            except Exception as exc:
                rebuild_reason = f"archive could not be validated ({exc})"
        if rebuild_reason:
            import os
            os.environ["PREPROC_STREAMING"] = "1"
            os.environ["MAX_PREPROCESS_PARTICLES"] = str(int(cfg["data"].get("max_particles", 2048)))
            os.environ["MAX_FIELD_QUERY_POINTS"] = str(int(cfg["data"].get("max_queries", 4096)))
            rss_cap = float(cfg.get("preprocessing", {}).get("max_rss_gb", 0) or 0)
            if rss_cap > 0:
                os.environ["PREPROC_MAX_RSS_GB"] = str(rss_cap)
            from gino.data.preprocessing import run_preprocessing_from_merged_frames
            print(f"[dataset] rebuilding from {merged_root}: {rebuild_reason}", flush=True)
            dataset_path = run_preprocessing_from_merged_frames(
                merged_root, dataset_path.parent,
                raw_root=cfg["data"].get("native_task2_root"),
                device=str(cfg.get("device", "auto")))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed training archive not found: {dataset_path}")
    print(f"[dataset] training archive: {dataset_path}", flush=True)
    data = ensure_legacy_input_feature_columns(load_processed_dataset(dataset_path), cfg["data"]["input_features"])
    cfg["data"]["input_features"] = [str(x) for x in data["feature_names"].tolist()]
    split = {
        "train_pair_ids": np.asarray(data["train_pair_ids"], dtype=np.int64),
        "val_pair_ids": np.asarray(data.get("val_pair_ids", []), dtype=np.int64),
        "test_pair_ids": np.asarray(data.get("test_pair_ids", []), dtype=np.int64),
    }
    validation_source = "case-level"
    internal_val = np.asarray(data.get("train_id_val_pair_ids", []), dtype=np.int64)
    if split["val_pair_ids"].size == 0:
        if internal_val.size:
            split["val_pair_ids"] = internal_val
            validation_source = "train-case temporal holdout"
        else:
            split = sequence_level_split(data, seed=int(cfg["seed"]))
            validation_source = "sequence-level fallback"
    elif internal_val.size:
        split["val_pair_ids"] = np.unique(np.concatenate((split["val_pair_ids"], internal_val)))
        validation_source = "case-level plus train-case temporal holdout"
    if split["train_pair_ids"].size == 0:
        raise ValueError("Training split is empty; check the processed dataset's case assignments.")
    names = [str(x) for x in data["feature_names"].tolist()]
    in_names = cfg["data"]["input_features"]
    if split["val_pair_ids"].size == 0:
        raise ValueError(
            "Validation split is empty and cannot be created: the dataset needs at least two non-test cases."
        )
    if np.intersect1d(split["train_pair_ids"], split["val_pair_ids"]).size:
        raise ValueError("Training and validation pair IDs overlap; provide a leakage-free split.")
    print(f"[split] validation source: {validation_source}")
    case_categories = {}
    for pair_id in split["test_pair_ids"]:
        context = data.get("pair_contexts", [])[int(pair_id)]
        if isinstance(context, np.ndarray):
            context = context.item()
        case_categories[str(context.get("case", "unknown"))] = [
            str(value) for value in context.get("evaluation_categories", [])
        ]
    print(f"[split] pair counts train/val/test: {len(split['train_pair_ids'])}/"
          f"{len(split['val_pair_ids'])}/{len(split['test_pair_ids'])}")
    print("[test protocol] held-out cases and categories:")
    for case, categories in sorted(case_categories.items()):
        print(f"  {case}: {', '.join(categories) if categories else 'uncategorized'}")
    max_h = int(cfg["training"].get("rollout_horizon_max", 16))
    sequence_training = bool(cfg["training"].get("use_rollout_loss") or cfg["training"].get("use_pushforward"))
    if sequence_training:
        contexts = data.get("pair_contexts", [])
        required_ids = np.concatenate((split["train_pair_ids"], split["val_pair_ids"]))
        unsupported = []
        for pair_id in required_ids:
            context = contexts[int(pair_id)] if len(contexts) else {}
            if isinstance(context, np.ndarray):
                context = context.item()
            source = str(context.get("correspondence_source", "missing"))
            if source not in SUPPORTED_ROLLOUT_CORRESPONDENCE:
                unsupported.append(source)
        if unsupported:
            raise ValueError(
                f"Rollout/pushforward training requires supported particle correspondence, but "
                f"{len(unsupported)} of {len(required_ids)} selected train/validation pairs are not supported "
                f"(sources: {sorted(set(unsupported))}). Rebuild the processed dataset with "
                "`python3 scripts/preprocess.py` before training."
            )
        requested_horizon = min(int(training_cfg.get("rollout_horizon_max", max(schedule))), max(schedule))
        chain_diagnostics = EvolutionDataset(
            data, split["train_pair_ids"], in_names,
            cfg["data"]["global_condition_channels"], rollout_horizon=requested_horizon)
        train_chain_lengths = {int(pair_id): chain_diagnostics.chain_length(int(pair_id))
                               for pair_id in split["train_pair_ids"]}
        available_horizon = max(train_chain_lengths.values(), default=0)
        if available_horizon < 2:
            raise ValueError(
                "The processed training split contains no multi-step chains with shared tracked particles, "
                "so rollout losses would remain inactive. Rebuild it with `python3 scripts/preprocess.py` "
                "using the updated pipeline, which retains particle transitions when Task-2 field data is absent."
            )
        print(f"[rollout] longest available training chain: {available_horizon} steps "
              f"(configured maximum: {requested_horizon})")
        # Do not spend epochs requesting rollout lengths the processed data
        # cannot supply. Keep the curriculum intact while capping its tail.
        max_h = min(max_h, available_horizon)
        schedule = [min(int(h), max_h) for h in schedule]
        training_cfg["rollout_horizon_max"] = max_h
        training_cfg["rollout_horizon_schedule"] = schedule

    indices = [names.index(n) for n in in_names]
    field_holdout_ids = np.asarray(data.get("field_superres_pair_ids", []), dtype=np.int64)
    field_train_ids = np.setdiff1d(split["train_pair_ids"], field_holdout_ids)
    stats = NormalizationStats.fit_train_sequences(data, split["train_pair_ids"], indices,
                                                   field_pair_ids=field_train_ids)
    max_p, max_q = cfg["data"]["max_particles"], cfg["data"]["max_queries"]
    delta_channels = 10 if cfg["model"].get("predict_delta_u", True) else 7
    make_ds = lambda ids, horizon, teacher=False, excluded_fields=None: EvolutionDataset(
        data, ids, in_names, cfg["data"]["global_condition_channels"], max_p, max_q,
        horizon, stats, delta_channels, include_teacher_inputs=teacher,
        require_verified_correspondence=sequence_training,
        field_target_exclude_ids=excluded_fields)
    training_pair_ids = split["train_pair_ids"]
    validation_pair_ids = split["val_pair_ids"]
    if sequence_training:
        diagnostic_ds = EvolutionDataset(data, training_pair_ids, in_names,
                                         cfg["data"]["global_condition_channels"], rollout_horizon=max_h)

        def has_rollout_overlap(pair_id):
            row = data["pair_ranges"][int(pair_id)]
            shared_ids = set(diagnostic_ds._pair_particle_ids(int(pair_id), "t").tolist())
            expected_frame = str(row[2])
            next_id = diagnostic_ds.next_pair.get(int(pair_id))
            future_count = 0
            while (next_id is not None and next_id in diagnostic_ds.allowed_pair_ids
                   and future_count < max_h - 1):
                next_row = data["pair_ranges"][next_id]
                if str(next_row[0]) != str(row[0]) or str(next_row[1]) != expected_frame:
                    return False
                shared_ids.intersection_update(diagnostic_ds._pair_particle_ids(next_id, "t").tolist())
                if not shared_ids:
                    return False
                future_count += 1
                expected_frame = str(next_row[2])
                next_id = diagnostic_ds.next_pair.get(next_id)
            return future_count > 0

        training_pair_ids = np.asarray([
            pid for pid in training_pair_ids
            if diagnostic_ds.chain_length(int(pid)) >= 2 and has_rollout_overlap(int(pid))
        ], dtype=np.int64)
        if not training_pair_ids.size:
            raise ValueError("Rollout training requires chainable training pairs; "
                             "re-run scripts/preprocess.py after identity-tracking fix.")
    train_ds = make_ds(training_pair_ids, max_h,
                       cfg["training"].get("use_scheduled_sampling", False), field_holdout_ids)
    # Keep all in-split transitions available as rollout context, including a
    # terminal pair that is not itself a valid multi-step training start.
    train_ds.allowed_pair_ids = set(split["train_pair_ids"].tolist())
    validation_diagnostics = EvolutionDataset(
        data, validation_pair_ids, in_names, cfg["data"]["global_condition_channels"],
        rollout_horizon=max_h)
    validation_chain_lengths = {
        int(pair_id): validation_diagnostics.chain_length(int(pair_id))
        for pair_id in validation_pair_ids
    }
    validation_horizon = min(max_h, max(validation_chain_lengths.values(), default=1))
    if sequence_training and validation_horizon < 2:
        print("[validation] no multi-step validation chain available; falling back to one-step validation for checkpoint selection.")
        validation_horizon = 1
        validation_starts = validation_pair_ids
    else:
        validation_starts = (np.asarray([pid for pid in validation_pair_ids
                                         if validation_chain_lengths[int(pid)] >= 2], dtype=np.int64)
                             if sequence_training else validation_pair_ids)
        if sequence_training and not validation_starts.size:
            raise ValueError("The validation split contains no chainable rollout starts.")
    val_ds = make_ds(validation_starts, validation_horizon)
    # Future pairs remain available as context, including terminal transitions
    # that are not themselves used as validation starts.
    val_ds.allowed_pair_ids = set(validation_pair_ids.tolist())
    print(f"[validation] longest consecutive chain: {validation_horizon} steps; "
          f"rollout starts: {len(validation_starts)}/{len(validation_pair_ids)}")
    retention_train = EvolutionDataset(
        data, split["train_pair_ids"], in_names,
        cfg["data"]["global_condition_channels"], rollout_horizon=max_h)
    report_identity_retention(retention_train, split["train_pair_ids"], max_h, "train")
    report_identity_retention(validation_diagnostics, validation_pair_ids,
                              validation_horizon, "validation")
    model = GINOSharedLatent(len(in_names), delta_channels, len(data["field_target_names"]),
                             len(cfg["data"]["global_condition_channels"]), cfg["model"]).to(device)
    latent = build_latent_grid(cfg["model"]["latent_res"], device)
    run_dir = (HERE / cfg["training"]["run_dir"]).resolve()
    norm_meta = {k: np.asarray(v).tolist() for k, v in vars(stats).items()}
    checkpoint_split = {k: np.asarray(split[k]).tolist()
                        for k in ("train_pair_ids", "val_pair_ids", "test_pair_ids")}
    checkpoint_split["field_superres_pair_ids"] = field_holdout_ids.tolist()
    checkpoint_split["validation_source"] = validation_source
    checkpoint_split["validation_rollout_horizon"] = validation_horizon
    checkpoint_split["test_case_categories"] = case_categories
    trainer = Trainer(model, latent, train_ds, val_ds, stats, training_cfg, run_dir, device)
    trainer.fit(cfg, {"dataset_path": str(dataset_path), "split": checkpoint_split, "normalization": norm_meta,
                      "feature_names": in_names, "target_names": [str(x) for x in data["target_names"][:delta_channels]],
                      "field_target_names": [str(x) for x in data["field_target_names"]],
                      "global_condition_channels": cfg["data"]["global_condition_channels"]})
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
