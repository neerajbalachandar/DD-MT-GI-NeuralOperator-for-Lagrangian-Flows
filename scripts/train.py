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
    dataset_path = (HERE / cfg["data"]["dataset"]).resolve()
    data = load_processed_dataset(dataset_path)
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
    if split["val_pair_ids"].size == 0:
        raise ValueError(
            "Validation split is empty and cannot be created: the dataset needs at least two non-test cases."
        )
    if np.intersect1d(split["train_pair_ids"], split["val_pair_ids"]).size:
        raise ValueError("Training and validation pair IDs overlap; provide a leakage-free split.")
    print(f"[split] validation source: {validation_source}")
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
        available_horizon = maximum_available_horizon(data, split["train_pair_ids"], requested_horizon)
        if available_horizon < 2:
            raise ValueError(
                "The processed training split contains no multi-step chains with shared tracked particles, "
                "so rollout losses would remain inactive. Rebuild it with `python3 scripts/preprocess.py` "
                "using the updated pipeline, which retains particle transitions when Task-2 field data is absent."
            )
        print(f"[rollout] longest available training chain: {available_horizon} steps "
              f"(configured maximum: {requested_horizon})")

    names = [str(x) for x in data["feature_names"].tolist()]
    in_names = cfg["data"]["input_features"]
    indices = [names.index(n) for n in in_names]
    field_holdout_ids = np.asarray(data.get("field_superres_pair_ids", []), dtype=np.int64)
    field_train_ids = np.setdiff1d(split["train_pair_ids"], field_holdout_ids)
    stats = NormalizationStats.fit_train_sequences(data, split["train_pair_ids"], indices,
                                                   field_pair_ids=field_train_ids)
    max_p, max_q = cfg["data"]["max_particles"], cfg["data"]["max_queries"]
    max_h = int(cfg["training"].get("rollout_horizon_max", 16))
    delta_channels = 10 if cfg["model"].get("predict_delta_u", True) else 7
    make_ds = lambda ids, teacher=False, excluded_fields=None: EvolutionDataset(
        data, ids, in_names, cfg["data"]["global_condition_channels"], max_p, max_q,
        max_h, stats, delta_channels, include_teacher_inputs=teacher,
        require_verified_correspondence=sequence_training,
        field_target_exclude_ids=excluded_fields)
    train_ds = make_ds(split["train_pair_ids"], cfg["training"].get("use_scheduled_sampling", False),
                       field_holdout_ids)
    val_ds = make_ds(split["val_pair_ids"])
    model = GINOSharedLatent(len(in_names), delta_channels, len(data["field_target_names"]),
                             len(cfg["data"]["global_condition_channels"]), cfg["model"]).to(device)
    latent = build_latent_grid(cfg["model"]["latent_res"], device)
    run_dir = (HERE / cfg["training"]["run_dir"]).resolve()
    norm_meta = {k: np.asarray(v).tolist() for k, v in vars(stats).items()}
    checkpoint_split = {k: np.asarray(split[k]).tolist()
                        for k in ("train_pair_ids", "val_pair_ids", "test_pair_ids")}
    checkpoint_split["field_superres_pair_ids"] = field_holdout_ids.tolist()
    checkpoint_split["validation_source"] = validation_source
    trainer = Trainer(model, latent, train_ds, val_ds, stats, training_cfg, run_dir, device)
    trainer.fit(cfg, {"dataset_path": str(dataset_path), "split": checkpoint_split, "normalization": norm_meta,
                      "feature_names": in_names, "target_names": [str(x) for x in data["target_names"][:delta_channels]],
                      "field_target_names": [str(x) for x in data["field_target_names"]],
                      "global_condition_channels": cfg["data"]["global_condition_channels"]})
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
