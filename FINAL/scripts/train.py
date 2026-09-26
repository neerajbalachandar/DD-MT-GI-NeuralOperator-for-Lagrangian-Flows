import argparse
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
from gino.model.gino import GINOSharedLatent, build_latent_grid
from gino.training.trainer import Trainer, set_deterministic_seed
from gino.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train the shared latent VPM GINO.")
    parser.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    set_deterministic_seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else ("cpu" if cfg["device"] == "auto" else cfg["device"]))
    dataset_path = (HERE / cfg["data"]["dataset"]).resolve()
    data = load_processed_dataset(dataset_path)
    names = [str(x) for x in data["feature_names"].tolist()]
    in_names = cfg["data"]["input_features"]
    indices = [names.index(n) for n in in_names]
    stats = NormalizationStats.fit_train_sequences(data, data["train_pair_ids"], indices)
    max_p, max_q = cfg["data"]["max_particles"], cfg["data"]["max_queries"]
    max_h = int(cfg["training"].get("rollout_horizon_max", 16))
    delta_channels = 10 if cfg["model"].get("predict_delta_u", True) else 7
    sequence_training = bool(cfg["training"].get("use_rollout_loss") or cfg["training"].get("use_pushforward"))
    make_ds = lambda ids, teacher=False: EvolutionDataset(
        data, ids, in_names, cfg["data"]["global_condition_channels"], max_p, max_q,
        max_h, stats, delta_channels, include_teacher_inputs=teacher,
        require_verified_correspondence=sequence_training)
    train_ds, val_ds = make_ds(data["train_pair_ids"], cfg["training"].get("use_scheduled_sampling", False)), make_ds(data.get("val_pair_ids", []))
    if not len(val_ds):
        raise ValueError("Validation split is empty; training uses validation only for checkpoint selection.")
    model = GINOSharedLatent(len(in_names), delta_channels, len(data["field_target_names"]),
                             len(cfg["data"]["global_condition_channels"]), cfg["model"]).to(device)
    latent = build_latent_grid(cfg["model"]["latent_res"], device)
    run_dir = (HERE / cfg["training"]["run_dir"]).resolve()
    norm_meta = {k: np.asarray(v).tolist() for k, v in vars(stats).items()}
    trainer = Trainer(model, latent, train_ds, val_ds, stats, cfg["training"], run_dir, device)
    trainer.fit(cfg, {"dataset_path": str(dataset_path), "split": {k: np.asarray(data[k]).tolist() for k in ("train_pair_ids", "val_pair_ids", "test_pair_ids")}, "normalization": norm_meta,
                      "feature_names": in_names, "target_names": [str(x) for x in data["target_names"][:delta_channels]],
                      "field_target_names": [str(x) for x in data["field_target_names"]],
                      "global_condition_channels": cfg["data"]["global_condition_channels"]})
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
