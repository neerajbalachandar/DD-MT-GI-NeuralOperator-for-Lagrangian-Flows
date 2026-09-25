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
from gino.model.gino import GINOSharedLatent, build_latent_grid
from gino.utils import load_config, write_json


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
    split = cfg["evaluation"]["split"]
    pair_ids = data[f"{split}_pair_ids"]
    ds = EvolutionDataset(data, pair_ids, features, global_names, cfg["data"]["max_particles"], cfg["data"]["max_queries"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_one)
    max_batches = int(cfg["evaluation"].get("max_batches", 0))
    if max_batches:
        from itertools import islice
        loader = list(islice(loader, max_batches))
    latent = build_latent_grid(model_cfg["latent_res"], device)
    records = evaluate_one_step(model, loader, latent, stats, device)
    out_dir = checkpoint_path.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "one_step.json", {"checkpoint": str(checkpoint_path), "dataset": str(data_path),
        "architecture": model_cfg, "input_features": features, "target_features": target_names,
        "normalization_source": checkpoint.get("dataset_path", str(data_path)), "rollout_horizons": cfg["evaluation"]["rollout_horizons"], "records": records})
    if records:
        with open(out_dir / "one_step.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    print(f"Saved {len(records)} one-step records to {out_dir}")


if __name__ == "__main__":
    main()
