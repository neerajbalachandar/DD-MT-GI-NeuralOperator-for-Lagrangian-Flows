"""Train the co-moving VortexGINO on the lossless vortex-ring archives.

    python scripts/train_vortex.py --config configs/vortex_gino.yaml [--set training.epochs=10]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gino.data.vortex_frames import load_split  # noqa: E402
from gino.model.vortex_operator import VortexGINO  # noqa: E402
from gino.training.vortex_trainer import VortexTrainer, estimate_scales  # noqa: E402
from gino.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "configs/vortex_gino.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default="", help="checkpoint to resume (weights + epoch counter)")
    parser.add_argument("--init", default="", help="initialise weights (EMA) from a checkpoint; fresh optimizer/history")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available()
                          else (cfg["device"] if cfg.get("device", "auto") != "auto" else "cpu"))
    root = HERE / cfg["data"]["root"]
    train = load_split(root, "train", device, cfg["data"].get("train_cases") or None)
    val = load_split(root, "val", device, cfg["data"].get("val_cases") or None)
    print(f"[data] train: {[c.name for c in train]}")
    print(f"[data] val:   {[c.name for c in val]}")
    model = VortexGINO(cfg["model"]).to(device)
    scales = estimate_scales(train, model)
    residual_scale = [0.05 * scales["velocity_scale"]] * 3 + [1.0] * 3
    model.set_scales(scales["gamma_scale"], scales["velocity_scale"], scales["gradient_scale"], residual_scale,
                     gamma_rate_scale=scales["gamma_rate_scale"], dt_ref=scales["dt_ref"])
    print(f"[scales] {json.dumps(scales)}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    tcfg = dict(cfg["training"])
    tcfg["seed"] = seed
    run_dir = HERE / tcfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    trainer = VortexTrainer(model, train, val, tcfg, run_dir, device)
    start_epoch = global_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("raw_model_state_dict", ck["model_state_dict"]))
        trainer.ema.load_state_dict(ck["model_state_dict"])
        trainer.history = ck.get("history", [])
        start_epoch, global_step = int(ck.get("epoch", 0)), int(ck.get("global_step", 0))
        if "optimizer_state_dict" in ck:
            trainer.opt.load_state_dict(ck["optimizer_state_dict"])
        if "rng_state" in ck:
            trainer.rng.bit_generator.state = ck["rng_state"]
        print(f"[resume] from {args.resume} at epoch {start_epoch}")
    elif args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        trainer.ema.load_state_dict(ck["model_state_dict"])
        print(f"[init] weights from {args.init} (epoch {ck.get('epoch')})")
    trainer.fit(start_epoch, global_step)
    print(f"[done] best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
