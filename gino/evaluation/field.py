import numpy as np
from gino.data.hdf5 import read_task2_field
from .metrics import mse, relative_l2


def evaluate_native_field(predicted_xyz, predicted_velocity, hdf5_path):
    """Compare only predictions sampled at the exact native physical coordinates."""
    true_xyz, true_velocity = read_task2_field(hdf5_path)
    predicted_xyz, predicted_velocity = np.asarray(predicted_xyz), np.asarray(predicted_velocity)
    if predicted_xyz.shape != true_xyz.shape or not np.allclose(predicted_xyz, true_xyz, rtol=1e-6, atol=1e-7):
        raise ValueError("Native field comparison requires prediction and reference at identical coordinates")
    if predicted_velocity.shape != true_velocity.shape:
        raise ValueError(f"Field channel/shape mismatch: prediction={predicted_velocity.shape}, truth={true_velocity.shape}")
    return {"mse": mse(predicted_velocity, true_velocity), "relative_l2": relative_l2(predicted_velocity, true_velocity),
            "matched_points": int(len(true_velocity)), "comparison": "native-grid exact coordinates", "hdf5": str(hdf5_path)}


import torch


FIELD_GROUPS = {"velocity": slice(0, 3), "gradient": slice(3, 12), "vorticity": slice(12, 15)}


def field_group_metrics_torch(pred_norm, target_norm, pred_phys, target_phys):
    """Per-group normalized MSE, physical RMSE, and physical relative L2."""
    if pred_norm.shape != target_norm.shape or pred_phys.shape != target_phys.shape:
        raise ValueError("Field prediction/reference shapes differ")
    result = {}
    for name, channels in FIELD_GROUPS.items():
        pn, tn = pred_norm[..., channels], target_norm[..., channels]
        pp, tp = pred_phys[..., channels], target_phys[..., channels]
        if not all(torch.isfinite(value).all() for value in (pn, tn, pp, tp)):
            raise ValueError(f"Non-finite values in {name} field metrics")
        result[f"{name}_normalized_mse"] = float((pn - tn).square().mean().cpu())
        result[f"{name}_physical_rmse"] = float((pp - tp).square().mean().sqrt().cpu())
        result[f"{name}_physical_relative_l2"] = float(
            torch.linalg.vector_norm((pp - tp).reshape(-1)).div(
                torch.linalg.vector_norm(tp.reshape(-1)).clamp_min(1e-8)).cpu())
    return result


@torch.inference_mode()
def evaluate_teacher_forced_field(model, loader, latent_grid, normalization, device):
    """Same-time field reconstruction using true particle inputs; no state rollout."""
    model.eval()
    records = []
    for batch in loader:
        batch = {key: (value.to(device) if torch.is_tensor(value) else value)
                 for key, value in batch.items()}
        encoded = model.encode_process(batch["input_geom"], latent_grid, batch["x"], batch["global_params"])
        pred_norm = model.decode_field(encoded, batch["output_queries"])
        if not bool(batch.get("field_available", True)):
            continue
        target_norm = batch["field_target"]
        pred_phys, target_phys = normalization.denormalize_field(pred_norm), normalization.denormalize_field(target_norm)
        context = batch.get("pair_context", {})
        records.append({"pair_id": int(batch["pair_id"][0]), "case": context.get("case", "unknown"),
                        "mode": "teacher_forced_field_reconstruction",
                        "task1_xmf_time": context.get("xmf_time_t"),
                        "task2_xmf_time": context.get("task2_xmf_time"),
                        **field_group_metrics_torch(pred_norm, target_norm, pred_phys, target_phys)})
    return records
