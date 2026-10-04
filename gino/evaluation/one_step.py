import torch
from gino.dynamics.state_transition import predict_next_state
from .metrics import state_component_metrics, normalized_state_component_metrics, mse, relative_l2


@torch.inference_mode()
def evaluate_one_step(model, loader, latent_grid, normalization, device, visual_samples=None):
    model.eval()
    records = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        result = predict_next_state(model, batch, latent_grid, normalization)
        pred = result["predicted_state"]
        truth = batch["next_state_phys"]
        pred_np, truth_np = pred.cpu().numpy(), truth.cpu().numpy()
        metrics = state_component_metrics(pred_np, truth_np)
        metrics.update(normalized_state_component_metrics(pred_np, truth_np, normalization.state_mean, normalization.state_std))
        field_available = bool(batch.get("field_available", True))
        context = batch.get("pair_context", {})
        case = context.get("case", "unknown")
        if visual_samples is not None:
            sample = visual_samples.setdefault(case, {
                "truth": truth[0].cpu().numpy(),
                "prediction": pred[0].cpu().numpy(),
                "phase": float(batch.get("phase_next", 0.0)),
            })
            if field_available and "field_truth" not in sample:
                sample.update({"coords": batch["query_xyz_phys"][0].cpu().numpy(),
                               "field_truth": normalization.denormalize_field(batch["field_target"])[0].cpu().numpy(),
                               "field_prediction": result["field_phys"][0].cpu().numpy(),
                               "field_phase": float(batch.get("phase_next", 0.0))})
        record = {"pair_id": int(batch["pair_id"][0]), "case": case,
                  "phase": float(batch.get("phase_next", 0.0)),
                  "field_phase": float(context.get("phase_t", 0.0)),
                  "physical_time": context.get("physical_time_tp1"),
                  "field_physical_time": context.get("task2_physical_time"),
                  "task1_xmf_time": context.get("xmf_time_t"),
                  "task2_xmf_time": context.get("task2_xmf_time"), **metrics}
        if field_available:
            true_field = normalization.denormalize_field(batch["field_target"])
            pred_field = result["field_phys"]
            pred_np, true_np = pred_field.cpu().numpy(), true_field.cpu().numpy()
            record.update({"field_mse_normalized": float(torch.mean((result["field_norm"] - batch["field_target"]) ** 2).cpu()),
                           "field_mse": float(mse(pred_np[..., :3], true_np[..., :3])),
                           "field_relative_l2": relative_l2(pred_np[..., :3], true_np[..., :3]),
                           "field_all_channel_mse": float(mse(pred_np, true_np))})
        records.append(record)
    return records
