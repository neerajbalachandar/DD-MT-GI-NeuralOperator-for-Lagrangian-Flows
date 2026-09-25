import torch
from gino.dynamics.state_transition import predict_next_state
from .metrics import state_component_metrics, normalized_state_component_metrics, mse, relative_l2


@torch.inference_mode()
def evaluate_one_step(model, loader, latent_grid, normalization, device):
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
        true_field = normalization.denormalize_field(batch["field_target"])
        pred_field = result["field_phys"]
        context = batch.get("pair_context", {})
        records.append({"pair_id": int(batch["pair_id"][0]), "case": context.get("case", "unknown"),
                        "phase": float(batch.get("phase_next", 0.0)),
                        "field_phase": float(context.get("phase_t", 0.0)), **metrics,
                        "field_mse_normalized": float(torch.mean((result["field_norm"] - batch["field_target"]) ** 2).cpu()),
                        "field_mse": float(mse(pred_field.cpu().numpy(), true_field.cpu().numpy())),
                        "field_relative_l2": relative_l2(pred_field.cpu().numpy(), true_field.cpu().numpy())})
    return records
