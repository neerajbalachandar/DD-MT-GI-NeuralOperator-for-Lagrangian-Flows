import torch
from gino.dynamics.state_transition import predict_next_state
from .metrics import state_component_metrics


@torch.inference_mode()
def evaluate_one_step(model, loader, latent_grid, normalization, device):
    model.eval()
    records = []
    mean = torch.as_tensor(normalization.residual_mean[:7], device=device)
    std = torch.as_tensor(normalization.residual_std[:7], device=device)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred, _, _, field_norm, _ = predict_next_state(model, batch, latent_grid, normalization)
        truth = batch["state_phys"] + batch["delta_target"][..., :7] * std + mean
        records.append({"pair_id": int(batch["pair_id"][0]), **state_component_metrics(pred.cpu().numpy(), truth.cpu().numpy()),
                        "field_mse_normalized": float(torch.mean((field_norm - batch["field_target"]) ** 2).cpu())})
    return records
