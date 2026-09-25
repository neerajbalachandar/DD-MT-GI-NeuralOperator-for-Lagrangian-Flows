import torch
import torch.nn.functional as F


def relative_l2(pred, target, eps=1e-8):
    pred, target = torch.nan_to_num(pred), torch.nan_to_num(target)
    numerator = torch.linalg.vector_norm((pred - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def state_loss(pred, target):
    return F.mse_loss(pred, target)


def residual_loss(pred, target):
    return F.mse_loss(pred, target)


def field_loss(pred, target, relative=False):
    return relative_l2(pred, target) if relative else F.mse_loss(pred, target)


def rollout_loss(predicted, target, step_weights=None):
    terms = [(predicted[:, k] - target[:, k]).square().mean() for k in range(predicted.shape[1])]
    weights = [1.0] * len(terms) if step_weights is None else step_weights
    return sum(float(weights[k]) * term for k, term in enumerate(terms)) / max(sum(float(w) for w in weights), 1e-12)


def normalized_rollout_loss(predicted, target, mean, std, step_weights=None):
    mean = torch.as_tensor(mean, dtype=predicted.dtype, device=predicted.device)
    std = torch.as_tensor(std, dtype=predicted.dtype, device=predicted.device).clamp_min(1e-8)
    return rollout_loss((predicted - mean) / std, (target - mean) / std, step_weights)


def homoscedastic_multitask(state_term, field_term, log_state_var, log_field_var):
    # Existing notebook formulation: exp(-s) * loss + s for each task.
    return torch.exp(-log_state_var) * state_term + log_state_var + torch.exp(-log_field_var) * field_term + log_field_var


def combined_loss(state_term=None, field_term=None, rollout_term=None, weights=None,
                  log_state_var=None, log_field_var=None, homoscedastic=False):
    weights = weights or {}
    state_term = state_term if state_term is not None else torch.zeros(())
    field_term = field_term if field_term is not None else torch.zeros_like(state_term)
    if homoscedastic and log_state_var is not None and log_field_var is not None:
        total = homoscedastic_multitask(state_term, field_term, log_state_var, log_field_var)
    else:
        total = float(weights.get("state", 1.0)) * state_term + float(weights.get("field", 1.0)) * field_term
    if rollout_term is not None:
        total = total + float(weights.get("rollout", 0.0)) * rollout_term
    return total
