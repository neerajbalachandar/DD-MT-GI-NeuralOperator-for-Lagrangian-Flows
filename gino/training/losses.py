import torch
import torch.nn.functional as F


def relative_l2(pred, target, eps=1e-8):
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Relative-L2 inputs contain NaN or infinite values")
    numerator = torch.linalg.vector_norm((pred - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def state_loss(pred, target):
    return F.mse_loss(pred, target)


def residual_loss(pred, target):
    return F.mse_loss(pred, target)


def field_loss(pred, target, relative=False):
    return relative_l2(pred, target) if relative else F.mse_loss(pred, target)


def grouped_field_loss(pred, target, weights=None, relative=True):
    """Return aggregate and raw relative-L2 losses for u, grad(u), and omega."""
    if pred.shape != target.shape or pred.shape[-1] != 15:
        raise ValueError(f"Expected matching field tensors with 15 channels, got {pred.shape} and {target.shape}")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Field prediction/target contains NaN or infinite values")
    weights = weights or {"velocity": 1.0, "gradient": 0.5, "vorticity": 0.1}
    groups = {"velocity": slice(0, 3), "gradient": slice(3, 12), "vorticity": slice(12, 15)}
    terms = {name: relative_l2(pred[..., channels], target[..., channels]) if relative
             else F.mse_loss(pred[..., channels], target[..., channels])
             for name, channels in groups.items()}
    aggregate = sum(float(weights.get(name, 0.0)) * term for name, term in terms.items())
    return aggregate, terms


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
    if homoscedastic and log_state_var is not None and log_field_var is not None:
        terms = []
        if state_term is not None:
            terms.append(torch.exp(-log_state_var) * state_term + log_state_var)
        if field_term is not None:
            terms.append(torch.exp(-log_field_var) * field_term + log_field_var)
        if terms:
            total = sum(terms)
        elif rollout_term is not None:
            total = rollout_term * 0.0
        else:
            raise ValueError("At least one task loss must be enabled")
    else:
        terms = []
        if state_term is not None:
            terms.append(float(weights.get("state", 1.0)) * state_term)
        if field_term is not None:
            terms.append(float(weights.get("field", 1.0)) * field_term)
        if terms:
            total = sum(terms)
        elif rollout_term is not None:
            total = rollout_term * 0.0
        else:
            raise ValueError("At least one task loss must be enabled")
    if rollout_term is not None:
        total = total + float(weights.get("rollout", 0.0)) * rollout_term
    return total
