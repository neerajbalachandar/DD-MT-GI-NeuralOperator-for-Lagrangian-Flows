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


def normalized_physical_state_loss(predicted_phys, target_phys, mean, std):
    if predicted_phys.shape != target_phys.shape or predicted_phys.shape[-1] != 7:
        raise ValueError(f"Expected matching physical states with seven channels, got {predicted_phys.shape} and {target_phys.shape}")
    mean = torch.as_tensor(mean, dtype=predicted_phys.dtype, device=predicted_phys.device).reshape(1, 1, 7)
    std = torch.as_tensor(std, dtype=predicted_phys.dtype, device=predicted_phys.device).reshape(1, 1, 7)
    if (mean.shape[-1] != 7 or std.shape[-1] != 7 or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or torch.any(std <= 0)
            or not torch.isfinite(predicted_phys).all() or not torch.isfinite(target_phys).all()):
        raise ValueError("Physical state normalization must contain seven finite means and scales")
    return F.mse_loss((predicted_phys - mean) / std, (target_phys - mean) / std)


def field_loss(pred, target, relative=False):
    return relative_l2(pred, target) if relative else F.mse_loss(pred, target)


def grouped_field_loss(pred, target, weights=None, relative=False):
    """Return independently averaged normalized MSEs for u, grad(u), and omega."""
    if pred.shape != target.shape or pred.shape[-1] != 15:
        raise ValueError(f"Expected matching field tensors with 15 channels, got {pred.shape} and {target.shape}")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Field prediction/target contains NaN or infinite values")
    groups = {"velocity": slice(0, 3), "gradient": slice(3, 12), "vorticity": slice(12, 15)}
    terms = {name: F.mse_loss(pred[..., channels], target[..., channels])
             for name, channels in groups.items()}
    # `weights` and `relative` remain accepted for old callers, but intentionally
    # do not alter the normalized-MSE objective or task balancing.
    aggregate = sum(terms.values())
    return aggregate, terms


def uncertainty_weighted_mse(loss, log_variance):
    """Gaussian homoscedastic regression objective for one MSE task."""
    if not torch.isfinite(loss).all() or not torch.isfinite(log_variance).all():
        raise ValueError("Uncertainty-weighted loss or log variance is non-finite")
    return 0.5 * torch.exp(-log_variance) * loss + 0.5 * log_variance


def uncertainty_weighted_sum(losses, log_variances):
    terms = [uncertainty_weighted_mse(losses[name], log_variances[name])
             for name in losses if losses[name] is not None]
    if not terms:
        raise ValueError("At least one regression loss must be enabled")
    return sum(terms)


def rollout_loss(predicted, target, step_weights=None):
    terms = [(predicted[:, k] - target[:, k]).square().mean() for k in range(predicted.shape[1])]
    weights = [1.0] * len(terms) if step_weights is None else step_weights
    return sum(float(weights[k]) * term for k, term in enumerate(terms)) / max(sum(float(w) for w in weights), 1e-12)


def normalized_rollout_loss(predicted, target, mean, std, step_weights=None):
    mean = torch.as_tensor(mean, dtype=predicted.dtype, device=predicted.device)
    std = torch.as_tensor(std, dtype=predicted.dtype, device=predicted.device).clamp_min(1e-8)
    return rollout_loss((predicted - mean) / std, (target - mean) / std, step_weights)


def homoscedastic_multitask(state_term, field_term, log_state_var, log_field_var):
    return uncertainty_weighted_mse(state_term, log_state_var) + uncertainty_weighted_mse(field_term, log_field_var)


def combined_loss(state_term=None, field_term=None, rollout_term=None, weights=None,
                  log_state_var=None, log_field_var=None, homoscedastic=False):
    weights = weights or {}
    if homoscedastic and log_state_var is not None and log_field_var is not None:
        terms = []
        if state_term is not None:
            terms.append(uncertainty_weighted_mse(state_term, log_state_var))
        if field_term is not None:
            terms.append(uncertainty_weighted_mse(field_term, log_field_var))
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
