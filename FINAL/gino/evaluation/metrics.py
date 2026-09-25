import numpy as np
import torch


def mse(pred, target):
    if torch.is_tensor(pred):
        return torch.mean((pred - target) ** 2)
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


def relative_l2(pred, target, eps=1e-8):
    if torch.is_tensor(pred):
        return torch.linalg.vector_norm((pred - target).reshape(-1)) / torch.linalg.vector_norm(target.reshape(-1)).clamp_min(eps)
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(target)) / max(np.linalg.norm(target), eps))


def state_component_metrics(pred, target):
    names, chunks = ("position", "circulation", "sigma"), (slice(0, 3), slice(3, 6), slice(6, 7))
    out = {}
    for name, sl in zip(names, chunks):
        err = pred[..., sl] - target[..., sl]
        out[f"{name}_mse"] = float(np.mean(np.square(err)))
        out[f"{name}_relative_l2"] = relative_l2(pred[..., sl], target[..., sl])
    out["mse"] = mse(pred, target)
    out["relative_l2"] = relative_l2(pred, target)
    return out
