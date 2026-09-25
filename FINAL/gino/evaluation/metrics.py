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
        squared = np.sum(np.square(err), axis=-1) if name != "sigma" else np.square(err[..., 0])
        out[f"{name}_mse"] = float(np.mean(squared))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(squared)))
        out[f"{name}_relative_l2"] = relative_l2(pred[..., sl], target[..., sl])
    out["mse"] = mse(pred, target)
    out["relative_l2"] = relative_l2(pred, target)
    return out


def normalized_state_component_metrics(pred, target, mean, std):
    pred, target = np.asarray(pred), np.asarray(target)
    mean, std = np.asarray(mean).reshape(-1)[:7], np.maximum(np.asarray(std).reshape(-1)[:7], 1e-8)
    pred_n, target_n = (pred - mean) / std, (target - mean) / std
    out = {}
    for name, sl in (("position", slice(0, 3)), ("circulation", slice(3, 6)), ("sigma", slice(6, 7))):
        err = pred_n[..., sl] - target_n[..., sl]
        squared = np.sum(err ** 2, axis=-1) if name != "sigma" else err[..., 0] ** 2
        out[f"{name}_normalized_rmse"] = float(np.sqrt(np.mean(squared)))
        out[f"{name}_normalized_relative_l2"] = relative_l2(pred_n[..., sl], target_n[..., sl])
    return out
