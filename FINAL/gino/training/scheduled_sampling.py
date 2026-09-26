import torch


def scheduled_sampling_probability(epoch, max_probability=0.3, ramp_epochs=30):
    return min(float(max_probability), max(0.0, (int(epoch) - 1) / max(float(ramp_epochs), 1) * float(max_probability)))


def choose_predicted_flow_inputs(batch, predicted_velocity, predicted_gradient, probability, generator=None,
                                 teacher_x=None):
    """Make one Bernoulli choice for one generated input; flow features are normalized."""
    use_prediction = bool(torch.rand((), device=batch["x"].device, generator=generator) < float(probability))
    x = batch["x"].clone()
    names = batch["feature_names"]
    mean = torch.as_tensor(batch.get("input_mean", torch.zeros(x.shape[-1])), dtype=x.dtype, device=x.device).reshape(-1)
    std = torch.as_tensor(batch.get("input_std", torch.ones(x.shape[-1])), dtype=x.dtype, device=x.device).reshape(-1).clamp_min(1e-8)
    for j, name in enumerate(names):
        if name in ("u_x", "u_y", "u_z"):
            if use_prediction:
                value = predicted_velocity[..., ("u_x", "u_y", "u_z").index(name)]
                x[..., j] = (value - mean[j]) / std[j]
            elif teacher_x is not None:
                x[..., j] = teacher_x[..., j]
        elif name.startswith("gradU_"):
            i, k = "xyz".index(name[-2]), "xyz".index(name[-1])
            if use_prediction:
                value = predicted_gradient[..., i, k]
                x[..., j] = (value - mean[j]) / std[j]
            elif teacher_x is not None:
                x[..., j] = teacher_x[..., j]
    return x, use_prediction
