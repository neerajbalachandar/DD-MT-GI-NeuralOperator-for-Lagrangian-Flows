import torch


def scheduled_sampling_probability(epoch, max_probability=0.3, ramp_epochs=30):
    return min(float(max_probability), max(0.0, (int(epoch) - 1) / max(float(ramp_epochs), 1) * float(max_probability)))


def choose_predicted_flow_inputs(batch, predicted_velocity, predicted_gradient, probability, generator=None):
    """Preserves the source notebook behavior: probabilistically replace only u and gradU."""
    use_prediction = bool(torch.rand((), device=batch["x"].device, generator=generator) < float(probability))
    if not use_prediction:
        return batch["x"], False
    x = batch["x"].clone()
    names = batch["feature_names"]
    for j, name in enumerate(names):
        if name in ("u_x", "u_y", "u_z"):
            x[..., j] = predicted_velocity[..., ("u_x", "u_y", "u_z").index(name)]
        elif name.startswith("gradU_"):
            i, k = "xyz".index(name[-2]), "xyz".index(name[-1])
            x[..., j] = predicted_gradient[..., i, k]
    return x, True
