import torch


def pushforward_step(predict_step, state, target_next, rebuild_input):
    """Predict from truth once, then expose a detached model-generated next input."""
    prediction = predict_step(state)
    generated_input = rebuild_input(prediction.detach())
    return prediction, generated_input, target_next


def pushforward_sequence(predict_step, initial_state, targets, rebuild_input):
    current = initial_state
    predictions = []
    for target in targets:
        pred = predict_step(current)
        predictions.append(pred)
        current = rebuild_input(pred.detach())
    return torch.stack(predictions, dim=1)
