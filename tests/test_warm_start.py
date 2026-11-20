import numpy as np
import pytest

from scripts.train import validate_warm_start_checkpoint
from gino.data.field_schema import MODEL_FIELD_TARGET_NAMES, REFERENCE_FIELD_TARGET_NAMES


ARCHITECTURE = {
    "latent_res": 16, "hidden_channels": 128, "fno_layers": 6, "fno_modes": 6,
    "gno_radius": 0.15, "mlp_layers": 3, "mlp_hidden": 128,
    "query_pos_encoding_frequencies": 4, "use_attention": True, "use_skip": True,
    "use_global_conditioning": True, "use_task_adapters": True, "predict_delta_u": True,
}
NORMALIZATION = {name: np.asarray([value], dtype=np.float32) for name, value in {
    "input_mean": 0.0, "input_std": 1.0, "residual_mean": 0.0, "residual_std": 1.0,
    "field_mean": 0.0, "field_std": 1.0, "coord_min": 0.0, "coord_span": 1.0,
}.items()}


def checkpoint():
    return {
        "feature_names": ["u_x"], "target_names": ["dx"],
        "field_target_names": list(REFERENCE_FIELD_TARGET_NAMES),
        "reference_field_target_names": list(REFERENCE_FIELD_TARGET_NAMES),
        "model_field_target_names": list(MODEL_FIELD_TARGET_NAMES),
        "global_condition_channels": ["phase"],
        "config": {"model": dict(ARCHITECTURE)}, "normalization": NORMALIZATION,
        "model_state_dict": {"weight": np.asarray([1.0])},
    }


def test_warm_start_accepts_matching_model_and_normalization_contract():
    state = validate_warm_start_checkpoint(checkpoint(), dict(ARCHITECTURE), ["u_x"], ["dx"],
        list(REFERENCE_FIELD_TARGET_NAMES), ["phase"], NORMALIZATION)
    assert "weight" in state


def test_warm_start_rejects_channel_order_mismatch():
    with pytest.raises(ValueError, match="reference field channel order mismatch"):
        validate_warm_start_checkpoint(checkpoint(), dict(ARCHITECTURE), ["u_x"], ["dx"],
            list(reversed(REFERENCE_FIELD_TARGET_NAMES)), ["phase"], NORMALIZATION)
