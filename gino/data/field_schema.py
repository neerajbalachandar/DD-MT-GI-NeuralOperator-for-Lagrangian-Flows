"""Reference and model field channel contracts.

The processed archive retains vorticity as a reference diagnostic. The learned
decoder predicts only velocity and its full 3x3 spatial gradient.
"""

REFERENCE_FIELD_TARGET_NAMES = (
    "u_x", "u_y", "u_z",
    "dUx_dx", "dUx_dy", "dUx_dz",
    "dUy_dx", "dUy_dy", "dUy_dz",
    "dUz_dx", "dUz_dy", "dUz_dz",
    "omega_x", "omega_y", "omega_z",
)
MODEL_FIELD_TARGET_NAMES = (
    "u_x", "u_y", "u_z",
    "gradU_xx", "gradU_xy", "gradU_xz",
    "gradU_yx", "gradU_yy", "gradU_yz",
    "gradU_zx", "gradU_zy", "gradU_zz",
)
MODEL_FIELD_CHANNELS = len(MODEL_FIELD_TARGET_NAMES)
REFERENCE_FIELD_CHANNELS = len(REFERENCE_FIELD_TARGET_NAMES)


def validate_field_names(names, expected, label):
    names = tuple(str(name) for name in names)
    if names != tuple(expected):
        raise ValueError(f"{label} channel order mismatch: expected {list(expected)}, got {list(names)}")
    return names


def validate_reference_model_order(reference_names):
    names = validate_field_names(reference_names, REFERENCE_FIELD_TARGET_NAMES, "Reference field")
    # The archived dU<component>_d<coordinate> labels are row-major and map
    # one-to-one to the public gradU_xy model labels.
    return names
