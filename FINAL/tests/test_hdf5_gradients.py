import numpy as np

from gino.data.hdf5 import physical_field_values


def test_native_velocity_derivatives_use_physical_nonunit_spacing():
    axes = (np.linspace(-2, 2, 5), np.linspace(-3, 3, 5), np.linspace(1, 9, 5))
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    x, y, z = (xyz[..., i] for i in range(3))
    velocity = np.stack((2*x + 3*y + 4*z, -x + 0.5*y + 0.2*z, 7*x - 2*y + z), axis=-1)
    values = physical_field_values(xyz.reshape(-1, 3), velocity.reshape(-1, 3), shape=(5, 5, 5))
    np.testing.assert_allclose(values[:, 3:12], np.tile([2, -1, 7, 3, 0.5, -2, 4, 0.2, 1], (125, 1)), atol=1e-5)
