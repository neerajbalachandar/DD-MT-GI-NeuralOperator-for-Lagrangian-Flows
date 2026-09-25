import pytest
import torch


def test_positional_encoding_shape():
    from gino.model.positional_encoding import fourier_positional_encoding
    coords = torch.zeros(2, 5, 3)
    assert fourier_positional_encoding(coords, 4).shape == (2, 5, 27)


def test_gino_forward_shapes_and_checkpoint_roundtrip():
    pytest.importorskip("neuralop")
    from gino.model.gino import GINOSharedLatent, build_latent_grid
    cfg = {"hidden_channels": 8, "latent_res": 4, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1}
    model = GINOSharedLatent(3, 7, 12, 2, cfg)
    geom = torch.rand(1, 5, 3)
    y = torch.rand(1, 6, 3)
    out = model(geom, build_latent_grid(4, "cpu"), y, torch.rand(1, 5, 3), torch.rand(1, 2))
    assert out[0].shape == (1, 5, 7)
    assert out[1].shape == (1, 6, 12)
    clone = GINOSharedLatent(3, 7, 12, 2, cfg)
    clone.load_state_dict(model.state_dict())
