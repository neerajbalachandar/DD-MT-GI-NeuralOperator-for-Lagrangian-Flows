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


def test_different_particle_and_field_queries_share_single_encoding():
    pytest.importorskip("neuralop")
    from gino.data.normalization import NormalizationStats
    from gino.dynamics.state_transition import predict_next_state
    from gino.model.gino import GINOSharedLatent, build_latent_grid
    cfg = {"hidden_channels": 8, "latent_res": 4, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1}
    model = GINOSharedLatent(3, 7, 12, 2, cfg)
    calls = {"encoder": 0, "fno": 0}
    hook_a = model.encoder.register_forward_hook(lambda *args: calls.__setitem__("encoder", calls["encoder"] + 1))
    hook_b = model.fno.register_forward_hook(lambda *args: calls.__setitem__("fno", calls["fno"] + 1))
    batch = {"input_geom": torch.rand(1, 5, 3), "particle_queries": torch.rand(1, 5, 3),
             "output_queries": torch.rand(1, 7, 3), "x": torch.rand(1, 5, 3),
             "global_params": torch.rand(1, 2), "state_phys": torch.zeros(1, 5, 7), "dt": 0.01}
    stats = NormalizationStats(torch.zeros(3), torch.ones(3), torch.zeros(7), torch.ones(7),
                               torch.zeros(12), torch.ones(12), torch.zeros(3), torch.ones(3))
    result = predict_next_state(model, batch, build_latent_grid(4, "cpu"), stats)
    hook_a.remove(); hook_b.remove()
    assert result["predicted_state"].shape == (1, 5, 7)
    assert result["field_phys"].shape == (1, 7, 12)
    assert calls == {"encoder": 1, "fno": 1}


def test_m0_forward_matches_explicit_shared_latent_decoders():
    pytest.importorskip("neuralop")
    from gino.model.gino import GINOSharedLatent, build_latent_grid
    cfg = {"hidden_channels": 8, "latent_res": 4, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1}
    model = GINOSharedLatent(3, 7, 12, 2, cfg).eval()
    geom, latent, queries = torch.rand(1, 5, 3), build_latent_grid(4, "cpu"), torch.rand(1, 6, 3)
    x, global_params = torch.rand(1, 5, 3), torch.rand(1, 2)
    with torch.inference_mode():
        forward = model(geom, latent, queries, x, global_params)
        encoded = model.encode_process(geom, latent, x, global_params)
        explicit = model.decode_particle(encoded), model.decode_field(encoded, queries)
    torch.testing.assert_close(forward[0], explicit[0])
    torch.testing.assert_close(forward[1], explicit[1])


@pytest.mark.parametrize("switch", [
    {"use_attention": False},
    {"use_skip": False},
    {"use_global_conditioning": False},
    {"use_task_adapters": True},
    {"use_attention": False, "use_task_adapters": True},
])
def test_architecture_switches_execute_independently(switch):
    pytest.importorskip("neuralop")
    from gino.model.gino import GINOSharedLatent, build_latent_grid
    cfg = {"hidden_channels": 8, "latent_res": 4, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1,
           "use_attention": True, "use_skip": True, "use_global_conditioning": True,
           "use_task_adapters": False}
    cfg.update(switch)
    model = GINOSharedLatent(3, 7, 12, 2, cfg)
    delta, field = model(torch.rand(1, 5, 3), build_latent_grid(4, "cpu"),
                         torch.rand(1, 6, 3), torch.rand(1, 5, 3), torch.rand(1, 2))
    assert delta.shape == (1, 5, 7)
    assert field.shape == (1, 6, 12)
