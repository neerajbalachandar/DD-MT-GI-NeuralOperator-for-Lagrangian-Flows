import pytest
import torch
import numpy as np


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


def test_canonical_next_state_uses_decoded_particle_velocity_gradient_and_residual():
    from gino.data.normalization import NormalizationStats
    from gino.dynamics.state_transition import predict_next_state

    class FixedDecoder(torch.nn.Module):
        def encode_process(self, *args):
            return None
        def decode_particle(self, _):
            residual = torch.zeros(1, 1, 10)
            residual[..., 0] = 0.5
            residual[..., 3] = 1.0
            return residual
        def decode_field(self, _, query):
            field = torch.zeros(query.shape[0], query.shape[1], 12)
            field[..., 0] = 2.0
            field[..., 3] = 1.0  # dUx/dx
            return field

    state = torch.tensor([[[1., 2., 3., 2., 0., 0., 0.4]]])
    batch = {"input_geom": torch.zeros(1, 1, 3), "particle_queries": torch.zeros(1, 1, 3),
             "output_queries": torch.zeros(1, 2, 3), "x": torch.zeros(1, 1, 4),
             "global_params": torch.zeros(1, 1), "state_phys": state, "next_state_phys": state.clone(), "dt": 0.1}
    stats = NormalizationStats(np.zeros(4), np.ones(4), np.zeros(10), np.ones(10),
                               np.zeros(15), np.ones(15), np.zeros(3), np.ones(3))
    result = predict_next_state(FixedDecoder(), batch, torch.zeros(1, 1, 3), stats)
    torch.testing.assert_close(result["predicted_state"][..., 0], torch.tensor([[1.7]]))
    torch.testing.assert_close(result["predicted_state"][..., 3], torch.tensor([[3.2]]))
    torch.testing.assert_close(result["predicted_state"][..., 6], torch.tensor([[0.4]]))
    assert result["residual_phys_all"].shape[-1] == 10


def test_legacy_checkpoint_uncertainty_keys_migrate_only_expected_parameters():
    pytest.importorskip("neuralop")
    from gino.model.gino import GINOSharedLatent
    cfg = {"hidden_channels": 8, "latent_res": 4, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1}
    source = GINOSharedLatent(4, 10, 15, 1, cfg)
    legacy = {key: value.clone() for key, value in source.state_dict().items()
              if not key.startswith("log_") and torch.is_tensor(value)}
    legacy["log_delta_var"] = torch.tensor(0.2)
    legacy["log_field_var"] = torch.tensor(0.3)
    target = GINOSharedLatent(4, 10, 12, 1, cfg)
    legacy["log_vorticity_var"] = torch.tensor(0.4)
    from gino.model.gino import migrate_legacy_15_channel_state_dict
    from gino.data.field_schema import MODEL_FIELD_TARGET_NAMES, REFERENCE_FIELD_TARGET_NAMES
    migrated = migrate_legacy_15_channel_state_dict(legacy, target, REFERENCE_FIELD_TARGET_NAMES, MODEL_FIELD_TARGET_NAMES)
    final = len(target.field_decoder) - 1
    torch.testing.assert_close(migrated[f"field_decoder.{final}.weight"],
                               legacy[f"field_decoder.{final}.weight"][:12])
    torch.testing.assert_close(migrated[f"field_decoder.{final}.bias"],
                               legacy[f"field_decoder.{final}.bias"][:12])
    assert "log_vorticity_var" not in migrated
    with pytest.warns(RuntimeWarning, match="Initialized missing uncertainty"):
        target.load_state_dict(migrated, strict=True)
    assert target.log_residual_var.item() == pytest.approx(0.2)
    assert target.log_velocity_var.item() == pytest.approx(0.3)
    assert target.log_physical_state_var.item() == 0.0
    assert target.field_decoder[-1].weight.shape[0] == 12
    assert "log_vorticity_var" not in target.state_dict()


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


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sample_grid_axis_order_matches_latent_layout(axis):
    """Regression: grid_sample reads coords as (W, H, D); the latent is (C, x, y, z).

    Build a latent grid exactly as encode_process does (build_latent_grid ->
    reshape(r, r, r, C) -> permute to channels-first) whose single channel equals
    the latent node's coordinate along ``axis``; sampling it at arbitrary queries
    must return that query coordinate (trilinear interpolation of a linear field
    is exact).  Before the fix, x and z were swapped.
    """
    pytest.importorskip("neuralop")
    from gino.model.gino import GINOSharedLatent, build_latent_grid
    cfg = {"hidden_channels": 8, "latent_res": 5, "query_pos_encoding_frequencies": 1,
           "gno_radius": 2., "fno_modes": 2, "fno_layers": 1, "mlp_hidden": 8, "mlp_layers": 1}
    model = GINOSharedLatent(3, 7, 12, 2, cfg)
    r = 5
    latent_coords = build_latent_grid(r, "cpu")[0]                # [r^3, 3] (x, y, z)
    values = latent_coords[:, axis:axis + 1]                      # [r^3, 1]
    grid = values.reshape(r, r, r, -1).permute(3, 0, 1, 2).unsqueeze(0)
    torch.manual_seed(0)
    queries = torch.rand(64, 3)
    sampled = model.sample_grid(grid, queries)                    # [64, 1]
    torch.testing.assert_close(sampled[:, 0], queries[:, axis], atol=1e-5, rtol=0)
