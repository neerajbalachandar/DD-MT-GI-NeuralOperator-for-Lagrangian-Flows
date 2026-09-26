import inspect


def build_gno_block(in_channels: int, out_channels: int, radius: float):
    try:
        from neuralop.layers.gno_block import GNOBlock
    except ImportError as exc:
        raise ImportError("The model requires neuraloperator (neuralop).") from exc
    kwargs = dict(in_channels=in_channels, out_channels=out_channels, coord_dim=3, radius=float(radius),
                  transform_type="linear", reduction="mean", pos_embedding_type="transformer",
                  pos_embedding_channels=12, channel_mlp_layers=[out_channels] * 3)
    accepted = set(inspect.signature(GNOBlock.__init__).parameters)
    for key in ("use_torch_scatter_reduce", "use_open3d_neighbor_search"):
        if key in accepted:
            kwargs[key] = False
    return GNOBlock(**{k: v for k, v in kwargs.items() if k in accepted})
