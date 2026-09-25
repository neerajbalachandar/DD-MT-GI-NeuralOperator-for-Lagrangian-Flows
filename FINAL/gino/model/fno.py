def build_fno(hidden: int, modes: int, layers: int, latent_res: int):
    try:
        from neuralop.models import FNO
    except ImportError as exc:
        raise ImportError("The model requires neuraloperator (neuralop).") from exc
    modes = min(int(modes), max(int(latent_res) // 2, 1))
    return FNO(n_modes=(modes, modes, modes), in_channels=hidden, out_channels=hidden,
               hidden_channels=hidden, n_layers=int(layers), positional_embedding=None)
