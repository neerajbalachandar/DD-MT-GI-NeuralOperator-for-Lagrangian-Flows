from torch import nn


def make_field_decoder(hidden: int, query_dim: int, width: int, layers: int, out_channels: int) -> nn.Sequential:
    modules = []
    for i in range(max(int(layers), 1)):
        modules.extend((nn.Linear(hidden + query_dim if i == 0 else width, width), nn.GELU()))
    modules.append(nn.Linear(width, out_channels))
    return nn.Sequential(*modules)


def make_particle_decoder(hidden: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_channels))
