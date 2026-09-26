import torch


def fourier_positional_encoding(coords: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    if int(num_frequencies) <= 0:
        return coords
    frequencies = (2.0 ** torch.arange(int(num_frequencies), device=coords.device, dtype=coords.dtype)).view(1, 1, -1)
    angles = coords.unsqueeze(-1) * frequencies * torch.pi
    return torch.cat((coords, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)), dim=-1)
