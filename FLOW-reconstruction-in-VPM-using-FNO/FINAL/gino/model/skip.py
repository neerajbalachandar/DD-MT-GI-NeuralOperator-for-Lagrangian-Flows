import torch


def add_skip(main: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    return main + skip
