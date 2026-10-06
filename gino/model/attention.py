import torch
import torch.nn.functional as F


def cross_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, dropout=None) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        queries, keys, values,
        dropout_p=0.0,
        is_causal=False,
    )
