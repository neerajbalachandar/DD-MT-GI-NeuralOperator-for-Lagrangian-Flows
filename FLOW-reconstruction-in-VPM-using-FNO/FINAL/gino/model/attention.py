import math
import torch
import torch.nn.functional as F


def cross_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, dropout=None) -> torch.Tensor:
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, values)
