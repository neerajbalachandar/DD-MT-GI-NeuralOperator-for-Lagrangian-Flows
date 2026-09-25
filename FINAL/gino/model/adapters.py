from torch import nn


class TaskAdapters(nn.Module):
    """Optional task-specific maps; the baseline does not instantiate these."""
    def __init__(self, channels: int):
        super().__init__()
        self.field = nn.Linear(channels, channels)
        self.particle = nn.Linear(channels, channels)

    def forward(self, latent):
        return self.particle(latent), self.field(latent)
