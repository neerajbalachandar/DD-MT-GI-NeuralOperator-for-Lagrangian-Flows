import numpy as np
import torch

try:
    from neuralop.layers.gno_block import GNOBlock  # noqa: F401
except Exception:
    GNOBlock = None


class _UGradUGNO(torch.nn.Module):
    """Must match notebook architecture used for task1_particle_field_evolution_2.ipynb."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 96, n_layers: int = 3, radius: float = 0.12):
        super().__init__()
        from neuralop.layers.gno_block import GNOBlock

        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
        )
        self.blocks = torch.nn.ModuleList(
            [
                GNOBlock(
                    in_channels=hidden,
                    out_channels=hidden,
                    coord_dim=3,
                    radius=radius,
                    transform_type="linear",
                    reduction="mean",
                    pos_embedding_type="transformer",
                    pos_embedding_channels=12,
                    channel_mlp_layers=[hidden, hidden, hidden],
                    use_torch_scatter_reduce=False,
                    use_open3d_neighbor_search=False,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, :3]
        h = self.enc(x)
        for blk, norm in zip(self.blocks, self.norms):
            u = blk(y=pos, x=pos, f_y=h)
            if u.ndim == 3 and u.shape[0] == 1:
                u = u.squeeze(0)
            h = norm(h + u)
        return self.head(h)


class Task1UGradUPredictor:
    def __init__(self, model_path: str, meta_npz: str, device: str = "cpu"):
        ds = np.load(meta_npz, allow_pickle=True)
        self.feature_names = [str(x) for x in ds["feature_names"].tolist()]
        self.target_names = [str(x) for x in ds["target_names"].tolist()]
        self.in_mean = ds["in_mean"].astype(np.float32)
        self.in_std = ds["in_std"].astype(np.float32)
        self.out_mean = ds["out_mean"].astype(np.float32)
        self.out_std = ds["out_std"].astype(np.float32)

        self.device = torch.device(device)
        self.model = _UGradUGNO(
            in_dim=len(self.feature_names),
            out_dim=len(self.target_names),
            hidden=96,
            n_layers=3,
            radius=0.12,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, xfeatures: np.ndarray) -> np.ndarray:
        x = np.asarray(xfeatures, dtype=np.float32)
        xn = (x - self.in_mean) / self.in_std
        xt = torch.from_numpy(xn).to(self.device)
        with torch.no_grad():
            yn = self.model(xt).cpu().numpy()
        y = yn * self.out_std + self.out_mean
        return y.astype(np.float32)
