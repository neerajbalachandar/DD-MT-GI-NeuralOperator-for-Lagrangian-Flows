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
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.meta_npz = str(meta_npz)
        self._load(model_path, meta_npz)
        self.model.eval()

    def _load(self, model_path: str, meta_npz: str) -> None:
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state and "config" in state:
            self._load_gino_checkpoint(state)
            return

        ds = np.load(meta_npz, allow_pickle=True)
        self.checkpoint_kind = "legacy_gno"
        self.feature_names_all = [str(x) for x in ds["feature_names"].tolist()]
        self.feature_names = list(self.feature_names_all)
        self.target_names = [str(x) for x in ds["target_names"].tolist()]
        self.in_mean = ds["in_mean"].astype(np.float32).reshape(-1)
        self.in_std = np.maximum(ds["in_std"].astype(np.float32).reshape(-1), 1e-8)
        self.out_mean = ds["out_mean"].astype(np.float32).reshape(-1)
        self.out_std = np.maximum(ds["out_std"].astype(np.float32).reshape(-1), 1e-8)
        self.model = _UGradUGNO(
            in_dim=len(self.feature_names),
            out_dim=len(self.target_names),
            hidden=96,
            n_layers=3,
            radius=0.12,
        ).to(self.device)
        self.model.load_state_dict(state)

    def _load_gino_checkpoint(self, checkpoint: dict) -> None:
        from neuralop.models import GINO
        import inspect

        cfg = dict(checkpoint["config"])
        self.checkpoint_kind = "gino"
        self.feature_names_all = [str(x) for x in checkpoint.get("feature_names_all", checkpoint["feature_names"])]
        self.feature_names = [str(x) for x in checkpoint["feature_names"]]
        self.target_names = [str(x) for x in checkpoint["target_names"]]
        self.active_input_feature_indices = [int(i) for i in checkpoint.get("active_input_feature_indices", [])]
        if not self.active_input_feature_indices:
            self.active_input_feature_indices = [self.feature_names_all.index(name) for name in self.feature_names]
        self.in_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32).reshape(-1)
        self.in_std = np.maximum(np.asarray(checkpoint["input_std"], dtype=np.float32).reshape(-1), 1e-8)
        self.out_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(-1)
        self.out_std = np.maximum(np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(-1), 1e-8)
        self.coord_min = np.asarray(checkpoint["coord_min"], dtype=np.float32).reshape(3)
        self.coord_span = np.maximum(np.asarray(checkpoint["coord_span"], dtype=np.float32).reshape(3), 1e-8)
        self.latent_res = int(cfg.get("latent_res", 8))
        kwargs = dict(
            in_channels=len(self.feature_names),
            out_channels=len(self.target_names),
            gno_coord_dim=3,
            in_gno_radius=cfg.get("in_gno_radius", 0.10),
            out_gno_radius=cfg.get("out_gno_radius", 0.12),
            in_gno_transform_type=cfg.get("in_gno_transform_type", "nonlinear_kernelonly"),
            out_gno_transform_type=cfg.get("out_gno_transform_type", "linear"),
            gno_embed_channels=cfg.get("gno_embed_channels", 24),
            gno_use_open3d=cfg.get("gno_use_open3d", False),
            gno_use_torch_scatter=cfg.get("gno_use_torch_scatter", False),
            fno_n_modes=tuple(cfg.get("fno_n_modes", (4, 4, 4))),
            fno_hidden_channels=cfg.get("fno_hidden_channels", 24),
            fno_n_layers=cfg.get("fno_n_layers", 3),
            projection_channel_ratio=cfg.get("projection_channel_ratio", 2),
        )
        accepted = set(inspect.signature(GINO).parameters)
        self.model = GINO(**{k: v for k, v in kwargs.items() if k in accepted}).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def required_feature_names(self):
        return list(self.feature_names_all)

    def _latent_queries(self):
        line = torch.linspace(0.0, 1.0, self.latent_res, dtype=torch.float32, device=self.device)
        xx, yy, zz = torch.meshgrid(line, line, line, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)

    def _coords_from_features(self, x: np.ndarray) -> np.ndarray:
        names = self.feature_names_all if x.shape[1] == len(self.feature_names_all) else self.feature_names
        try:
            cols = [names.index("x"), names.index("y"), names.index("z")]
        except ValueError:
            cols = [0, 1, 2]
        return x[:, cols].astype(np.float32)

    def predict(self, xfeatures: np.ndarray) -> np.ndarray:
        x = np.asarray(xfeatures, dtype=np.float32)
        if self.checkpoint_kind == "gino":
            if x.shape[1] == len(self.feature_names_all):
                x_active = x[:, self.active_input_feature_indices]
            elif x.shape[1] == len(self.feature_names):
                x_active = x
            else:
                raise ValueError(
                    f"GINO predictor expected {len(self.feature_names_all)} full features "
                    f"or {len(self.feature_names)} active features, got {x.shape[1]}"
                )
            xyz = self._coords_from_features(x)
            geom = np.clip((xyz - self.coord_min[None, :]) / self.coord_span[None, :], 0.0, 1.0).astype(np.float32)
            xn = np.clip((x_active - self.in_mean[None, :]) / self.in_std[None, :], -8.0, 8.0).astype(np.float32)
            with torch.no_grad():
                yn = self.model(
                    input_geom=torch.from_numpy(geom).unsqueeze(0).to(self.device),
                    latent_queries=self._latent_queries(),
                    output_queries=torch.from_numpy(geom).unsqueeze(0).to(self.device),
                    x=torch.from_numpy(xn).unsqueeze(0).to(self.device),
                ).squeeze(0).cpu().numpy()
        else:
            xn = (x - self.in_mean) / self.in_std
            xt = torch.from_numpy(xn).to(self.device)
            with torch.no_grad():
                yn = self.model(xt).cpu().numpy()
        y = yn * self.out_std + self.out_mean
        return y.astype(np.float32)
