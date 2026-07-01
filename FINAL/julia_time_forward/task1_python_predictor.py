import numpy as np
import torch
import sys

try:
    from neuralop.layers.gno_block import GNOBlock  # noqa: F401
except Exception:
    GNOBlock = None

try:
    from neuralop.layers.neighbor_search import NeighborSearch
except Exception:
    NeighborSearch = None


class EdgeFeatureMessageBlock(torch.nn.Module):
    """Radius graph block matching FINAL/task1_v2_GNO.ipynb checkpoints."""

    def __init__(self, hidden_size: int, neighbor_radius: float, use_open3d_neighbor_search: bool = False, dropout: float = 0.04):
        super().__init__()
        if NeighborSearch is None:
            raise RuntimeError("neuralop.layers.neighbor_search.NeighborSearch is not available")
        self.neighbor_radius = float(neighbor_radius)
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search, return_norm=False)
        edge_feature_size = 2 * int(hidden_size) + 14
        self.message_network = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.normalization = torch.nn.LayerNorm(hidden_size)

    def forward(self, hidden, search_positions, physical_positions, gamma_values, sigma_values):
        neighbors = self.neighbor_search(data=search_positions, queries=search_positions, radius=self.neighbor_radius)
        source_index = neighbors["neighbors_index"].long()
        row_splits = neighbors["neighbors_row_splits"].long()
        neighbor_counts = row_splits[1:] - row_splits[:-1]

        if source_index.numel() == 0:
            return hidden

        target_index = torch.repeat_interleave(
            torch.arange(search_positions.shape[0], device=search_positions.device),
            neighbor_counts,
        )
        source_hidden = hidden[source_index]
        target_hidden = hidden[target_index]
        relative_position = physical_positions[source_index] - physical_positions[target_index]
        distance = torch.linalg.norm(relative_position, dim=-1, keepdim=True)

        sigma_source = sigma_values[source_index].abs().clamp_min(1.0e-8)
        sigma_target = sigma_values[target_index].abs().clamp_min(1.0e-8)
        gamma_source = gamma_values[source_index]
        gamma_target = gamma_values[target_index]

        edge_features = torch.cat(
            [
                source_hidden,
                target_hidden,
                relative_position,
                distance,
                distance / sigma_source,
                distance / sigma_target,
                gamma_source,
                gamma_target,
                sigma_source,
                sigma_target,
            ],
            dim=-1,
        )
        messages = self.message_network(edge_features)
        aggregated = torch.zeros(hidden.shape, device=hidden.device, dtype=messages.dtype)
        aggregated.index_add_(0, target_index, messages)
        aggregated = aggregated / neighbor_counts.to(hidden.dtype).clamp_min(1.0).unsqueeze(-1)
        return self.normalization(hidden + aggregated)


class ParticleVelocityGradientModel(torch.nn.Module):
    """Task-1 edge-feature particle model used by best_task1_v3_model.pt."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        feature_names,
        input_mean,
        input_standard_deviation,
        hidden_size: int = 96,
        graph_layers: int = 2,
        neighbor_radius: float = 0.12,
        dropout: float = 0.04,
        use_physical_edge_features: bool = True,
    ):
        super().__init__()
        self.neighbor_radius = float(neighbor_radius)
        self.feature_names = list(feature_names)
        self.use_physical_edge_features = bool(use_physical_edge_features)
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32).reshape(1, -1))
        self.register_buffer(
            "input_standard_deviation",
            torch.as_tensor(input_standard_deviation, dtype=torch.float32).reshape(1, -1),
        )

        self.sigma_index = self.feature_names.index("sigma") if "sigma" in self.feature_names else -1
        self.gamma_indices = [
            self.feature_names.index(name)
            for name in ["Gamma_x", "Gamma_y", "Gamma_z"]
            if name in self.feature_names
        ]
        if len(self.gamma_indices) != 3:
            self.gamma_indices = []

        self.input_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )

        # Keep deployment device-safe. The torch neighbor search path follows the
        # model tensors onto CUDA; Open3D can force CPU-specific behavior.
        use_open3d = False

        self.graph_blocks = torch.nn.ModuleList(
            [
                EdgeFeatureMessageBlock(
                    hidden_size=hidden_size,
                    neighbor_radius=self.neighbor_radius,
                    use_open3d_neighbor_search=use_open3d,
                    dropout=dropout,
                )
                for _ in range(graph_layers)
            ]
        )
        self.output_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, output_size),
        )

    def physical_feature_values(self, particle_features):
        if not self.use_physical_edge_features:
            return particle_features
        return particle_features * self.input_standard_deviation.to(particle_features.dtype) + self.input_mean.to(particle_features.dtype)

    def forward(self, particle_features):
        search_positions = particle_features[:, :3]
        physical_features = self.physical_feature_values(particle_features)
        physical_positions = physical_features[:, :3]

        if self.sigma_index >= 0:
            sigma_values = physical_features[:, self.sigma_index:self.sigma_index + 1]
        else:
            sigma_values = torch.ones((particle_features.shape[0], 1), device=particle_features.device, dtype=particle_features.dtype)

        if self.gamma_indices:
            gamma_values = physical_features[:, self.gamma_indices]
        else:
            gamma_values = torch.zeros((particle_features.shape[0], 3), device=particle_features.device, dtype=particle_features.dtype)

        hidden = self.input_network(particle_features)
        for graph_block in self.graph_blocks:
            hidden = graph_block(
                hidden=hidden,
                search_positions=search_positions,
                physical_positions=physical_positions,
                gamma_values=gamma_values,
                sigma_values=sigma_values,
            )
        return self.output_network(hidden)


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
    def __init__(self, model_path: str, meta_npz: str, device: str = "auto"):
        self.device = resolve_torch_device(device)
        self.model_path = str(model_path)
        self.meta_npz = str(meta_npz)
        print(f"Task1 predictor device: {self.device}", file=sys.stderr, flush=True)
        self._load(model_path, meta_npz)
        self.model.eval()

    def _load(self, model_path: str, meta_npz: str) -> None:
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state and "model_config" in state:
            self._load_edge_checkpoint(state, meta_npz)
            return

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

    def _load_edge_checkpoint(self, checkpoint: dict, meta_npz: str) -> None:
        cfg = dict(checkpoint["model_config"])
        state = checkpoint["model_state_dict"]
        self.checkpoint_kind = "edge_particle"
        self.feature_names = [str(x) for x in checkpoint["feature_names"]]
        self.feature_names_all = list(self.feature_names)
        self.target_names = [str(x) for x in checkpoint["target_names"]]

        ds = np.load(meta_npz, allow_pickle=True)
        ds_feature_names = [str(x) for x in ds["feature_names"].tolist()]
        ds_target_names = [str(x) for x in ds["target_names"].tolist()]
        self.feature_names_all = ds_feature_names
        self.active_input_feature_indices = [ds_feature_names.index(name) for name in self.feature_names]
        target_indices = [ds_target_names.index(name) for name in self.target_names]

        if "input_mean" in state and "input_standard_deviation" in state:
            self.in_mean = state["input_mean"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            self.in_std = np.maximum(
                state["input_standard_deviation"].detach().cpu().numpy().astype(np.float32).reshape(-1),
                1e-8,
            )
        else:
            input_mean_all = np.asarray(ds["in_mean"], dtype=np.float32).reshape(-1)
            input_std_all = np.maximum(np.asarray(ds["in_std"], dtype=np.float32).reshape(-1), 1e-8)
            self.in_mean = input_mean_all[self.active_input_feature_indices]
            self.in_std = input_std_all[self.active_input_feature_indices]

        target_mean_all = np.asarray(ds["out_mean"], dtype=np.float32).reshape(-1)
        target_std_all = np.maximum(np.asarray(ds["out_std"], dtype=np.float32).reshape(-1), 1e-8)
        self.out_mean = target_mean_all[target_indices]
        self.out_std = target_std_all[target_indices]

        self.model = ParticleVelocityGradientModel(
            input_size=len(self.feature_names),
            output_size=len(self.target_names),
            feature_names=self.feature_names,
            input_mean=self.in_mean,
            input_standard_deviation=self.in_std,
            hidden_size=int(cfg.get("hidden_size", 128)),
            graph_layers=int(cfg.get("graph_layers", 3)),
            neighbor_radius=float(cfg.get("neighbor_radius", 0.12)),
            dropout=float(cfg.get("dropout", 0.06)),
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
        if self.checkpoint_kind in {"gino", "edge_particle"}:
            if x.shape[1] == len(self.feature_names):
                x_active = x
            elif x.shape[1] == len(self.feature_names_all):
                x_active = x[:, self.active_input_feature_indices]
            else:
                raise ValueError(
                    f"{self.checkpoint_kind} predictor expected {len(self.feature_names_all)} full features "
                    f"or {len(self.feature_names)} active features, got {x.shape[1]}"
                )

        if self.checkpoint_kind == "edge_particle":
            xn = np.clip((x_active - self.in_mean[None, :]) / self.in_std[None, :], -8.0, 8.0).astype(np.float32)
            with torch.no_grad():
                yn = self.model(torch.from_numpy(xn).to(self.device)).cpu().numpy()
        elif self.checkpoint_kind == "gino":
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


def resolve_torch_device(requested: str | None) -> torch.device:
    req = str(requested or "auto").strip().lower()
    if req in {"auto", "gpu", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if req.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"Task1 predictor warning: requested device '{requested}' but CUDA is unavailable; using CPU.",
            file=sys.stderr,
            flush=True,
        )
        return torch.device("cpu")

    return torch.device(req)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-csv", required=True, help="Input feature matrix CSV, no header")
    parser.add_argument("--output-csv", required=True, help="Output prediction matrix CSV, no header")
    parser.add_argument("--model", required=True, help="Task1 U/gradU .pt checkpoint")
    parser.add_argument("--meta", required=True, help="particle_ugradu_dataset.npz metadata")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:<id>")
    args = parser.parse_args()

    x = np.loadtxt(args.predict_csv, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    predictor = Task1UGradUPredictor(args.model, args.meta, args.device)
    y = predictor.predict(x)
    np.savetxt(args.output_csv, y, delimiter=",")


if __name__ == "__main__":
    _main()
