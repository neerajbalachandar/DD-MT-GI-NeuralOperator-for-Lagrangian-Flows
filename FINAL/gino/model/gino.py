import torch
import torch.nn.functional as F
from torch import nn

from .attention import cross_attention
from .decoders import make_field_decoder, make_particle_decoder
from .fno import build_fno
from .gno import build_gno_block
from .positional_encoding import fourier_positional_encoding
from .skip import add_skip
from .adapters import TaskAdapters


def build_latent_grid(resolution: int, device) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, int(resolution), dtype=torch.float32, device=device)
    xyz = torch.meshgrid(line, line, line, indexing="ij")
    return torch.stack(xyz, dim=-1).reshape(1, -1, 3)


class GINOSharedLatent(nn.Module):
    """Shared GNO/FNO latent and dual decoder, preserving baseline state keys."""
    def __init__(self, in_channels, delta_channels, field_channels, global_channels, cfg):
        super().__init__()
        hidden = int(cfg["hidden_channels"])
        self.latent_res = int(cfg["latent_res"])
        self.query_pe_freqs = int(cfg["query_pos_encoding_frequencies"])
        self.hidden = hidden
        self.use_attention = bool(cfg.get("use_attention", True))
        self.use_skip = bool(cfg.get("use_skip", True))
        self.use_global_conditioning = bool(cfg.get("use_global_conditioning", True))
        self.use_task_adapters = bool(cfg.get("use_task_adapters", False))

        self.lift = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.global_condition_mlp = nn.Sequential(nn.Linear(global_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.encoder = build_gno_block(hidden, hidden, cfg["gno_radius"])
        self.fno = build_fno(hidden, cfg["fno_modes"], cfg["fno_layers"], self.latent_res)

        query_dim = 3 + 6 * self.query_pe_freqs
        self.particle_pos_enc_proj = nn.Linear(query_dim, hidden)
        self.grid_pos_enc_proj = nn.Linear(query_dim, hidden)
        self.field_decoder = make_field_decoder(hidden, query_dim, int(cfg["mlp_hidden"]), cfg["mlp_layers"], field_channels)
        self.particle_skip_proj = nn.Linear(hidden, hidden)
        self.particle_query_proj = nn.Linear(hidden, hidden)
        self.grid_kv_proj = nn.Linear(hidden, 2 * hidden)
        self.attn_dropout = nn.Dropout(0.0)
        self.attn_norm = nn.LayerNorm(hidden)
        self.delta_fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.delta_head = make_particle_decoder(hidden, delta_channels)
        self.log_delta_var = nn.Parameter(torch.zeros(()))
        self.log_field_var = nn.Parameter(torch.zeros(()))
        if self.use_task_adapters:
            self.task_adapters = TaskAdapters(hidden)

    def apply_gno(self, block, source_coords, query_coords, source_features):
        return block(y=source_coords, x=query_coords, f_y=source_features)

    def encode_process(self, input_geom, latent_queries, x, global_params):
        base_latent = latent_queries[0]
        r = self.latent_res
        grids, flattened, particle_skips = [], [], []
        for b in range(x.shape[0]):
            h_raw = self.lift(x[b])
            latent = self.apply_gno(self.encoder, input_geom[b], base_latent, h_raw)
            if latent.ndim == 3:
                latent = latent.squeeze(0)
            grid = latent.reshape(r, r, r, -1).permute(3, 0, 1, 2).unsqueeze(0)
            if self.use_global_conditioning:
                cond = self.global_condition_mlp(global_params[b]).view(1, -1, 1, 1, 1)
                grid = grid + cond
            processed = self.fno(grid)
            flat = processed.squeeze(0).permute(1, 2, 3, 0).reshape(-1, processed.shape[1])
            grids.append(processed)
            flattened.append(flat)
            particle_skips.append(self.particle_skip_proj(h_raw))
        return base_latent, grids, flattened, particle_skips

    def sample_grid(self, grid, queries):
        q = queries.clamp(0.0, 1.0)
        sample_coords = (q * 2.0 - 1.0).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(grid, sample_coords, align_corners=True, mode="bilinear")
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)

    def delta_from_latents(self, base_latent, flat_latents, particle_skips, input_geom):
        grid_pe = fourier_positional_encoding(base_latent.unsqueeze(0), self.query_pe_freqs).squeeze(0)
        result = []
        for b, (flat, skip) in enumerate(zip(flat_latents, particle_skips)):
            particle_pe = fourier_positional_encoding(input_geom[b].unsqueeze(0), self.query_pe_freqs).squeeze(0)
            skip_features = skip if self.use_skip else torch.zeros_like(skip)
            qbase = skip_features + self.particle_pos_enc_proj(particle_pe)
            q = self.particle_query_proj(qbase)
            kv = self.grid_kv_proj(flat + self.grid_pos_enc_proj(grid_pe))
            k, v = kv[..., :self.hidden], kv[..., self.hidden:]
            if self.use_attention:
                particle_latent = cross_attention(q, k, v, self.attn_dropout)
            else:
                particle_latent = flat.mean(dim=0, keepdim=True).expand_as(q)
            if self.use_task_adapters:
                particle_latent = self.task_adapters.particle(particle_latent)
            if self.use_skip:
                particle_latent = add_skip(particle_latent, q)
            particle_latent = self.attn_norm(particle_latent)
            fused = self.delta_fusion(torch.cat((particle_latent, skip_features), dim=-1))
            result.append(self.delta_head(fused).unsqueeze(0))
        return torch.cat(result, dim=0)

    def forward(self, input_geom, latent_queries, output_queries, x, global_params, batch_dict=None):
        base, grids, flat, skips = self.encode_process(input_geom, latent_queries, x, global_params)
        delta = self.delta_from_latents(base, flat, skips, input_geom)
        field = []
        for b, grid in enumerate(grids):
            q = output_queries[b].clamp(0.0, 1.0)
            sampled = self.sample_grid(grid, q)
            pe = fourier_positional_encoding(q.unsqueeze(0), self.query_pe_freqs).squeeze(0)
            if self.use_task_adapters:
                sampled = self.task_adapters.field(sampled)
            field.append(self.field_decoder(torch.cat((sampled, pe), dim=-1)).unsqueeze(0))
        return delta, torch.cat(field, dim=0)
