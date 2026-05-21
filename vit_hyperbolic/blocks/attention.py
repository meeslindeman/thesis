import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.linear import LorentzFullyConnected


ConcatMode = Literal["direct", "logradius"]


class LorentzMHA(nn.Module):
    def __init__(
            self,
            input_dim: int,
            curvature: float = 1.0,
            num_heads: int = 1,
            compute_scores: str = "lorentz_inner",
            *,
            concat_mode: ConcatMode = "direct",
            out_dim: int | None = None,
            reset_params: str = "lorentz_kaiming",
            a_default: float = 0.0,
            attn_mask: torch.Tensor | None = None,
            attn_dropout_rate: float = 0.0,
            use_weight_norm: bool = False
        ) -> None:
        super().__init__()

        # Input x is Lorentz-valued ([B, N, D])
        # D = 1 + spatial_dim
        self.lorentz_dim = input_dim        # D
        self.spatial_dim = input_dim - 1    # d
        self.curvature = curvature
        self.num_heads = num_heads
        self.compute_scores = compute_scores
        self.attn_mask = attn_mask

        self.manifold = Lorentz(curvature)

        if self.spatial_dim % self.num_heads != 0:
            raise ValueError(f"spatial_dim {self.spatial_dim} not divisible by num_heads {self.num_heads}")
        self.head_spatial_dim = self.spatial_dim // self.num_heads

        if concat_mode not in ("direct", "logradius"):
            raise ValueError(f"Unknown concat_mode: {concat_mode}")
        self.head_fusion = f"concat_{concat_mode}"

        # Per-head Lorentz dimension: 1 + d_head
        self.head_lorentz_dim = self.head_spatial_dim + 1   # D_head
        self.scale = 1.0 / math.sqrt(self.head_spatial_dim)

        # Define per-head projection layers
        # Each head outputs [B, N, D_head] in Lorentz space
        self.W_q = nn.ModuleList([
            LorentzFullyConnected(
                in_features=self.lorentz_dim,           # D
                out_features=self.head_lorentz_dim,     # D_head
                manifold=self.manifold, 
                reset_params=reset_params, 
                a_default=a_default,
                activation=nn.Identity(),
                use_weight_norm=False
            )
            for _ in range(self.num_heads)
        ])
        self.W_k = nn.ModuleList([
            LorentzFullyConnected(
                in_features=self.lorentz_dim,
                out_features=self.head_lorentz_dim,
                manifold=self.manifold,
                reset_params=reset_params,
                a_default=a_default,
                activation=nn.Identity(),
                use_weight_norm=False
            )
            for _ in range(self.num_heads)
        ])
        self.W_v = nn.ModuleList([
            LorentzFullyConnected(
                in_features=self.lorentz_dim,
                out_features=self.head_lorentz_dim,
                manifold=self.manifold,
                reset_params=reset_params,
                a_default=a_default,
                activation=nn.Identity(),
                use_weight_norm=False
            )
            for _ in range(self.num_heads)
        ])

        self.dropout = nn.Dropout(attn_dropout_rate)

        # After fusion we always return to full Lorentz dimension D = 1 + d
        if out_dim is None:
            out_dim = self.spatial_dim
        self.out_dim = out_dim

        self.W_o = LorentzFullyConnected(
            in_features=self.spatial_dim + 1,   # D
            out_features=self.out_dim + 1,      # 1 + out_dim 
            manifold=self.manifold, 
            reset_params=reset_params,
            a_default=a_default,
            activation=nn.Identity(),
            use_weight_norm=use_weight_norm
        ) 
        
        # Precompute log-radius scale for concat_logradius
        if self.head_fusion == "concat_logradius":
            from torch.special import digamma
            n = self.spatial_dim
            ni = self.head_spatial_dim
            s_val = torch.exp(
                0.5 * (digamma(torch.tensor(n / 2.0)) - digamma(torch.tensor(ni / 2.0)))
            )
            self.register_buffer("log_radius_scale", s_val)
        
    def _batched_project(self, x: torch.Tensor, W_list: nn.ModuleList) -> torch.Tensor:
        """Project x through all heads in a single batched matmul."""
        B, N, _ = x.shape
        # Stack V matrices along output dim: (D_in, H * head_spatial_dim)
        V = torch.cat([W.create_spacelike_vector() for W in W_list], dim=1)
        # Single matmul: (B, N, D_in) @ (D_in, H*head_spatial_dim) -> (B, N, H*head_spatial_dim)
        out_space = x @ V
        # Reshape to (B, N, H, head_spatial_dim) then project to hyperboloid
        out_space = out_space.view(B, N, self.num_heads, self.head_spatial_dim)
        out = self.manifold.projection_space_orthogonal(out_space)  # (B, N, H, D_head)
        return out.permute(0, 2, 1, 3).contiguous()  # (B, H, N, D_head)

    @staticmethod
    def _lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_t, x_x = x[..., :1], x[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]
        y_t, y_x = y[..., :1], y[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]

        time = torch.matmul(x_t, y_t.transpose(-1, -2))    # [B, H, N, 1] @ [B, H, 1, N] -> [B, H, N, N]
        space = torch.matmul(x_x, y_x.transpose(-1, -2))   # [B, H, N, head_spatial_dim] @ [B, H, head_spatial_dim, N] -> [B, H, N, N]

        return -time + space  # [B, H, N, N]

    def _lorentz_sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = self._lorentz_inner(x, y)          # [B, H, N, N]
        k = self.manifold.k()
        d2 = -2.0 * k - 2.0 * lp  # [B, H, N, N]
        return d2

    def _concat_heads_direct(self, x: torch.Tensor) -> torch.Tensor:
        # Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, N, 1+head_spatial_dim] -> [B, N, H, 1+head_spatial_dim]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, head_spatial_dim]

        space_cat = space.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        time_sq_sum = (time ** 2).sum(dim=2) - (H - 1) / k  # [B, N, 1]
        t_prime = torch.sqrt(time_sq_sum.clamp_min(1e-9))   # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)  # [B, N, 1+spatial_dim]
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        # Shi, et al. (https://openreview.net/forum?id=NNnkLi1ALt)
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        s = self.log_radius_scale.to(device=x.device, dtype=x.dtype)  # scalar

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, N, 1+head_spatial_dim] -> [B, N, H, 1+head_spatial_dim]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, head_spatial_dim]

        space_scaled = (s * space)  # [B, N, H, head_spatial_dim]
        space_cat = space_scaled.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        time_sq_plus_invk = (time ** 2) - (1.0 / k)  # [B, N, H, 1]
        time_sum = time_sq_plus_invk.sum(dim=2)      # [B, N, H, 1] -> [B, N, 1]
        t_prime = torch.sqrt((1.0 / k + s ** 2 * time_sum).clamp_min(1e-9))  # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)  # [B, N, 1+spatial_dim]
        return out

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None, return_attn: bool = False) -> torch.Tensor:
        B, _, D = x.shape
        if D != self.lorentz_dim:
            raise ValueError(f"Expected last dim {self.lorentz_dim}, got {D}")

        # Per-head Q/K/V projections via single batched matmul per projection type
        q = self._batched_project(x, self.W_q)  # [B, H, N, D_head]
        k = self._batched_project(x, self.W_k)  # [B, H, N, D_head]
        v = self._batched_project(x, self.W_v)  # [B, H, N, D_head]

        # Attention scores ([B, H, N, N])
        # Scale uses spatial head dim
        if self.compute_scores == "lorentz_inner":
            scores = self._lorentz_inner(q, k) * self.scale
        elif self.compute_scores == "signed_dist":
            d2 = self._lorentz_sqdist(q, k)
            scores = -d2 * self.scale
        else:
            raise ValueError(f"Unknown compute_scores mode: {self.compute_scores}")
        
        # Optional attention mask
        if attn_mask is None:
            attn_mask = self.attn_mask

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                # [N, N] -> [1, 1, N, N]
                m = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                # [B, N, N] -> [B, 1, N, N]
                if attn_mask.shape[0] != B:
                    raise ValueError(f"attn_mask batch dim mismatch: got {attn_mask.shape[0]}, expected {B}")
                m = attn_mask[:, None, :, :]
            else:
                raise ValueError(f"attn_mask must be [N,N] or [B,N,N], got shape {tuple(attn_mask.shape)}")

            # Use a large negative number so softmax ~ 0 on masked entries
            scores = scores.masked_fill(~m, -1e9)

        # Softmax weights [B, H, N, N]
        attn = F.softmax(scores, dim=-1)  
        attn = self.dropout(attn)

        # Lorentz-valued aggregation (per head)
        out_heads = self.manifold.lorentz_centroid(v, attn)  # [B, H, N, D_head]
        
        # Fuse heads into a single Lorentz representation of full dim D = 1 + d
        if self.num_heads == 1:
            out = out_heads.squeeze(1)  # [B, 1, N, D_head] -> [B, N, D_head]
        elif self.head_fusion == "concat_direct":
            out = self._concat_heads_direct(out_heads)   # [B, H, N, D_head] -> [B, N, D]
        elif self.head_fusion == "concat_logradius":
            out = self._concat_heads_log_radius(out_heads)  # [B, H, N, D_head] -> [B, N, D]
        else:
            raise ValueError(f"Unknown head_fusion mode: {self.head_fusion}")

        out = self.W_o(out) # [B, N, D] -> [B, N, 1 + out_dim]

        if return_attn:
            return out, attn
        return out