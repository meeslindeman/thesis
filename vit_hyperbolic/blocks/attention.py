import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from vit_hyperbolic.lorentz import Lorentz
from layers.LLinear import LorentzFullyConnected


HeadMode = Literal["full_midpoint", "split_concat"]
ConcatMode = Literal["direct", "logradius"]


class LorentzMHA(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            curvature: float = 0.1, 
            num_heads: int = 1, 
            compute_scores: str = "lorentz_inner", 
            *,
            head_mode: HeadMode = "full_midpoint",
            concat_mode: ConcatMode = "direct",
            out_dim: int | None = None,
            reset_params: str = "lorentz_kaiming",
            a_default: float = 0.0,
            attn_mask: torch.Tensor | None = None,
            dropout_rate: float = 0.0
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

        self.head_mode = head_mode

        if head_mode == "full_midpoint":
            # Each head has full dimension, fused by Lorentz midpoint
            self.head_fusion = "midpoint"
            self.head_spatial_dim = self.spatial_dim 
        
        elif head_mode == "split_concat":
            # Split only spatial dimensions among heads, fused by concatenation
            if self.spatial_dim % self.num_heads != 0:
                raise ValueError(f"spatial_dim {self.spatial_dim} not divisible by num_heads {self.num_heads}")
            self.head_spatial_dim = self.spatial_dim // self.num_heads

            if concat_mode not in ("direct", "logradius"):
                raise ValueError(f"Unknown concat_mode: {concat_mode}")
            self.head_fusion = f"concat_{concat_mode}"
        
        else:
            raise ValueError("head_mode must be one of: 'full_midpoint', 'split_concat'")

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
                activation=nn.Identity()
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
                activation=nn.Identity()
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
                activation=nn.Identity()
            )
            for _ in range(self.num_heads)
        ])

        self.dropout = nn.Dropout(dropout_rate)

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
            activation=nn.Identity()
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
        # Recompute time coordinate to ensure output is on the manifold after concatenating spatial components.
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, N, 1+head_spatial_dim] -> [B, N, H, 1+head_spatial_dim]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, head_spatial_dim]

        space_cat = space.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        # t'² = Σt_i² + (H-1)/k
        time_sq_sum = (time ** 2).sum(dim=2)                # [B, N, H, 1] -> [B, N, 1]
        time_sq_sum = time_sq_sum + (H - 1) / k  # [B, N, 1]
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

        # scale spatial components: ũi = s * ui
        space_scaled = (s * space)  # [B, N, H, head_spatial_dim]
        space_cat = space_scaled.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        # recompute time coordinate: t' = sqrt(1/K + s² * Σ(ti² + 1/K))
        time_sq_plus_invk = (time ** 2) + (1.0 / k)  # [B, N, H, 1]
        time_sum = time_sq_plus_invk.sum(dim=2)      # [B, N, H, 1] -> [B, N, 1]
        t_prime = torch.sqrt((1.0 / k + s ** 2 * time_sum).clamp_min(1e-9))  # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)  # [B, N, 1+spatial_dim]
        return out

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        if D != self.lorentz_dim:
            raise ValueError(f"Expected last dim {self.lorentz_dim}, got {D}")

        # Per-head Q/K/V projections
        # Each head maps [B, N, D] -> q_h/k_h/v_h: [B, N, D_head]
        q = torch.stack([W(x) for W in self.W_q], dim=1)  # [B, H, N, D_head]
        k = torch.stack([W(x) for W in self.W_k], dim=1)  # [B, H, N, D_head]
        v = torch.stack([W(x) for W in self.W_v], dim=1)  # [B, H, N, D_head]

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
        # [B, H, N, D_head]
        out_heads = self.manifold.lorentz_midpoint(v, attn)  # [B, H, N, D_head]
        
        # Fuse heads into a single Lorentz representation of full dim D = 1 + d
        if self.num_heads == 1:
            # [B, 1, N, D_head] -> [B, N, D_head]
            out = out_heads.squeeze(1)  
        else:
            if self.head_fusion == "midpoint":
                # out_heads: [B, H, N, D] -> midpoint over H for each (B,N)
                # Permute to [B, N, H, D], then flatten (B*N) for manifold call.
                out = out_heads.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
                out = out.view(B * N, self.num_heads, self.lorentz_dim)  # [B*N, H, D]
                out = self.manifold.lorentz_midpoint(out)  # [B*N, D]
                out = out.view(B, N, self.lorentz_dim)  # [B, N, D]

            elif self.head_fusion == "concat_direct":
                # out_heads: [B, H, N, D_head] -> [B, N, D]
                out = self._concat_heads_direct(out_heads)

            elif self.head_fusion == "concat_logradius":
                # out_heads: [B, H, N, D_head] -> [B, N, D]
                out = self._concat_heads_log_radius(out_heads)
            
            else:
                raise ValueError(f"Unknown head_fusion mode: {self.head_fusion}")


        # Output projection in Lorentz space: [B, N, D] -> [B, N, 1 + out_dim]
        out = self.W_o(out) 
        
        return out 