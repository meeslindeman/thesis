from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Literal, Optional

from vit_hyperbolic_jax.lorentz import Lorentz
from vit_hyperbolic_jax.layers import LorentzFullyConnected

HeadMode   = Literal["full_midpoint", "split_concat"]
ConcatMode = Literal["direct", "logradius"]

class LorentzMHA(nn.Module):
    """
    Multi-head attention in Lorentz space.

    input_dim: total Lorentz dimension D = 1 + spatial_dim.
    Two head modes:
      - full_midpoint : each head operates at full D, fused via Lorentz midpoint.
      - split_concat  : spatial dims split among heads, fused by concatenation.
    Two score modes:
      - lorentz_inner : attention score = <q,k>_L (negative semi-definite).
      - signed_dist   : attention score = -d_L(q,k)^2.
    """
    input_dim:      int
    curvature:      float = 0.1
    num_heads:      int   = 1
    compute_scores: str   = "lorentz_inner"
    head_mode:      str   = "full_midpoint"
    concat_mode:    str   = "direct"
    out_dim:        Optional[int] = None
    reset_params:   str   = "lorentz_kaiming"
    a_default:      float = 0.0

    def setup(self):
        spatial_dim = self.input_dim - 1

        if self.head_mode == "full_midpoint":
            head_spatial_dim = spatial_dim
        elif self.head_mode == "split_concat":
            if spatial_dim % self.num_heads != 0:
                raise ValueError(
                    f"spatial_dim {spatial_dim} not divisible by num_heads {self.num_heads}"
                )
            if self.concat_mode not in ("direct", "logradius"):
                raise ValueError(f"Unknown concat_mode: {self.concat_mode}")
            head_spatial_dim = spatial_dim // self.num_heads
        else:
            raise ValueError("head_mode must be 'full_midpoint' or 'split_concat'")

        head_lorentz_dim = head_spatial_dim + 1
        out_dim          = self.out_dim if self.out_dim is not None else spatial_dim

        # Store for use in __call__
        self._spatial_dim      = spatial_dim
        self._head_spatial_dim = head_spatial_dim
        self._head_lorentz_dim = head_lorentz_dim
        self._out_dim          = out_dim
        self._scale            = 1.0 / math.sqrt(head_spatial_dim)

        # Per-head Q/K/V projection layers
        fc_kwargs = dict(
            in_features  = self.input_dim,
            out_features = head_lorentz_dim,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
        )
        self.W_q = [LorentzFullyConnected(**fc_kwargs, name=f"W_q_{i}") for i in range(self.num_heads)]
        self.W_k = [LorentzFullyConnected(**fc_kwargs, name=f"W_k_{i}") for i in range(self.num_heads)]
        self.W_v = [LorentzFullyConnected(**fc_kwargs, name=f"W_v_{i}") for i in range(self.num_heads)]

        # Output projection
        self.W_o = LorentzFullyConnected(
            in_features  = spatial_dim + 1,
            out_features = out_dim + 1,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "W_o",
        )

        # Precompute log-radius scale for concat_logradius
        if self.head_mode == "split_concat" and self.concat_mode == "logradius":
            from scipy.special import digamma
            n  = float(spatial_dim)
            ni = float(head_spatial_dim)
            self._log_radius_scale = math.exp(0.5 * (digamma(n / 2.0) - digamma(ni / 2.0)))
        else:
            self._log_radius_scale = None

    @staticmethod
    def _lorentz_inner(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Batched Lorentz inner product matrix. x,y: [B, H, N, D_head] -> [B, H, N, N]."""
        x_t, x_x = x[..., :1], x[..., 1:]
        y_t, y_x = y[..., :1], y[..., 1:]
        time  = jnp.matmul(x_t, y_t.swapaxes(-1, -2))  # [B, H, N, N]
        space = jnp.matmul(x_x, y_x.swapaxes(-1, -2))  # [B, H, N, N]
        return -time + space

    def _lorentz_sqdist(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        lp = self._lorentz_inner(x, y)
        k  = float(self.curvature)
        return -2.0 * k - 2.0 * lp

    def _concat_heads_direct(self, x: jnp.ndarray) -> jnp.ndarray:
        """Qu et al. direct concatenation with time recomputation."""
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k      = float(self.curvature)
        x      = x.transpose(0, 2, 1, 3)                            # [B, N, H, Dh]
        time   = x[..., :1]                                         # [B, N, H, 1]
        space  = x[..., 1:]                                         # [B, N, H, d_head]
        space_cat   = space.reshape(B, N, H * d_head)               # [B, N, spatial_dim]
        time_sq_sum = (time ** 2).sum(axis=2) + (H - 1) / k         # [B, N, 1]
        t_prime     = jnp.sqrt(jnp.clip(time_sq_sum, a_min=1e-9))   # [B, N, 1]
        return jnp.concatenate([t_prime, space_cat], axis=-1)

    def _concat_heads_log_radius(self, x: jnp.ndarray) -> jnp.ndarray:
        """Shi et al. log-radius concatenation."""
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k      = float(self.curvature)
        s      = self._log_radius_scale
        x      = x.transpose(0, 2, 1, 3)                            # [B, N, H, Dh]
        time   = x[..., :1]                                         # [B, N, H, 1]
        space  = x[..., 1:]                                         # [B, N, H, d_head]
        space_cat  = (s * space).reshape(B, N, H * d_head)          # [B, N, spatial_dim]
        time_sum   = ((time ** 2) + (1.0 / k)).sum(axis=2)          # [B, N, 1]
        t_prime    = jnp.sqrt(jnp.clip(1.0 / k + s ** 2 * time_sum, a_min=1e-9))
        return jnp.concatenate([t_prime, space_cat], axis=-1)

    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        x: [B, N, D]
        attn_mask: optional bool array [N, N] or [B, N, N]  (True = attend, False = mask out)
        Returns: [B, N, 1 + out_dim]
        """
        B, N, D = x.shape
        manifold = Lorentz(k_val=self.curvature)

        # Per-head Q/K/V projections: [B, H, N, D_head]
        q = jnp.stack([W(x) for W in self.W_q], axis=1)
        k = jnp.stack([W(x) for W in self.W_k], axis=1)
        v = jnp.stack([W(x) for W in self.W_v], axis=1)

        # Attention scores [B, H, N, N]
        if self.compute_scores == "lorentz_inner":
            scores = self._lorentz_inner(q, k) * self._scale
        elif self.compute_scores == "signed_dist":
            scores = -self._lorentz_sqdist(q, k) * self._scale
        else:
            raise ValueError(f"Unknown compute_scores: {self.compute_scores}")

        # Optional mask (True = keep, False = mask out)
        if attn_mask is not None:
            attn_mask = attn_mask.astype(jnp.bool_)
            if attn_mask.ndim == 2:
                m = attn_mask[jnp.newaxis, jnp.newaxis, :, :]       # [1, 1, N, N]
            elif attn_mask.ndim == 3:
                m = attn_mask[:, jnp.newaxis, :, :]                 # [B, 1, N, N]
            else:
                raise ValueError(f"attn_mask must be [N,N] or [B,N,N], got {attn_mask.shape}")
            scores = jnp.where(m, scores, jnp.full_like(scores, -1e9))

        attn = jax.nn.softmax(scores, axis=-1)   # [B, H, N, N]

        # Lorentz-valued aggregation: [B, H, N, D_head]
        out_heads = manifold.lorentz_midpoint(v, attn)

        # Fuse heads
        if self.num_heads == 1:
            out = out_heads[:, 0, :, :]   # [B, N, D_head]
        else:
            if self.head_mode == "full_midpoint":
                # [B, H, N, D] -> midpoint over heads
                out = out_heads.transpose(0, 2, 1, 3)              # [B, N, H, D]
                out = out.reshape(B * N, self.num_heads, self.input_dim)
                out = manifold.lorentz_midpoint(out)               # [B*N, D]
                out = out.reshape(B, N, self.input_dim)

            elif self.head_mode == "split_concat":
                if self.concat_mode == "direct":
                    out = self._concat_heads_direct(out_heads)     # [B, N, D]
                else:
                    out = self._concat_heads_log_radius(out_heads) # [B, N, D]
            else:
                raise ValueError(f"Unknown head_mode: {self.head_mode}")

        # Output projection: [B, N, D] -> [B, N, 1 + out_dim]
        return self.W_o(out)
