from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from vit_hyperbolic_jax.lorentz import Lorentz
from vit_hyperbolic_jax.layers import LorentzFullyConnected
from vit_hyperbolic_jax.attention import LorentzMHA
from vit_hyperbolic_jax.lorentz_utils import LorentzDropout, LorentzLayerNorm


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------

class LorentzPatchEmbeddings(nn.Module):
    """Project image patches to Lorentz space."""
    image_size:   int
    patch_size:   int
    num_channels: int
    hidden_size:  int
    curvature:    float
    reset_params: str   = "lorentz_kaiming"
    a_default:    float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, C, H, W = x.shape
        manifold   = Lorentz(k_val=self.curvature)
        grid_size  = self.image_size // self.patch_size
        num_patches = grid_size * grid_size
        p          = self.patch_size
        patch_dim  = p * p * self.num_channels

        # Rearrange to patches: [B, num_patches, patch_dim]
        x = x.reshape(B, C, grid_size, p, grid_size, p)     # [B, C, Gh, p, Gw, p]
        x = x.transpose(0, 2, 4, 3, 5, 1)                   # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, num_patches, patch_dim)            # [B, N, patch_dim]

        # Lift to hyperboloid
        x_lor = manifold.projection_space_orthogonal(x)     # [B, N, patch_dim+1]

        # Project to hidden size
        projection = LorentzFullyConnected(
            in_features  = patch_dim + 1,
            out_features = self.hidden_size + 1,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "projection",
        )
        return projection(x_lor)  # [B, N, hidden_size+1]


class LorentzEmbeddings(nn.Module):
    """Lorentz patch + positional embeddings with optional CLS token."""
    image_size:   int
    patch_size:   int
    num_channels: int
    hidden_size:  int
    curvature:    float
    final_op:     str
    dropout_rate: float = 0.0
    reset_params: str   = "lorentz_kaiming"
    a_default:    float = 0.0
    init_mode:    str   = "normalized"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        manifold    = Lorentz(k_val=self.curvature)
        grid_size   = self.image_size // self.patch_size
        num_patches = grid_size * grid_size

        patch_emb = LorentzPatchEmbeddings(
            image_size   = self.image_size,
            patch_size   = self.patch_size,
            num_channels = self.num_channels,
            hidden_size  = self.hidden_size,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "patch_embeddings",
        )
        x = patch_emb(x)  # [B, N, D]
        B = x.shape[0]

        if self.final_op == "cls":
            n_tokens = num_patches + 1
            cls_token = self.param(
                "cls_token",
                nn.initializers.normal(),
                (1, 1, self.hidden_size + 1),
            )
            cls = jnp.broadcast_to(cls_token, (B, 1, self.hidden_size + 1))
            x   = jnp.concatenate([cls, x], axis=1)  # [B, 1+N, D]
        else:
            n_tokens = num_patches

        # Learnable positional embeddings (tangent space, mapped to manifold)
        if self.init_mode == "normalized":
            def pos_init(key, shape, dtype=jnp.float32):
                pos = jax.random.normal(key, shape, dtype=dtype)
                return pos / jnp.linalg.norm(pos, axis=-1, keepdims=True)
        elif self.init_mode == "random":
            def pos_init(key, shape, dtype=jnp.float32):
                return jax.random.normal(key, shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid init_mode: {self.init_mode!r}. Must be 'normalized' or 'random'.")

        pos_space = self.param("pos_space", pos_init, (1, n_tokens, self.hidden_size))
        pos_lor   = manifold.expmap0(pos_space)              # [1, n_tokens, D]
        x         = manifold.gyro_add(x, pos_lor)            # [B, n_tokens, D]
        return x


# ---------------------------------------------------------------------------
# Euclidean (fallback) Embeddings
# ---------------------------------------------------------------------------

class PatchEmbeddings(nn.Module):
    image_size:      int
    patch_size:      int
    num_channels:    int
    hidden_size:     int
    patch_embed_type: str = "linear"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, C, H, W = x.shape
        grid_size   = self.image_size // self.patch_size
        num_patches = grid_size * grid_size
        p           = self.patch_size

        if self.patch_embed_type == "conv":
            # Conv2d expects [B, H, W, C] in Flax (channels last)
            x = x.transpose(0, 2, 3, 1)   # [B, H, W, C]
            x = nn.Conv(
                features    = self.hidden_size,
                kernel_size = (self.patch_size, self.patch_size),
                strides     = (self.patch_size, self.patch_size),
                use_bias    = True,
                name        = "projection",
            )(x)                           # [B, Gh, Gw, D]
            x = x.reshape(B, num_patches, self.hidden_size)
            return x

        patch_dim = p * p * self.num_channels
        x = x.reshape(B, C, grid_size, p, grid_size, p)
        x = x.transpose(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, num_patches, patch_dim)
        x = nn.Dense(self.hidden_size, use_bias=True, name="projection")(x)
        return x


class Embeddings(nn.Module):
    image_size:      int
    patch_size:      int
    num_channels:    int
    hidden_size:     int
    final_op:        str
    patch_embed_type: str   = "linear"
    dropout_rate:    float  = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        grid_size   = self.image_size // self.patch_size
        num_patches = grid_size * grid_size
        B           = x.shape[0]

        patch_emb = PatchEmbeddings(
            image_size       = self.image_size,
            patch_size       = self.patch_size,
            num_channels     = self.num_channels,
            hidden_size      = self.hidden_size,
            patch_embed_type = self.patch_embed_type,
            name             = "patch_embeddings",
        )
        x = patch_emb(x)  # [B, N, d]

        if self.final_op == "cls":
            n_tokens  = num_patches + 1
            cls_token = self.param("cls_token", nn.initializers.normal(), (1, 1, self.hidden_size))
            cls       = jnp.broadcast_to(cls_token, (B, 1, self.hidden_size))
            x         = jnp.concatenate([cls, x], axis=1)
        else:
            n_tokens = num_patches

        def pos_init(key, shape, dtype=jnp.float32):
            pos = jax.random.normal(key, shape, dtype=dtype)
            return pos / jnp.linalg.norm(pos, axis=-1, keepdims=True)

        pos_emb = self.param("position_embeddings", pos_init, (1, n_tokens, self.hidden_size))
        x       = x + pos_emb
        x       = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LorentzMLP(nn.Module):
    hidden_size:  int
    mlp_dim:      int
    curvature:    float
    reset_params: str   = "lorentz_kaiming"
    a_default:    float = 0.0

    def setup(self):
        self.fc1 = LorentzFullyConnected(
            in_features  = self.hidden_size + 1,
            out_features = self.mlp_dim + 1,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "fc1",
        )
        self.fc2 = LorentzFullyConnected(
            in_features  = self.mlp_dim + 1,
            out_features = self.hidden_size + 1,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "fc2",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class LorentzTransformerBlock(nn.Module):
    hidden_size:        int
    mlp_dim:            int
    curvature:          float
    num_attention_heads: int
    compute_scores:     str
    head_mode:          str
    concat_mode:        str
    skip_connection:    bool
    use_midpoint:       bool
    use_layer_norm:     bool
    dropout_rate:       float = 0.0
    reset_params:       str   = "lorentz_kaiming"
    a_default:          float = 0.0

    def setup(self):
        D = self.hidden_size + 1
        self.attention = LorentzMHA(
            input_dim       = D,
            curvature       = self.curvature,
            num_heads       = self.num_attention_heads,
            compute_scores  = self.compute_scores,
            head_mode       = self.head_mode,
            concat_mode     = self.concat_mode,
            reset_params    = self.reset_params,
            a_default       = self.a_default,
            name            = "attention",
        )
        self.mlp = LorentzMLP(
            hidden_size  = self.hidden_size,
            mlp_dim      = self.mlp_dim,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            name         = "mlp",
        )
        self.layernorm1 = LorentzLayerNorm(
            dim       = self.hidden_size,
            curvature = self.curvature,
            name      = "layernorm1",
        )
        self.layernorm2 = LorentzLayerNorm(
            dim       = self.hidden_size,
            curvature = self.curvature,
            name      = "layernorm2",
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        manifold = Lorentz(k_val=self.curvature)

        # Learnable residual scale params
        res_scale_att = self.param(
            "res_scale_att", lambda key: jnp.array(0.1)
        )
        res_scale_mlp = self.param(
            "res_scale_mlp", lambda key: jnp.array(0.1)
        )

        if self.use_layer_norm:
            x = self.layernorm1(x)

        att_output = self.attention(x)

        if self.skip_connection:
            if self.use_midpoint:
                xs2 = jnp.stack([x, att_output], axis=-2)
                w2  = jnp.stack([1.0 - res_scale_att, res_scale_att]).astype(x.dtype)
                x   = manifold.lorentz_midpoint(xs2, w2)
            else:
                x = manifold.geodesic_residual(x, att_output, weight=res_scale_att)
        else:
            x = att_output

        if self.use_layer_norm:
            x = self.layernorm2(x)

        mlp_output = self.mlp(x)

        if self.skip_connection:
            if self.use_midpoint:
                xs2        = jnp.stack([x, mlp_output], axis=-2)
                w2         = jnp.stack([1.0 - res_scale_mlp, res_scale_mlp]).astype(x.dtype)
                mlp_output = manifold.lorentz_midpoint(xs2, w2)
            else:
                mlp_output = manifold.geodesic_residual(x, mlp_output, weight=res_scale_mlp)

        return mlp_output


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LorentzEncoder(nn.Module):
    num_hidden_layers:   int
    hidden_size:         int
    mlp_dim:             int
    curvature:           float
    num_attention_heads: int
    compute_scores:      str
    head_mode:           str
    concat_mode:         str
    skip_connection:     bool
    use_midpoint:        bool
    use_layer_norm:      bool
    dropout_rate:        float = 0.0
    reset_params:        str   = "lorentz_kaiming"
    a_default:           float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        block_kwargs = dict(
            hidden_size         = self.hidden_size,
            mlp_dim             = self.mlp_dim,
            curvature           = self.curvature,
            num_attention_heads = self.num_attention_heads,
            compute_scores      = self.compute_scores,
            head_mode           = self.head_mode,
            concat_mode         = self.concat_mode,
            skip_connection     = self.skip_connection,
            use_midpoint        = self.use_midpoint,
            use_layer_norm      = self.use_layer_norm,
            dropout_rate        = self.dropout_rate,
            reset_params        = self.reset_params,
            a_default           = self.a_default,
        )
        for i in range(self.num_hidden_layers):
            x = LorentzTransformerBlock(**block_kwargs, name=f"block_{i}")(x, deterministic)
        return x


# ---------------------------------------------------------------------------
# Full ViT
# ---------------------------------------------------------------------------

class LorentzViT(nn.Module):
    """
    Lorentz Vision Transformer.

    Supports two embedding spaces:
      - 'euclidean': standard patch embedding -> expmap0 -> Lorentz encoder
      - 'lorentz':   fully hyperbolic patch embedding
    Final token pooling via 'cls' or 'pooling' (Lorentz midpoint).
    """
    image_size:          int
    patch_size:          int
    num_channels:        int
    hidden_size:         int
    num_hidden_layers:   int
    num_attention_heads: int
    mlp_dim:             int
    num_classes:         int
    curvature:           float
    head_mode:           str
    concat_mode:         str
    compute_scores:      str
    reset_params:        str
    skip_connection:     bool
    use_midpoint:        bool
    use_layer_norm:      bool
    final_op:            str
    embeddings:          str
    patch_embed_type:    str   = "linear"
    dropout_rate:        float = 0.0
    a_default:           float = 0.0
    init_mode:           str   = "normalized"

    def setup(self):
        encoder_kwargs = dict(
            num_hidden_layers   = self.num_hidden_layers,
            hidden_size         = self.hidden_size,
            mlp_dim             = self.mlp_dim,
            curvature           = self.curvature,
            num_attention_heads = self.num_attention_heads,
            compute_scores      = self.compute_scores,
            head_mode           = self.head_mode,
            concat_mode         = self.concat_mode,
            skip_connection     = self.skip_connection,
            use_midpoint        = self.use_midpoint,
            use_layer_norm      = self.use_layer_norm,
            dropout_rate        = self.dropout_rate,
            reset_params        = self.reset_params,
            a_default           = self.a_default,
        )
        self.encoder = LorentzEncoder(**encoder_kwargs, name="encoder")

        if self.embeddings == "euclidean":
            self.embedding = Embeddings(
                image_size       = self.image_size,
                patch_size       = self.patch_size,
                num_channels     = self.num_channels,
                hidden_size      = self.hidden_size,
                final_op         = self.final_op,
                patch_embed_type = self.patch_embed_type,
                dropout_rate     = self.dropout_rate,
                name             = "embedding",
            )
        elif self.embeddings == "lorentz":
            self.embedding = LorentzEmbeddings(
                image_size   = self.image_size,
                patch_size   = self.patch_size,
                num_channels = self.num_channels,
                hidden_size  = self.hidden_size,
                curvature    = self.curvature,
                final_op     = self.final_op,
                dropout_rate = self.dropout_rate,
                reset_params = self.reset_params,
                a_default    = self.a_default,
                init_mode    = self.init_mode,
                name         = "embedding",
            )
        else:
            raise ValueError(f"Unknown embeddings type: {self.embeddings}")

        self.classifier = LorentzFullyConnected(
            in_features  = self.hidden_size + 1,
            out_features = self.num_classes + 1,
            curvature    = self.curvature,
            reset_params = self.reset_params,
            a_default    = self.a_default,
            do_mlr       = True,
            name         = "classifier",
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        return_metrics: bool = False,
    ):
        manifold  = Lorentz(k_val=self.curvature)
        pre_scale = self.param("pre_scale", lambda key: jnp.array(0.3))
        r_logit   = self.param("r_logit",   lambda key: jnp.array(1.0))

        def learned_radius():
            return jax.nn.softplus(r_logit) + 1e-6

        metrics = {} if return_metrics else None

        if self.embeddings == "euclidean":
            x_euc = self.embedding(x, deterministic=deterministic)  # [B, N, d]
            r_max = learned_radius()
            norm = jnp.linalg.norm(x_euc, axis=-1, keepdims=True)
            scale = jnp.clip(r_max / jnp.clip(norm, a_min=1e-6), a_max=1.0)
            x_euc = x_euc * scale
            x_lor = manifold.expmap0(x_euc)                         # [B, N, D]
        else:
            x_lor = self.embedding(x, deterministic=deterministic)  # [B, N, D]

        x_lor = self.encoder(x_lor, deterministic=deterministic)    # [B, N, D]

        # Pool tokens
        if self.final_op == "cls":
            pooled = x_lor[:, 0, :]                                  # [B, D]
        elif self.final_op == "pooling":
            pooled = manifold.lorentz_midpoint(x_lor)                # [B, D]
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}")

        logits = self.classifier(pooled)                             # [B, num_classes+1]

        return (logits, metrics) if return_metrics else (logits, None)
