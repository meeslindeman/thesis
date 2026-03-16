from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from vit_hyperbolic_jax.lorentz import Lorentz


class LorentzDropout(nn.Module):
    """
    Lorentz dropout (Shi et al.).
    Masks spatial coordinates, keeps time component intact, then re-projects to manifold.
    """
    dropout_rate: float
    curvature: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        if deterministic or self.dropout_rate == 0.0:
            return x
        manifold   = Lorentz(k_val=self.curvature)
        keep_prob  = 1.0 - self.dropout_rate
        key        = self.make_rng("dropout")
        mask       = jax.random.bernoulli(key, keep_prob, x.shape).astype(x.dtype)
        # Always keep the time coordinate
        mask       = mask.at[..., 0].set(1.0)
        x_tilde    = x * mask / (keep_prob + 1e-6)
        return manifold.projx(x_tilde)


class LorentzLayerNorm(nn.Module):
    """
    Lorentz layer normalization.

    Assumes x has set axis at -2 and Lorentz coords at -1.
    Steps:
      1. Compute Lorentz centroid (mean).
      2. Compute Lorentz variance (mean squared distance to centroid).
      3. Centre via gyro-subtraction of mean.
      4. Scale by learned gamma / sqrt(var + eps).
      5. Shift by learned beta (via gyro-addition of expmap0(beta)).
    """
    dim: int          # spatial dim (D - 1)
    curvature: float
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        manifold   = Lorentz(k_val=self.curvature)

        beta      = self.param("beta",     nn.initializers.zeros,                (self.dim,))
        gamma_raw = self.param("gamma_raw", lambda key, shape: jnp.array(0.0),   ())

        # Lorentz centroid
        mu  = manifold.lorentz_midpoint(x)           # [..., D]
        # Variance (mean squared Lorentz distance)
        var = manifold.lorentz_variance(x, mu, set_dim=-2)   # [...]

        # Centre: x - mu (in gyro sense)
        mu_exp    = jnp.expand_dims(mu, -2)           # [..., 1, D]
        x_centred = manifold.gyro_add(manifold.gyro_inv(mu_exp), x)

        # Scale
        gamma     = jax.nn.softplus(gamma_raw) + 1e-6
        scale     = gamma * jax.lax.rsqrt(var + self.eps)
        scale     = scale[..., jnp.newaxis, jnp.newaxis]     # broadcast over tokens & D
        x_scaled  = manifold.gyro_mul(scale, x_centred)

        # Shift: expmap0(beta) then gyro-add
        beta_point = manifold.expmap0_safe(beta)              # [D]
        beta_point = beta_point[jnp.newaxis, jnp.newaxis, :]  # [1, 1, D]
        out        = manifold.gyro_add(beta_point, x_scaled)
        return out
