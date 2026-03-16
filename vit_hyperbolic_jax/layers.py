from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

from vit_hyperbolic_jax.lorentz import Lorentz


def _make_U_init(reset_params: str, in_spatial: int, out_spatial: int):
    """Return a Flax initializer for weight matrix U based on reset_params strategy."""
    if reset_params == "lorentz_kaiming":
        std = (1.0 / in_spatial) ** 0.5
        return nn.initializers.normal(stddev=std)
    elif reset_params == "xavier":
        std = (1.0 / (in_spatial + out_spatial)) ** 0.5
        return nn.initializers.normal(stddev=std)
    elif reset_params in ("eye", "mlr_eye"):
        def eye_init(key, shape, dtype=jnp.float32):
            return 0.5 * jnp.eye(shape[0], shape[1], dtype=dtype)
        return eye_init
    elif reset_params == "kaiming":
        std = (2.0 / in_spatial) ** 0.5
        return nn.initializers.normal(stddev=std)
    elif reset_params == "mlr":
        std = (5.0 / in_spatial) ** 0.5
        return nn.initializers.normal(stddev=std)
    else:
        raise ValueError(f"Unknown reset_params: {reset_params}")


class LorentzFullyConnected(nn.Module):
    """
    Lorentz fully-connected layer.

    Maps Lorentz-valued input x of shape [..., in_features] to [..., out_features].
    in_features and out_features include the time coordinate (total D = 1 + spatial_dim).

    Forward (non-MLR):
        V = create_spacelike_vector(U, a)      # [in_features, out_spatial]
        output_space = activation(x @ V)       # [..., out_spatial]
        return projection_space_orthogonal(output_space)  # [..., out_features]

    Forward (MLR / classifier):
        return (1/sqrt_k) * arcsinh(sqrt_k * x @ V)
    """
    in_features: int
    out_features: int
    curvature: float
    reset_params: str = "lorentz_kaiming"
    a_default: float = 0.0
    activation: Callable = jax.nn.relu
    do_mlr: bool = False

    @staticmethod
    def _create_spacelike_vector(U: jnp.ndarray, a: jnp.ndarray, curvature: float) -> jnp.ndarray:     
        """Build spacelike hyperplane normal V from weight U and bias a."""
        U_norm = jnp.linalg.norm(U, axis=0, keepdims=True).clip(1e-10)      # [1, out_spatial]
        arg    = jnp.clip(curvature ** 0.5 * a / U_norm, a_min=-100.0, a_max=100.0)
        time   = -U_norm * jnp.sinh(arg)                                    # [1, out_spatial]
        space  = jnp.cosh(arg) * U                                          # [in_spatial, out_spatial]
        return jnp.concatenate([time, space], axis=0)                       # [in_features, out_spatial]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_spatial  = self.in_features - 1
        out_spatial = self.out_features - 1

        if self.do_mlr:
            init_key = "mlr"
        else:
            init_key = self.reset_params

        U_init = _make_U_init(init_key, in_spatial, out_spatial)
        U = self.param("U", U_init, (in_spatial, out_spatial))
        a = self.param("a", lambda key, shape: jnp.full(shape, self.a_default), (1, out_spatial))

        manifold = Lorentz(k_val=self.curvature)
        V = self._create_spacelike_vector(U, a, self.curvature)

        if self.do_mlr:
            sqrt_k = float(self.curvature) ** 0.5
            return (1.0 / sqrt_k) * jnp.arcsinh(sqrt_k * x @ V)

        output_space = self.activation(x @ V)
        return manifold.projection_space_orthogonal(output_space)
