from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Lorentz:
    """
    The Lorentz model: hyperboloid in Minkowski space with curvature k > 0.
    Constraint: <x,x>_L = -x_0^2 + x_1^2 + ... + x_n^2 = -1/k, x_0 > 0.

    Implemented as a frozen dataclass so it can be used as a Flax module attribute.
    All operations are pure JAX functions (no in-place mutation).
    """
    k_val: float = 1.0
    eps: float = 1e-5

    def k(self) -> float:
        return self.k_val

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------

    def inner(self, u: jnp.ndarray, v: Optional[jnp.ndarray] = None, keepdim: bool = False) -> jnp.ndarray:
        """Lorentz inner product <u,v>_L = -u0*v0 + sum(u_i*v_i)."""
        if v is None:
            v = u
        uv = u * v
        time = uv[..., :1]                                        # [..., 1]
        space = uv[..., 1:].sum(axis=-1, keepdims=True)           # [..., 1]
        res = -time + space                                        # [..., 1]
        if not keepdim:
            res = res[..., 0]
        return res

    def inner_stable(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Numerically stable <x,y>_L for points on the hyperboloid.
        Uses x_s·y_s = (||x_s||² + ||y_s||² - ||x_s-y_s||²) / 2.
        """
        x0  = x[..., :1]
        y0  = y[..., :1]
        x_s = x[..., 1:]
        y_s = y[..., 1:]
        diff_sq   = ((x_s - y_s) ** 2).sum(axis=-1, keepdims=True)
        xs_sq     = (x_s * x_s).sum(axis=-1, keepdims=True)
        ys_sq     = (y_s * y_s).sum(axis=-1, keepdims=True)
        space_dot = (xs_sq + ys_sq - diff_sq) * 0.5
        return -x0 * y0 + space_dot

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def lorentz_distance2(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Squared geodesic distance: acosh(-k <x,y>_L)^2 / k."""
        k  = float(self.k())
        xf = x.astype(jnp.float32)
        yf = y.astype(jnp.float32)
        ip    = self.inner_stable(xf, yf)
        alpha = jnp.clip(-k * ip, a_min=1.0 + self.eps)
        ac    = jnp.arccosh(alpha)
        return ((ac * ac) / k).astype(x.dtype)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def projection_space_orthogonal(self, x: jnp.ndarray, manifold_dim: int = -1) -> jnp.ndarray:
        """Project space coords to the hyperboloid by solving for the time coordinate."""
        k    = float(self.k())
        time = jnp.sqrt(jnp.clip(
            1.0 / k + (x ** 2).sum(axis=manifold_dim, keepdims=True),
            a_min=self.eps
        ))
        return jnp.concatenate([time, x], axis=manifold_dim)

    def projx(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project ambient point to hyperboloid by adjusting the time coordinate only."""
        space = x[..., 1:]
        time  = jnp.sqrt(jnp.clip(
            1.0 / self.k() + (space * space).sum(axis=-1, keepdims=True),
            a_min=self.eps
        ))
        return jnp.concatenate([time, space], axis=-1)

    def proju(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Project ambient vector v to the tangent space at x: v + k<x,v>_L * x."""
        k  = float(self.k())
        xv = self.inner(x, v, keepdim=True)
        return v + (k * xv) * x

    # ------------------------------------------------------------------
    # Exponential / logarithmic maps
    # ------------------------------------------------------------------

    def expmap0(self, x: jnp.ndarray) -> jnp.ndarray:
        """Exponential map at the origin. Input is space-only tangent [..., D], output [..., 1+D]."""
        k      = float(self.k())
        sqrt_k = k ** 0.5
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        is_zero = norm_x < 1e-9
        arg    = sqrt_k * norm_x
        time   = (1.0 / sqrt_k) * jnp.cosh(arg)
        factor = jnp.where(is_zero, jnp.ones_like(norm_x), jnp.sinh(arg) / arg)
        space  = factor * x
        return jnp.concatenate([time, space], axis=-1)

    def expmap0_safe(self, x: jnp.ndarray) -> jnp.ndarray:
        """Exponential map at the origin with Taylor approximation near zero."""
        k      = float(self.k())
        sqrt_k = k ** 0.5
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        arg    = sqrt_k * norm_x
        time   = (1.0 / sqrt_k) * jnp.cosh(arg)
        small  = arg < 1e-4
        arg2   = arg * arg
        factor_small = 1.0 + arg2 / 6.0 + arg2 * arg2 / 120.0
        factor_large = jnp.sinh(arg) / jnp.clip(arg, a_min=1e-8)
        factor = jnp.where(small, factor_small, factor_large)
        space  = factor * x
        return jnp.concatenate([time, space], axis=-1)

    def logmap0(self, y: jnp.ndarray) -> jnp.ndarray:
        """Log map at the origin. Input [..., 1+D], output tangent [..., 1+D] with zero time."""
        k      = float(self.k())
        sqrt_k = k ** 0.5
        y_time  = y[..., :1]
        y_space = y[..., 1:]
        norm_y_space_sq = (y_space * y_space).sum(axis=-1, keepdims=True)
        denom   = jnp.sqrt(jnp.clip(k * norm_y_space_sq, a_min=1e-9))
        acosh_arg = jnp.clip(sqrt_k * y_time, a_min=1.0 + self.eps)
        factor  = jnp.arccosh(acosh_arg) / denom
        v_space = factor * y_space
        v_time  = jnp.zeros_like(y_time)
        return jnp.concatenate([v_time, v_space], axis=-1)

    def logmap(self, base_point: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Log map at an arbitrary base point. Returns ambient tangent at base_point."""
        bp     = base_point.astype(jnp.float32)
        y      = x.astype(jnp.float32)
        k      = float(self.k())
        sqrt_k = k ** 0.5
        if bp.ndim < y.ndim:
            bp = jnp.expand_dims(bp, -2)
        xy     = self.inner_stable(bp, y)
        alpha  = jnp.clip(-k * xy, a_min=1.0 + self.eps)
        dist   = jnp.arccosh(alpha) / sqrt_k
        u      = y + (k * xy) * bp
        norm_u = jnp.clip(jnp.sinh(sqrt_k * dist) / sqrt_k, a_min=1e-9)
        out    = dist * (u / norm_u)
        out    = self.proju(bp, out)
        return out.astype(x.dtype)

    def expmap(self, base_point: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exponential map at an arbitrary base point. Expects ambient tangent v in T_x."""
        if base_point.ndim == 2 and v.ndim == 3:
            base_point = jnp.expand_dims(base_point, 1)
        bp     = base_point.astype(jnp.float32)
        vf     = v.astype(jnp.float32)
        k      = float(self.k())
        sqrt_k = k ** 0.5
        vv     = jnp.clip(self.inner(vf, vf, keepdim=True), a_min=1e-8)
        norm_v = jnp.sqrt(vv)
        arg    = sqrt_k * norm_v
        left   = jnp.cosh(arg)
        right  = jnp.where(arg < 1e-6, jnp.ones_like(arg), jnp.sinh(arg) / arg)
        out    = left * bp + right * vf
        out    = self.projx(out)
        return out.astype(base_point.dtype)

    # ------------------------------------------------------------------
    # Parallel transport
    # ------------------------------------------------------------------

    def parallel_transport(
        self,
        base_point: jnp.ndarray,
        tangent_vec: jnp.ndarray,
        to_point: jnp.ndarray,
    ) -> jnp.ndarray:
        """Parallel transport of tangent_vec from base_point to to_point."""
        if base_point.ndim == 2 and tangent_vec.ndim == 3:
            x = jnp.expand_dims(base_point, 1)
            v = tangent_vec
            y = jnp.expand_dims(to_point, 1)
        else:
            x, v, y = base_point, tangent_vec, to_point
        yLv = (jnp.expand_dims(-y[..., 0] * v[..., 0], -1)
               + (y[..., 1:] * v[..., 1:]).sum(axis=-1, keepdims=True))
        xLy = (jnp.expand_dims(-x[..., 0] * y[..., 0], -1)
               + (x[..., 1:] * y[..., 1:]).sum(axis=-1, keepdims=True))
        denom = 1.0 / self.k() - xLy
        denom = jnp.where(jnp.abs(denom) < self.eps, jnp.sign(denom) * self.eps, denom)
        return v + (yLv / denom) * (x + y)

    # ------------------------------------------------------------------
    # Barycenter / midpoint
    # ------------------------------------------------------------------

    def lorentz_midpoint(self, xs: jnp.ndarray, w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Lorentz barycenter via normalized weighted ambient sum over the set axis (-2)."""
        if w is None:
            num = xs.sum(axis=-2)
        else:
            num = w @ xs
            if num.shape[-2] == 1:
                num = num[..., 0, :]
        t        = num[..., :1]
        s        = num[..., 1:]
        denom_sq = (t * t) - (s * s).sum(axis=-1, keepdims=True)
        denom_sq = jnp.clip(denom_sq, a_min=self.eps)
        denom    = jnp.sqrt(denom_sq) * (float(self.k()) ** 0.5)
        return num / denom

    def lorentz_variance(self, xs: jnp.ndarray, mu: jnp.ndarray, set_dim: int = -2) -> jnp.ndarray:
        """Mean squared Lorentz distance to the mean."""
        mu = jnp.expand_dims(mu, set_dim)
        d2 = self.lorentz_distance2(xs, mu)
        return d2.mean(axis=set_dim)

    # ------------------------------------------------------------------
    # Residual connections
    # ------------------------------------------------------------------

    def geodesic_residual(self, x: jnp.ndarray, y: jnp.ndarray, weight: float = 0.5) -> jnp.ndarray:
        """Residual: Exp_x(weight * Log_x(y))."""
        xf  = x.astype(jnp.float32)
        yf  = y.astype(jnp.float32)
        v   = self.logmap(xf, yf).astype(jnp.float32)
        v   = v * weight
        out = self.expmap(xf, v)
        return out.astype(x.dtype)

    def lorentz_residual(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        wx: float = 0.5,
        wy: float = 0.5,
        eps: float = 1e-9,
    ) -> jnp.ndarray:
        """LResNet-style residual via ambient interpolation + renormalization."""
        z   = wx * x + wy * y
        t   = z[..., :1]
        s   = z[..., 1:]
        zz  = -t * t + (s * s).sum(axis=-1, keepdims=True)
        k   = float(self.k())
        scale = jnp.sqrt(jnp.clip((1.0 / k) / (-zz + eps), a_min=eps))
        return scale * z

    # ------------------------------------------------------------------
    # Gyro operations
    # ------------------------------------------------------------------

    def gyro_add(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Gyro-addition: Exp_x(PT_{0->x}(Log_0(y)))."""
        xf = x.astype(jnp.float32)
        yf = y.astype(jnp.float32)
        kf = float(self.k())
        # origin on manifold
        o_time  = jnp.full(xf.shape[:-1] + (1,), 1.0 / kf ** 0.5, dtype=jnp.float32)
        o_space = jnp.zeros(xf.shape[:-1] + (xf.shape[-1] - 1,), dtype=jnp.float32)
        o       = jnp.concatenate([o_time, o_space], axis=-1)
        v0  = self.logmap0(yf)
        vx  = self.parallel_transport(o, v0, xf)
        vx  = self.proju(xf, vx)
        out = self.expmap(xf, vx)
        return out.astype(x.dtype)

    def gyro_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gyro-inverse (additive inverse)."""
        return self.gyro_mul(-1.0, x)

    def gyro_mul(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Gyro-scalar multiplication: Exp_0(t * Log_0(x))."""
        v0  = self.logmap0(x)
        tv0 = t * v0
        o_time  = jnp.full(x.shape[:-1] + (1,), 1.0 / float(self.k()) ** 0.5, dtype=x.dtype)
        o_space = jnp.zeros(x.shape[:-1] + (x.shape[-1] - 1,), dtype=x.dtype)
        o       = jnp.concatenate([o_time, o_space], axis=-1)
        return self.expmap(base_point=o, v=tv0)

    # ------------------------------------------------------------------
    # Debug utilities
    # ------------------------------------------------------------------

    def normL(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorentz norm with metric diag([-1, 1, ..., 1])."""
        metric = jnp.ones(x.shape[-1]).at[0].set(-1.0)
        return jnp.sqrt(jnp.clip((x * x * metric).sum(axis=-1, keepdims=True), a_min=0.0))
