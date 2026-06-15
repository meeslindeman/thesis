import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union


class Lorentz(nn.Module):
    """
    The Lorentz model is defined as the hyperboloid in Minkowski space with curvature k > 0.
    The manifold is defined by the equation: <x,x>_L = -x_0^2 + x_1^2 + ... + x_n^2 = -1/k, with x_0 > 0 (upper sheet).
    """
    def __init__(
        self,
        k: float = 1.0,
        requires_grad: bool = False,
        k_min: float = 1e-5,
        eps: float = 1e-5,
        max_space_norm: float = 100.0,
    ) -> None:
        super().__init__()
        self.k_min = float(k_min)
        self.eps = float(eps)
        self.max_space_norm = float(max_space_norm)
        self._k = nn.Parameter(torch.tensor(float(k)), requires_grad=requires_grad)

    def k(self) -> torch.Tensor:
        return self._k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expmap0(x)

    def normL(self, x: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lorentz norm sqrt(-<x,x>_L) for points on the hyperboloid."""
        if metric is None:
            # diag(-1, 1, 1, ..., 1)
            signs = x.new_ones(x.size(-1))
            signs[0] = -1.0
            inner = (x * x * signs).sum(dim=-1, keepdim=True)
        else:
            inner = (x * x * metric).sum(dim=-1, keepdim=True)

        return torch.sqrt(torch.clamp(-inner, min=0.0))

    def inner(self, u: torch.Tensor, v: Optional[torch.Tensor] = None, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
        """Compute the Lorentz inner product <u,v>_L along the specified dimension."""
        if v is None:
            v = u

        d = u.size(dim) - 1
        uv = u * v
        time = uv.narrow(dim, 0, 1)  # [..., 1]
        space = uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True)  # [..., 1]
        res = -time + space  # [..., 1]
        if not keepdim:
            res = res.squeeze(dim)
        return res

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, dim=-1) -> torch.Tensor:
        """Alias for inner(u, v) with keepdim=True."""
        if v is None:
            v = u
        return self.inner(u, v, keepdim=True, dim=dim)

    def lorentz_distance(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Lorentz distance: d(x,y) = (1/sqrt(k)) * arcosh(-k <x,y>_L)."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        ip = self.inner(x, y, keepdim=True)
        alpha = (-k * ip).clamp_min(1.0 + self.eps)
        return torch.acosh(alpha) / torch.sqrt(k)

    def lorentz_distance2(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Squared geodesic distance: d_L(x,y)^2 = acosh(-k <x,y>_L)^2 / k."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        ip = self.inner(x, y, keepdim=True)
        alpha = (-k * ip).clamp_min(1.0 + self.eps)
        ac = torch.acosh(alpha)
        return (ac * ac) / k

    def projection_space_orthogonal(
        self, x: torch.Tensor, manifold_dim: int = -1) -> torch.Tensor:
        """Project space coords x to the hyperboloid by solving time from constraint."""
        norm = x.norm(dim=manifold_dim, keepdim=True).clamp_min(self.eps)
        x = x * (norm.clamp(max=self.max_space_norm) / norm)
        k = self.k()
        time = torch.sqrt(
            torch.clamp(
                1.0 / k + x.pow(2).sum(dim=manifold_dim, keepdim=True), min=self.eps
            )
        )
        return torch.cat([time, x], dim=manifold_dim)

    def radial_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project a time-like vector onto the hyperboloid via radial scaling."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        norm_L = self.normL(x).clamp_min(self.eps)  # sqrt(-<x,x>_L), [..., 1]
        sign_t = torch.sign(x[..., :1])
        sign_t = torch.where(sign_t == 0, torch.ones_like(sign_t), sign_t)
        return (sign_t / (torch.sqrt(k) * norm_L)) * x

    def expmap_origin(self, v: torch.Tensor) -> torch.Tensor:
        """Lift space components v onto the hyperboloid via simplified exp map at origin."""
        c = self.k()
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        sqrt_c = torch.sqrt(c)
        x_space = (torch.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm)) * v
        x_time = torch.sqrt(1.0 / c + x_space.pow(2).sum(dim=-1, keepdim=True))
        return torch.cat([x_time, x_space], dim=-1)

    def projx(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """Project an ambient point to the hyperboloid by adjusting time coordinate only."""
        space = x.narrow(dim, 1, x.size(dim) - 1)
        norm = space.norm(dim=dim, keepdim=True).clamp_min(self.eps)
        space = space * (norm.clamp(max=self.max_space_norm) / norm)
        time = torch.sqrt(
            torch.clamp(
                1.0 / self.k() + (space * space).sum(dim=dim, keepdim=True),
                min=self.eps,
            )
        )
        return torch.cat([time, space], dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """Project ambient vector v to the tangent space at x: v + k<x,v> x."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        xv = self.inner(x, v, keepdim=True, dim=dim)  # [..., 1]
        return v + (k * xv) * x

    def relu(self, x: torch.Tensor, manifold_dim: int) -> torch.Tensor:
        """Apply ReLU on space coordinates and project back to the hyperboloid."""
        x_space = F.relu(
            x.narrow(dim=manifold_dim, start=1, length=x.shape[manifold_dim] - 1)
        )
        return self.projection_space_orthogonal(x_space, manifold_dim=manifold_dim)

    def preactivation_map(self, a: torch.Tensor) -> torch.Tensor:
        """Map Euclidean activations to hyperboloid via space->time reconstruction."""
        y_space = 1 / self.k() * torch.sinh(self.k().sqrt() * a)
        return self.projection_space_orthogonal(y_space, manifold_dim=1)

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin. Input is space-only tangent [..., D], output is point [..., 1+D]."""
        k = self.k()
        sqrt_k = torch.sqrt(k)

        norm_x = torch.norm(x, dim=-1, keepdim=True)
        is_zero = norm_x < 1e-9

        arg = sqrt_k * norm_x
        time = (1.0 / sqrt_k) * torch.cosh(arg)

        # sinh(arg)/arg -> 1 as arg->0
        factor = torch.where(is_zero, torch.ones_like(norm_x), torch.sinh(arg) / arg)
        space = factor * x
        return torch.cat([time, space], dim=-1)

    def expmap0_safe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at the origin with a zero-norm safeguard.

        Uses a Taylor approximation of sinh(arg)/arg when ||x|| ≈ 0 to avoid
        the removable 0/0 singularity and NaN gradients, while matching the
        standard exponential map elsewhere.
        """
        k = self.k()
        sqrt_k = torch.sqrt(k)

        norm_x = torch.norm(x, dim=-1, keepdim=True)
        arg = sqrt_k * norm_x

        time = (1.0 / sqrt_k) * torch.cosh(arg)

        # sinh(arg)/arg -> 1 as arg->0
        small = arg < 1e-4
        arg2 = arg * arg
        factor_small = 1.0 + arg2 / 6.0 + arg2 * arg2 / 120.0
        factor_large = torch.sinh(arg) / torch.clamp(arg, min=1e-8)
        factor = torch.where(small, factor_small, factor_large)

        space = factor * x
        return torch.cat([time, space], dim=-1)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Log map at the origin. Input point [...,1+D], output tangent [...,1+D] with zero time component."""
        k = self.k()
        sqrt_k = torch.sqrt(k)

        y_time = y[..., :1]
        y_space = y[..., 1:]

        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denom = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=1e-9))

        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + self.eps)
        factor = torch.acosh(acosh_arg) / denom

        v_space = factor * y_space
        v_time = torch.zeros_like(y_time)
        return torch.cat([v_time, v_space], dim=-1)

    def logmap0_safe(self, y: torch.Tensor) -> torch.Tensor:
        """
        Log map at the origin with a zero-norm safeguard.

        torch.where evaluates gradients for both branches, so if denom → 0 the
        gradient of ac/denom diverges and produces NaN via ∞×0 in the backward.
        Fix: substitute safe_denom = 1 for the near-origin branch so the not-
        selected branch has a bounded gradient, then use a Taylor approximation
        (factor → 1 as ‖y_space‖ → 0) for the selected branch.
        """
        k = self.k()
        sqrt_k = torch.sqrt(k)

        y_time = y[..., :1]
        y_space = y[..., 1:]

        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)

        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + self.eps)
        ac = torch.acosh(acosh_arg)

        small = norm_y_space_sq < 1e-7
        # safe_denom = 1 when small → d(ac/safe_denom)/dy_space stays finite
        safe_denom = torch.sqrt(
            torch.where(small, torch.ones_like(norm_y_space_sq), k * norm_y_space_sq)
        )
        factor_full = ac / safe_denom  # finite gradient everywhere
        factor_taylor = 1.0 - k * norm_y_space_sq / 24.0  # Taylor: factor(0) = 1
        factor = torch.where(small, factor_taylor, factor_full)

        v_space = factor * y_space
        v_time = torch.zeros_like(y_time)
        return torch.cat([v_time, v_space], dim=-1)

    def logmap(self, base_point: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Log map at an arbitrary base point. Returns ambient tangent [...,1+D] at base_point."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        sqrt_k = torch.sqrt(k)

        if base_point.dim() < x.dim():
            base_point = base_point.unsqueeze(-2)

        xy = self.inner(base_point, x, keepdim=True)
        alpha = (-k * xy).clamp_min(1.0 + self.eps)
        denom = torch.sqrt((alpha * alpha - 1).clamp_min(0))
        factor = torch.where(denom < self.eps, torch.ones_like(denom), torch.acosh(alpha) / denom)
        # u = x + k·<bp,x>·bp is already in T_bp (provably: <bp,u>_L = 0), no proju needed
        u = x + (k * xy) * base_point
        return factor * u

    def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at an arbitrary base point. Expects ambient tangent v in T_x."""
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)

        k = self.k().to(dtype=base_point.dtype, device=base_point.device)
        sqrt_k = torch.sqrt(k)

        vv = self.inner(v, v, keepdim=True, dim=-1).clamp_min(1e-8)
        norm_v = torch.sqrt(vv)
        arg = (sqrt_k * norm_v).clamp(max=85.0)

        left = torch.cosh(arg)
        right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
        out = left * base_point + right * v
        return self.projx(out, dim=-1)

    def parallel_transport(
        self, base_point: torch.Tensor, tangent_vec: torch.Tensor, to_point: torch.Tensor) -> torch.Tensor:
        """Parallel transport of tangent_vec from base_point to to_point (broadcasting over token dim)."""
        if base_point.dim() == 2 and tangent_vec.dim() == 3:
            x = base_point.unsqueeze(1)
            v = tangent_vec
            y = to_point.unsqueeze(1)
        else:
            x, v, y = base_point, tangent_vec, to_point

        # <y, v>_L  and <x, y>_L
        yLv = (-y[..., 0] * v[..., 0]).unsqueeze(-1) + torch.sum(
            y[..., 1:] * v[..., 1:], dim=-1, keepdim=True
        )
        xLy = (-x[..., 0] * y[..., 0]).unsqueeze(-1) + torch.sum(
            x[..., 1:] * y[..., 1:], dim=-1, keepdim=True
        )

        denom = 1.0 / self.k() - xLy
        denom = torch.where(denom.abs() < self.eps, denom.sign() * self.eps, denom)

        return v + (yLv / denom) * (x + y)

    def direct_concat(self, xs: torch.Tensor) -> torch.Tensor:
        """Lorentz direct concatenation (Qu et al.)."""
        time_sq_sum = (xs[..., 0] ** 2).sum(dim=-1, keepdim=True)
        sqrt_arg = time_sq_sum - (xs.shape[-2] - 1) / self.k()
        time = torch.sqrt(torch.clamp(sqrt_arg, min=1e-7))
        space = xs[..., 1:].reshape(*xs.shape[:-2], -1)
        return torch.cat([time, space], dim=-1)

    def lorentz_midpoint(self, xs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lorentz barycenter via ambient sum. Uniform if weights is None, else weighted sum."""
        if weights is None:
            num = xs.sum(dim=-2)
        else:
            num = weights @ xs
        t = num[..., :1]
        s = num[..., 1:]
        denom_sq = (t * t) - (s * s).sum(dim=-1, keepdim=True)
        denom_sq = denom_sq.clamp_min(self.eps)
        denom = torch.sqrt(denom_sq) * torch.sqrt(self.k())
        return num / denom

    def lorentz_centroid(self, xs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted Lorentz centroid for attention aggregation. w: (..., N), xs: (..., N, D)"""
        num = torch.einsum("...qk,...kd->...qd", w, xs)
        t = num[..., :1]
        s = num[..., 1:]
        denom_sq = (t * t) - (s * s).sum(dim=-1, keepdim=True)
        denom_sq = denom_sq.clamp_min(self.eps)
        return num / (torch.sqrt(denom_sq) * torch.sqrt(self.k()))

    def lorentz_variance(self, xs: torch.Tensor, mu: torch.Tensor, set_dim: int = -2) -> torch.Tensor:
        """Lorentz variance: mean squared Lorentz distance to the mean."""
        mu = mu.unsqueeze(set_dim)  # [B, D] -> [B, 1, D] for broadcasting
        d2 = self.lorentz_distance2(xs, mu)
        return d2.mean(dim=set_dim, keepdim=False)

    def geodesic_residual(self, x: torch.Tensor, y: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
        """Residual: Exp_x( weight * Log_x(y) )."""
        v = self.logmap(x, y) * weight
        return self.expmap(x, v)

    def lorentz_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        wx: float = 1.0,
        wy: float = 1.0,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """LResNet-style residual via ambient interpolation + renormalization (He et al.)."""
        z = wx * x + wy * y
        t = z[..., :1]
        s = z[..., 1:]
        zz = -t * t + (s * s).sum(dim=-1, keepdim=True)  # negative for timelike

        k = self.k()
        scale = torch.sqrt(torch.clamp((1.0 / k) / (-zz + eps), min=eps))
        return scale * z

    def lorentz_scale(self, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """LResNet optional scaling."""
        s = gamma * x[..., 1:]
        k = self.k()
        t = torch.sqrt(
            torch.clamp((s * s).sum(dim=-1, keepdim=True) + (1.0 / k), min=eps)
        )
        return torch.cat([t, s], dim=-1)

    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gyro-addition via Exp_x( PT_{0->x}( Log_0(y) ) ) as in (Shi, et al.)."""
        o = torch.zeros_like(x)
        o[..., :1] = 1.0 / torch.sqrt(self.k().to(dtype=x.dtype, device=x.device))
        v0 = self.logmap0(y)
        vx = self.parallel_transport(o, v0, x)
        vx = self.proju(x, vx, dim=-1)
        return self.expmap(x, vx)

    def gyro_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Gyro-inverse (additive inverse) for the gyro-addition defined in gyro_add."""
        return self.gyro_mul(-1.0, x)

    def gyro_mul(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        """Gyro-scalar multiplication: Exp_0( t * Log_0(x) ), as in (Shi, et al.)."""
        v0 = self.logmap0(x)

        t_tensor = (
            t if torch.is_tensor(t) else torch.tensor(t, device=x.device, dtype=x.dtype)
        )
        while t_tensor.dim() < v0.dim():
            t_tensor = t_tensor.unsqueeze(-1)

        tv0 = t_tensor * v0

        o = torch.zeros_like(x)
        o[..., :1] = (1.0 / torch.sqrt(self.k())).to(dtype=x.dtype, device=x.device)

        out = self.expmap(base_point=o, v=tv0)
        return out

    def gyro_scale(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        k = self.k()
        sqrt_k = k.sqrt()
        # Distance from origin — single acosh, not recomputed inside gyro_mul
        d = torch.acosh((sqrt_k * x[..., 0:1]).clamp_min(1.0 + 1e-6)) / sqrt_k
        d_new = d + F.softplus(delta)
        # Closed-form: point in same direction as x at distance d_new from origin.
        # Satisfies hyperboloid constraint exactly, no logmap/expmap needed.
        x_space = x[..., 1:]
        norm_space = x_space.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        kd = (k * d_new).clamp(max=80.0)
        space = (torch.sinh(kd) / (sqrt_k * norm_space)) * x_space
        return self.projx(torch.cat([torch.cosh(kd) / sqrt_k, space], dim=-1))

    def metrics(self, x: torch.Tensor) -> dict:
        """Return useful geometric metrics for a batch of points on the manifold."""
        with torch.no_grad():
            xf = x.float()
            k = self.k().float()

            # Euclidean norm of spatial components
            spatial_norm = xf[..., 1:].norm(dim=-1)

            # Hyperbolic distance to origin
            hyp_dist = (
                torch.acosh((k.sqrt() * xf[..., 0]).clamp_min(1.0 + self.eps))
                / k.sqrt()
            )

            return {
                "norm_E_mean": spatial_norm.mean().item(),
                "norm_E_max": spatial_norm.max().item(),
                "hyp_dist_mean": hyp_dist.mean().item(),
                "hyp_dist_max": hyp_dist.max().item(),
            }

    @staticmethod
    def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
        # <x,x>_L = -t^2 + ||space||^2
        x64 = x.double()
        time = x64[..., :1]
        space = x64[..., 1:]
        return -time * time + (space * space).sum(dim=-1, keepdim=True)
