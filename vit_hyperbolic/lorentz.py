import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union


class Lorentz(nn.Module):
    """
    The Lorentz model is defined as the hyperboloid in Minkowski space with curvature k > 0.
    The manifold is defined by the equation: <x,x>_L = -x_0^2 + x_1^2 + ... + x_n^2 = -1/k, with x_0 > 0 (upper sheet).
    """
    def __init__(self, k: float = 1.0, requires_grad: bool = False, constraining_strategy: nn.Module = nn.Identity(), k_min: float = 1e-5, eps: float = 1e-5, max_space_norm: float = 100.0) -> None:
        super().__init__()
        self.k_min = float(k_min)
        self.eps = float(eps)
        self.max_space_norm = float(max_space_norm)
        self.constraining_strategy = constraining_strategy
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
        time = uv.narrow(dim, 0, 1)                             # [..., 1]
        space = uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True) # [..., 1]
        res = -time + space                                     # [..., 1]
        if not keepdim:
            res = res.squeeze(dim)
        return res

    def inner_stable(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable <x,y>_L for points on the hyperboloid.
        Uses x_s·y_s = (‖x_s‖² + ‖y_s‖² - ‖x_s-y_s‖²) / 2
        to avoid cancellation between large time and space terms.
        Safe to call in bf16 — the polarization identity eliminates the
        catastrophic cancellation that would otherwise require float32.
        """
        x0  = x[..., :1]
        y0  = y[..., :1]
        x_s = x[..., 1:]
        y_s = y[..., 1:]
        diff_sq  = ((x_s - y_s) ** 2).sum(dim=-1, keepdim=True)
        xs_sq    = (x_s * x_s).sum(dim=-1, keepdim=True)
        ys_sq    = (y_s * y_s).sum(dim=-1, keepdim=True)
        space_dot = (xs_sq + ys_sq - diff_sq) * 0.5
        return -x0 * y0 + space_dot

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, dim=-1) -> torch.Tensor:
        """Alias for inner(u, v) with keepdim=True."""
        if v is None:
            v = u
        return self.inner(u, v, keepdim=True, dim=dim)

    def lorentz_distance(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Lorentz distance: d(x,y) = (1/sqrt(k)) * arcosh(-k <x,y>_L)."""
        k     = self.k().to(dtype=x.dtype, device=x.device)
        ip    = self.inner_stable(x, y)
        alpha = (-k * ip).clamp_min(1.0 + self.eps)
        return torch.acosh(alpha) / torch.sqrt(k)

    def lorentz_distance2(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Squared geodesic distance: d_L(x,y)^2 = acosh(-k <x,y>_L)^2 / k."""
        k     = self.k().to(dtype=x.dtype, device=x.device)
        ip    = self.inner_stable(x, y)
        alpha = (-k * ip).clamp_min(1.0 + self.eps)
        ac    = torch.acosh(alpha)
        return (ac * ac) / k

    def projection_space_orthogonal(self, x: torch.Tensor, manifold_dim: int = -1) -> torch.Tensor:
        """Project space coords x to the hyperboloid by solving time from constraint."""
        norm = x.norm(dim=manifold_dim, keepdim=True).clamp_min(self.eps)
        x = x * (norm.clamp(max=self.max_space_norm) / norm)
        k = self.k()
        time = torch.sqrt(torch.clamp(1.0 / k + x.pow(2).sum(dim=manifold_dim, keepdim=True), min=self.eps))
        return torch.cat([time, x], dim=manifold_dim)

    def radial_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project a time-like vector onto the hyperboloid via radial scaling."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        norm_L = self.normL(x).clamp_min(self.eps)          # sqrt(-<x,x>_L), [..., 1]
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
        time = torch.sqrt(torch.clamp(1.0 / self.k() + (space * space).sum(dim=dim, keepdim=True), min=self.eps))
        return torch.cat([time, space], dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """Project ambient vector v to the tangent space at x: v + k<x,v> x."""
        k = self.k().to(dtype=x.dtype, device=x.device)
        xv = self.inner(x, v, keepdim=True, dim=dim)  # [..., 1]
        return v + (k * xv) * x

    def relu(self, x: torch.Tensor, manifold_dim: int) -> torch.Tensor:
        """Apply ReLU on space coordinates and project back to the hyperboloid."""
        x_space = F.relu(x.narrow(dim=manifold_dim, start=1, length=x.shape[manifold_dim]-1))
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
        safe_denom = torch.sqrt(torch.where(small, torch.ones_like(norm_y_space_sq), k * norm_y_space_sq))
        factor_full   = ac / safe_denom                       # finite gradient everywhere
        factor_taylor = 1.0 - k * norm_y_space_sq / 24.0    # Taylor: factor(0) = 1
        factor = torch.where(small, factor_taylor, factor_full)

        v_space = factor * y_space
        v_time = torch.zeros_like(y_time)
        return torch.cat([v_time, v_space], dim=-1)

    def logmap(self, base_point: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Log map at an arbitrary base point. Returns ambient tangent [...,1+D] at base_point."""
        k      = self.k().to(dtype=x.dtype, device=x.device)
        sqrt_k = torch.sqrt(k)

        if base_point.dim() < x.dim():
            base_point = base_point.unsqueeze(-2)

        xy     = self.inner_stable(base_point, x)
        alpha  = (-k * xy).clamp_min(1.0 + self.eps)
        dist   = torch.acosh(alpha) / sqrt_k
        u      = x + (k * xy) * base_point
        norm_u = (torch.sinh(sqrt_k * dist) / sqrt_k).clamp_min(1e-9)
        out    = dist * (u / norm_u)
        return self.proju(base_point, out, dim=-1)

    def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at an arbitrary base point. Expects ambient tangent v in T_x."""
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)

        k      = self.k().to(dtype=base_point.dtype, device=base_point.device)
        sqrt_k = torch.sqrt(k)

        vv     = self.inner(v, v, keepdim=True, dim=-1).clamp_min(1e-8)
        norm_v = torch.sqrt(vv)
        arg    = (sqrt_k * norm_v).clamp(max=85.0)

        left  = torch.cosh(arg)
        right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
        out   = left * base_point + right * v
        return self.projx(out, dim=-1)

    def parallel_transport(self, base_point: torch.Tensor, tangent_vec: torch.Tensor, to_point: torch.Tensor) -> torch.Tensor:
        """Parallel transport of tangent_vec from base_point to to_point (broadcasting over token dim)."""
        if base_point.dim() == 2 and tangent_vec.dim() == 3:
            x = base_point.unsqueeze(1)
            v = tangent_vec
            y = to_point.unsqueeze(1)
        else:
            x, v, y = base_point, tangent_vec, to_point

        # <y, v>_L  and <x, y>_L
        yLv = (-y[..., 0] * v[..., 0]).unsqueeze(-1) + torch.sum(y[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        xLy = (-x[..., 0] * y[..., 0]).unsqueeze(-1) + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)

        denom = (1.0 / self.k() - xLy)
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
        num = torch.einsum('...qk,...kd->...qd', w, xs)
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
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            yf = y.float()
            v = self.logmap(xf, yf).float()
            v = v * weight
            out = self.expmap(xf, v)
        return out.to(x.dtype)

    def lorentz_residual(self, x: torch.Tensor, y: torch.Tensor, wx: float = 1.0, wy: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
        """LResNet-style residual via ambient interpolation + renormalization (He et al.)."""
        z = wx * x + wy * y
        t = z[..., :1]
        s = z[..., 1:]
        zz = -t * t + (s * s).sum(dim=-1, keepdim=True)  # negative for timelike

        k = self.k()
        scale = torch.sqrt(torch.clamp((1.0 / k) / (-zz + eps), min=eps))
        return scale * z

    def lorentz_scale(self, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Slide x along its geodesic from the origin by factor gamma (LResNet optional scaling)."""
        s = gamma * x[..., 1:]
        k = self.k()
        t = torch.sqrt(torch.clamp((s * s).sum(dim=-1, keepdim=True) + (1.0 / k), min=eps))
        return torch.cat([t, s], dim=-1)

    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gyro-addition via Exp_x( PT_{0->x}( Log_0(y) ) ) as in (Shi, et al.)."""
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            yf = y.float()
            kf = self.k().float()

            o = torch.zeros_like(xf)
            o[..., :1] = 1.0 / torch.sqrt(kf)

            v0 = self.logmap0(yf)                                # [...,1+D], time=0
            vx = self.parallel_transport(o, v0, xf)              # [...,1+D]
            vx = self.proju(xf, vx, dim=-1)                      # enforce tangent at x
            out = self.expmap(xf, vx)
        return out.to(x.dtype)

    def gyro_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Gyro-inverse (additive inverse) for the gyro-addition defined in gyro_add."""
        return self.gyro_mul(-1.0, x)

    def gyro_mul(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        """Gyro-scalar multiplication: Exp_0( t * Log_0(x) ), as in (Shi, et al.)."""
        v0 = self.logmap0(x)

        t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=x.device, dtype=x.dtype)
        while t_tensor.dim() < v0.dim():
            t_tensor = t_tensor.unsqueeze(-1)

        tv0 = t_tensor * v0

        o = torch.zeros_like(x)
        o[..., :1] = (1.0 / torch.sqrt(self.k())).to(dtype=x.dtype, device=x.device)

        out = self.expmap(base_point=o, v=tv0)
        return out

    def gyro_scale(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        k = self.k()
        # current hyperbolic distance to origin
        # d(o, x) = (1/sqrt(k)) * arccosh(sqrt(k) * x0)
        d = (1.0 / k.sqrt()) * torch.acosh(
            torch.clamp(k.sqrt() * x[..., 0:1], min=1.0 + 1e-6)
        )  # [..., 1]

        # t such that d(o, t⊙x) = d + delta
        delta = F.softplus(delta)  # keep positive
        t = 1.0 + delta / d.clamp(min=1e-6)  # [..., 1]

        # apply gyro scalar multiplication with per-sample t
        return self.gyro_mul(t, x)

    def metrics(self, x: torch.Tensor) -> dict:
        """Return useful geometric metrics for a batch of points on the manifold."""
        with torch.no_grad():
            xf = x.float()
            k = self.k().float()

            # Euclidean norm of spatial components
            spatial_norm = xf[..., 1:].norm(dim=-1)

            # Hyperbolic distance to origin
            hyp_dist = (torch.acosh(
                (k.sqrt() * xf[..., 0]).clamp_min(1.0 + self.eps)
            ) / k.sqrt())

            return {
                "norm_E_mean": spatial_norm.mean().item(),
                "norm_E_max":  spatial_norm.max().item(),
                "hyp_dist_mean": hyp_dist.mean().item(),
                "hyp_dist_max":  hyp_dist.max().item(),
            }

    @staticmethod
    def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
        # <x,x>_L = -t^2 + ||space||^2
        x64 = x.double()
        time = x64[..., :1]
        space = x64[..., 1:]
        return -time * time + (space * space).sum(dim=-1, keepdim=True)