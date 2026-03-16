# TODO: Credit Ricardo

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lorentz(nn.Module):
    """
    The Lorentz model is defined as the hyperboloid in Minkowski space with curvature k > 0.
    The manifold is defined by the equation: <x,x>_L = -x_0^2 + x_1^2 + ... + x_n^2 = -1/k, with x_0 > 0 (upper sheet).
    """
    def __init__(self, k: float = 1.0, requires_grad: bool = False, constraining_strategy: nn.Module = nn.Identity(), k_min: float = 1e-5, eps: float = 1e-5) -> None:
        super().__init__()
        self.k_min = float(k_min)
        self.eps = float(eps)
        self.constraining_strategy = constraining_strategy
        self._k = nn.Parameter(torch.tensor(float(k)), requires_grad=requires_grad)

    def k(self) -> torch.Tensor:
        return self._k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expmap0(x)

    def normL(self, x: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lorentz norm (uses metric diag([-1, 1, ..., 1]))."""
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1
        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()

    def inner(self, u: torch.Tensor, v: Optional[torch.Tensor] = None, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
        """
        Compute the Lorentz inner product <u,v>_L along the specified dimension.
        """
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
        Both x, y must be float32 and on the manifold.
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
        k   = self.k().to(dtype=x.dtype, device=x.device)
        xf  = x.float()
        yf  = y.float()
        ip  = self._inner_stable(xf, yf)
        alpha = (-k.float() * ip).clamp_min(1.0 + self.eps)
        return (torch.acosh(alpha) / torch.sqrt(k.float())).to(x.dtype)

    def lorentz_distance2(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Squared geodesic distance: d_L(x,y)^2 = acosh( -k <x,y>_L )^2 / k
        """
        k   = self.k().to(dtype=x.dtype, device=x.device)
        xf  = x.float()
        yf  = y.float()
        ip  = self.inner_stable(xf, yf)
        alpha = (-k.float() * ip).clamp_min(1.0 + self.eps)
        ac  = torch.acosh(alpha)
        return ((ac * ac) / k.float()).to(x.dtype)

    def projection_space_orthogonal(self, x: torch.Tensor, manifold_dim: int = -1) -> torch.Tensor:
        """Project space coords x to the hyperboloid by solving time from constraint."""
        k = self.k()
        time = torch.sqrt(torch.clamp(1.0 / self.k() + x.pow(2).sum(dim=manifold_dim, keepdim=True), min=self.eps))
        return torch.cat([time, x], dim=manifold_dim)

    def projx(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """Project an ambient point to the hyperboloid by adjusting time coordinate only."""
        space = x.narrow(dim, 1, x.size(dim) - 1)
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

        # denom = sqrt(k) * ||y_space||
        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denom = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=1e-9))

        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + self.eps)
        factor = torch.acosh(acosh_arg) / denom

        v_space = factor * y_space
        v_time = torch.zeros_like(y_time)
        return torch.cat([v_time, v_space], dim=-1)

    def logmap(self, base_point: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Log map at an arbitrary base point. Returns ambient tangent [...,1+D] at base_point."""
        bp = base_point.float()
        y = x.float()
        k = self.k().float()
        sqrt_k = torch.sqrt(k)

        if bp.dim() < y.dim():
            bp = bp.unsqueeze(-2)

        # # <bp, y>_L
        # inner = -bp[..., 0] * y[..., 0] + torch.sum(bp[..., 1:] * y[..., 1:], dim=-1)

        # # on-manifold domain: <bp,y>_L <= -1/k  => alpha = -k<bp,y>_L >= 1
        # inner_max = -(1.0 / k + 1e-6)
        # inner_clamped = torch.minimum(inner, inner_max)
        # alpha = -k * inner_clamped  # >= 1

        # direction = y - alpha.unsqueeze(-1) * bp
        # denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=1e-8))
        # factor = torch.acosh(alpha) / denom

        # out = factor.unsqueeze(-1) * direction

        xy     = self.inner_stable(bp, y)
        alpha  = (-k * xy).clamp_min(1.0 + self.eps)
        dist   = torch.acosh(alpha) / sqrt_k
        u      = y + (k * xy) * bp
        norm_u = (torch.sinh(sqrt_k * dist) / sqrt_k).clamp_min(1e-9)
        out    = dist * (u / norm_u)
        out    = self.proju(bp, out, dim=-1)
        return out.to(x.dtype)

    def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at an arbitrary base point. Expects ambient tangent v in T_x."""
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)

        bp = base_point.float()
        vf = v.float()
        k = self.k().float()
        sqrt_k = torch.sqrt(k)

        # vv = self.inner(vf, vf, keepdim=True, dim=-1).clamp_min(1e-8)
        # norm_v = torch.sqrt(vv)
        # arg = sqrt_k * norm_v

        # left = torch.cosh(arg)
        # right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
        # out = left * bp + right * vf

        vv     = self.inner(vf, vf, keepdim=True, dim=-1).clamp_min(1e-8)
        norm_v = torch.sqrt(vv)
        arg    = sqrt_k * norm_v

        left  = torch.cosh(arg)
        right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
        out   = left * bp + right * vf
        out   = self.projx(out, dim=-1)

        return out.to(base_point.dtype)

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

    def lorentz_midpoint(self, xs: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lorentz barycenter via normalized weighted ambient sum over the set axis (-2)."""
        if w is None:
            num = xs.sum(dim=-2)
        else:
            num = w @ xs  # supports (...,Nq,Nk)@(....,Nk,D)->(...,Nq,D) and (...,1,Nk)@(...,Nk,D)->(...,1,D)
            # if you used a singleton-query dimension (like (...,1,Nk)), squeeze it:
            if num.size(-2) == 1:
                num = num.squeeze(-2)

        t = num[..., :1]
        s = num[..., 1:]
        denom_sq = (t * t) - (s * s).sum(dim=-1, keepdim=True)
        denom_sq = denom_sq.clamp_min(self.eps) 
        denom = torch.sqrt(denom_sq) * torch.sqrt(self.k())
        return num / denom

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
            v = v * float(weight)
            out = self.expmap(xf, v)
        return out.to(x.dtype)


    def lorentz_residual(self, x: torch.Tensor, y: torch.Tensor, wx: float = 0.5, wy: float = 0.5, eps: float = 1e-9) -> torch.Tensor:
        """LResNet-style residual via ambient interpolation + renormalization (He et al.)."""
        z = wx * x + wy * y
        t = z[..., :1]
        s = z[..., 1:]
        zz = -t * t + (s * s).sum(dim=-1, keepdim=True)  # negative for timelike

        k = self.k()
        scale = torch.sqrt(torch.clamp((1.0 / k) / (-zz + eps), min=eps))
        return scale * z

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

    @staticmethod
    def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
        # <x,x>_L = -t^2 + ||space||^2
        x64 = x.double()
        time = x64[..., :1]
        space = x64[..., 1:]
        return -time * time + (space * space).sum(dim=-1, keepdim=True)

    def _is_on_manifold(self, x: torch.Tensor, name: str = "", tol: float = 1e-2):
        """Debug: report hyperboloid constraint error and sheet check."""
        with torch.no_grad():
            xf = x.float()
            k = self.k().float()
            target = (-1.0 / k)

            norm_sq = self._minkowski_norm_sq(xf).squeeze(-1)
            abs_err = (norm_sq - target).abs()
            rel_err = abs_err / target.abs()

            x0 = xf[..., 0]
            sheet_bad = (x0 <= 0)

            print(f"[{name}] max abs err {abs_err.max().item():.3e}, mean abs err {abs_err.mean().item():.3e}")
            print(f"[{name}] max rel err {rel_err.max().item():.3e}, mean rel err {rel_err.mean().item():.3e}")
            print(f"[{name}] sheet bad frac {(sheet_bad.float().mean().item()):.3e}, x0 min {x0.min().item():.3e}")
            print(f"[{name}] on-manifold tol={tol} -> {(abs_err.max() < tol).item()}")
    
    @staticmethod
    def _chk(t: torch.Tensor, tag: str):
        """Raise if non-finite values exist."""
        if not torch.isfinite(t).all():
            finite = t[torch.isfinite(t)]
            fin_min = float(finite.min()) if finite.numel() else float("nan")
            fin_max = float(finite.max()) if finite.numel() else float("nan")
            bad = (~torch.isfinite(t)).sum().item()
            raise RuntimeError(
                f"NON-FINITE at {tag}: shape={tuple(t.shape)} "
                f"finite_min={fin_min} finite_max={fin_max} bad_count={bad}"
            )

    @staticmethod
    def _stat(t: torch.Tensor, tag: str):
        """Optional magnitude warning."""
        finite = t[torch.isfinite(t)]
        if finite.numel():
            m = float(finite.abs().max())
            if m > 1e6:
                print(f"[warn] {tag} |max|={m:.3e}")
