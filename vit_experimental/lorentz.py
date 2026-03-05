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
    The Lorentz model is defined as the hyperboloid in Minkowski space.
    The manifold is defined by the equation: -x_0^2 + x_1^2 + ... + x_n^2 = -1/k
    """
    def __init__(self, k: float = 1.0, requires_grad: bool = False, constraining_strategy: nn.Module = nn.Identity(), k_min: float = 1e-5, eps: float = 1e-5) -> None:
        super().__init__()
        self.k_min = float(k_min)
        self.eps = float(eps)
        self.constraining_strategy = constraining_strategy

        self._k = torch.nn.Parameter(torch.tensor(k), requires_grad=requires_grad)

        self.register_buffer("cap_hits_geo", torch.zeros((), dtype=torch.long))
        self.register_buffer("cap_total_geo", torch.zeros((), dtype=torch.long))

    def k(self) -> torch.Tensor:
        return self._k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expmap0(x)

    def normL(self, x: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1
        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()

    def relu(self, x: torch.Tensor, manifold_dim: int) -> torch.Tensor:
        x_space = F.relu(x.narrow(dim=manifold_dim, start=1, length=x.shape[manifold_dim]-1))
        return self.projection_space_orthogonal(x_space, manifold_dim=manifold_dim)
    
    def preactivation_map(self, a: torch.Tensor) -> torch.Tensor:
        y_space = 1 / self.k() * torch.sinh(self.k().sqrt() * a)
        return self.projection_space_orthogonal(y_space, manifold_dim=1)

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, dim=-1) -> torch.Tensor:
        if v is None:
            v = u
        return self.inner(u, v, keepdim=True, dim=dim)

    # note: no clamping
    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps tangent vectors from the origin of the tangent space T_0 H^n_k to the Lorentz hyperboloid H^n_k.
        Handles the case where the input vector norm is zero.
        """
        k = self.k()
        sqrt_k = torch.sqrt(k)

        norm_x = torch.norm(x, dim=-1, keepdim=True)

        eps = 1e-9
        is_zero = norm_x < eps

        sqrt_k_norm_x = norm_x * sqrt_k
        time = 1.0 / sqrt_k * torch.cosh(sqrt_k_norm_x)

        factor = torch.where(
            is_zero, torch.ones_like(norm_x), torch.sinh(sqrt_k_norm_x) / sqrt_k_norm_x
        )
        space = factor * x
        return torch.cat([time, space], dim=-1)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """
        Maps points from the Lorentz hyperboloid H^n_k back to the tangent space T_0 H^n_k.
        Returns [..., 1+D] with zero time component.
        """
        k = self.k()
        sqrt_k = torch.sqrt(k)

        y_time = y[..., :1]   # [...,1]
        y_space = y[..., 1:]  # [...,D]

        eps = 1e-9
        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denom = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=eps))

        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + self.eps)
        factor = torch.acosh(acosh_arg) / denom

        v_space = factor * y_space          # [...,D]
        v_time = torch.zeros_like(y_time)   # [...,1]
        return torch.cat([v_time, v_space], dim=-1)  # [...,1+D]

    def projection_space_orthogonal(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Projects a point onto the Lorentz model orthogonally from the space dimensions.
        """
        return torch.cat(
            [torch.sqrt(1 / self.k() + x.pow(2).sum(dim=dim, keepdim=True)), x], dim=dim
        )

    def projx(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """
        Projects an arbitrary point in R^(1+n) onto the Lorentz hyperboloid along the time dimension, forcing norm to be -1/k.
        """
        space = x.narrow(dim, 1, x.size(dim)-1)
        time = torch.sqrt(torch.clamp(1.0/self.k() + (space*space).sum(dim=dim, keepdim=True), min=self.eps))
        return torch.cat([time, space], dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        k = self.k().to(dtype=x.dtype, device=x.device)
        xv = self.inner(x, v, keepdim=True, dim=dim)  # [...,1]
        return v + (k * xv) * x

    def stats(self, name, t):
        print(f"{name}: min={t.min().item():.4e}, max={t.max().item():.4e}, "
            f"mean={t.mean().item():.4e}, any_nan={torch.isnan(t).any().item()}")

    def logmap(self, base_point: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map for the Lorentz model at an arbitrary base point.
        """
        # Compute in float32 for stability
        bp = base_point.float()
        y  = x.float()
        k  = self.k().float()

        if bp.dim() < y.dim():
            bp = bp.unsqueeze(-2)

        # Lorentz inner <bp, y>_L (keepdim False here, then unsqueeze)
        inner = -bp[..., 0] * y[..., 0] + torch.sum(bp[..., 1:] * y[..., 1:], dim=-1)

        # Domain for hyperboloid: inner <= -1/k
        # Use a small margin; scale eps by dtype magnitude if desired.
        invk = (1.0 / k)
        eps = 1e-6
        inner_max = -(invk + eps)

        invalid = inner > inner_max
        invalid_frac = invalid.float().mean().item()
        worst_excess = (inner - inner_max).clamp_min(0).max().item()

        # alpha = -k * <bp, y>_L  (should be >= 1)
        inner_clamped = torch.minimum(inner, inner_max)
        alpha = -k * inner_clamped

        # direction = y - alpha * bp  (tangent at bp)
        direction = y - alpha.unsqueeze(-1) * bp

        # denom = sqrt(alpha^2 - 1)
        denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=1e-8))

        factor = torch.acosh(alpha) / denom
        v = factor.unsqueeze(-1) * direction

        return v.to(x.dtype)

    def normalize(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        k = self.k().to(dtype=x.dtype, device=x.device)
        xx = self.inner(x, x, keepdim=True, dim=dim)

        if torch.any(xx >= -self.eps):
            raise RuntimeError(f"normalize: non-timelike input, xx_max={float(xx.max())}")

        scale = torch.sqrt((-1.0 / k) / xx)
        out = scale * x

        out_time = out.narrow(dim, 0, 1).abs()
        out_space = out.narrow(dim, 1, out.size(dim) - 1)
        return torch.cat([out_time, out_space], dim=dim)

    def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)

        bp = base_point.float()
        vf = v.float()

        k = self.k().float()
        sqrt_k = torch.sqrt(k)

        vv = self.inner(vf, vf, keepdim=True, dim=-1).clamp_min(1e-8)
        norm_v = torch.sqrt(vv)
        arg = sqrt_k * norm_v

        left = torch.cosh(arg)
        right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)

        out = left * bp + right * vf

        # # critical: renormalize to kill float32 cancellation error
        # out = self.normalize(out, dim=-1)

        # print("[expmap] arg max", float(arg.max().cpu()))

        return out.to(base_point.dtype)

    def parallel_transport(self, base_point: torch.Tensor, tangent_vec: torch.Tensor, to_point: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport with broadcasting support.
        """
        # 1. Align dimensions for broadcasting
        # We unsqueeze dim 1 so shapes become [Batch, 1, Dim] to match [Batch, M, Dim]
        if base_point.dim() == 2 and tangent_vec.dim() == 3:
            x = base_point.unsqueeze(1)
            v = tangent_vec
            y = to_point.unsqueeze(1)
        else:
            # Fallback for when all inputs are [Batch, Dim]
            x, v, y = base_point, tangent_vec, to_point

        # 2. Lorentz Inner Product <to_point, tangent_vec>_L
        # y: [B, 1, D], v: [B, M, D] -> y*v: [B, M, D] -> sum: [B, M, 1]
        yLv = (-y[..., 0] * v[..., 0]).unsqueeze(-1) + torch.sum(y[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        
        # 3. Lorentz Inner Product <base_point, to_point>_L
        # u: [B, 1, D], y: [B, 1, D] -> u*y: [B, 1, D] -> sum: [B, 1, 1]
        xLy = (-x[..., 0] * y[..., 0]).unsqueeze(-1) + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        
        # 4. Scaling Coefficient
        numerator = yLv
        
        # denominator = 1/self.k() - xLy 
        denominator = (1/self.k() - xLy).clamp_min(self.eps)

        # 5. Compute Transport
        # Broadcasts: [B, M, 1] * ([B, 1, D] + [B, 1, D]) -> [B, M, D]
        return v + (numerator / denominator) * (x + y)
    
    def direct_concat(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Perform Lorentz direct concatenation as described by Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)
        """
        time_sq_sum = ((xs[..., 0]) ** 2).sum(dim=-1, keepdim=True)
        sqrt_arg = time_sq_sum - (xs.shape[-2] - 1) / self.k()
        eps = 1e-7
        time = torch.sqrt(torch.clamp(sqrt_arg, min=eps))
        space = xs[..., 1:].reshape(*xs.shape[:-2], -1)
        out = torch.cat([time, space], dim=-1)
        return out

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

    def lorentz_midpoint(self, xs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Lorentz midpoint of a set of points xs with optional weights.
        """
        # xs: [..., M, D] 
        if weights is None:
            numerator = xs.sum(dim=-2)
        else:
            numerator = weights @ xs  # [..., 1, M] @ [..., M, D] -> [..., 1, D] or [..., D] depending on shapes

        # Compute denom^2 = t^2 - ||space||^2  (must be > 0 for timelike)
        t = numerator[..., :1]
        s = numerator[..., 1:]
        denom_sq = (t * t) - (s * s).sum(dim=-1, keepdim=True)

        # Clamp away negatives / zeros 
        denom_sq = denom_sq.clamp_min(self.eps)

        denom = torch.sqrt(denom_sq) * torch.sqrt(self.k())
        out = numerator / denom

        # # Optional but recommended: force onto manifold exactly
        # out = self.projx(out)
        return out

    def geodesic_residual(self, x: torch.Tensor, y: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            yf = y.float()

            v = self.logmap(xf, yf).float()
            v = v * float(weight)

            # optional cap (remove if you insist)
            # vv = self.inner(v, v, keepdim=True, dim=-1).clamp_min(1e-8)
            # norm_v = torch.sqrt(vv)
            # max_step = 5.0 / torch.sqrt(self.k().float())
            # scale = (max_step / norm_v).clamp_max(1.0)
            # v = v * scale

            out = self.expmap(xf, v)

        return out.to(x.dtype)


    def lorentz_residual(self, x: torch.Tensor, y: torch.Tensor, wx: float = 0.5, wy: float = 0.5, eps: float = 1e-9) -> torch.Tensor:
        """
        Residual connection used in LResNet (https://arxiv.org/abs/2412.14695)
        """
        # x,y: [..., 1+d] on Lorentz hyperboloid
        z = wx * x + wy * y

        # Minkowski inner <z,z>_L = -t^2 + ||space||^2
        t = z[..., :1]
        s = z[..., 1:]
        zz = -t * t + (s * s).sum(dim=-1, keepdim=True)  # [...,1] typically negative

        k = self.k()  # positive
        scale = torch.sqrt(torch.clamp((1.0 / k) / (-zz + eps), min=eps))
        out = scale * z
        return out

    def gyro_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gyrovector addition in the Lorentz model.
            x ⊕ y = Exp_x( PT_{0→x}( Log_0(y) ) )
        As in Shi, et al. (https://openreview.net/forum?id=NNnkLi1ALt).
        """
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            yf = y.float()
            kf = self.k().float()

            # Origin (same shape as xf)
            o = torch.zeros_like(xf)
            o[..., :1] = (1.0 / torch.sqrt(kf))

            # 1) log at origin (must be Lorentz-dim tangent: [..., D+1])
            v0 = self.logmap0(yf)  # after your fix: returns [0, v_space]

            # 2) PT to x
            vx = self.parallel_transport(base_point=o, tangent_vec=v0, to_point=xf)

            # 2b) force vx ∈ T_x (critical)
            vx = self.proju(xf, vx, dim=-1)

            # 3) exp at x
            out = self.expmap(base_point=xf, v=vx)

        return out.to(x.dtype)

    def gyro_mul(self, t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        """
        Gyroscalar multiplication in the Lorentz model.
            t ⊙ x = Exp_0( t * Log_0(x) )
        As in Shi, et al. (https://openreview.net/forum?id=NNnkLi1ALt).
        """
        v0 = self.logmap0(x)

        # Ensure t broadcasts over x's last dimension
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=x.device, dtype=x.dtype)
        while t_tensor.dim() < v0.dim():
            t_tensor = t_tensor.unsqueeze(-1)

        tv0 = t_tensor * v0

        # Lorentz origin with batch shape compatible with x
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
        with torch.no_grad():
            xf = x.float()
            k = self.k().float()
            target = (-1.0 / k)

            # point norm error
            norm_sq = self._minkowski_norm_sq(xf).squeeze(-1)  # [...]
            abs_err = (norm_sq - target).abs()
            rel_err = abs_err / target.abs()

            # sheet
            x0 = xf[..., 0]
            sheet_bad = (x0 <= 0)

            # summarize
            print(f"[{name}] max abs err {abs_err.max().item():.3e}, mean abs err {abs_err.mean().item():.3e}")
            print(f"[{name}] max rel err {rel_err.max().item():.3e}, mean rel err {rel_err.mean().item():.3e}")
            print(f"[{name}] sheet bad frac {(sheet_bad.float().mean().item()):.3e}, x0 min {x0.min().item():.3e}")
            print(f"[{name}] on-manifold tol={tol} -> {(abs_err.max() < tol).item()}")
    
    @torch.no_grad()
    def cap_stats(self):
        def frac(h, t):
            t = int(t.item())
            return (float(h.item()) / t) if t > 0 else 0.0

        return {
            "geo_frac": frac(self.cap_hits_geo, self.cap_total_geo),
            "geo_hits": int(self.cap_hits_geo.item()),
            "geo_total": int(self.cap_total_geo.item()),
        }

    @torch.no_grad()
    def reset_cap_stats(self):
        self.cap_hits_geo.zero_()
        self.cap_total_geo.zero_()

    @staticmethod
    def _chk(t: torch.Tensor, tag: str):
        if not torch.isfinite(t).all():
            finite = t[torch.isfinite(t)]
            fin_min = float(finite.min()) if finite.numel() else float("nan")
            fin_max = float(finite.max()) if finite.numel() else float("nan")
            bad = (~torch.isfinite(t)).sum().item()
            raise RuntimeError(f"NON-FINITE at {tag}: shape={tuple(t.shape)} "
                            f"finite_min={fin_min} finite_max={fin_max} bad_count={bad}")

    @staticmethod
    def _stat(t: torch.Tensor, tag: str):
        # optional: quick magnitude sanity (finite-only)
        finite = t[torch.isfinite(t)]
        if finite.numel():
            m = float(finite.abs().max())
            if m > 1e6:
                print(f"[warn] {tag} |max|={m:.3e}")
