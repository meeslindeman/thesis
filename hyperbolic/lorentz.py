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
        """
        k = self.k()
        sqrt_k = torch.sqrt(k)

        y_time = y[..., :1]  
        y_space = y[..., 1:] 

        eps = 1e-9

        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denominator_sqrt = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=eps))

        # Clamp the argument of acosh to be >= 1 to avoid NaNs
        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + self.eps)
        factor = torch.acosh(acosh_arg) / denominator_sqrt

        return factor * y_space

    def projection_space_orthogonal(self, x: torch.Tensor, manifold_dim: int = -1) -> torch.Tensor:
        """
        Projects a point onto the Lorentz model orthogonally from the space dimensions.
        """
        return torch.cat(
            [torch.sqrt(1 / self.k() + x.pow(2).sum(dim=manifold_dim, keepdim=True)), x], dim=manifold_dim
        )

    def projx(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """
        Projects an arbitrary point in R^(1+n) onto the Lorentz hyperboloid along the time dimension, forcing norm to be -1/k.
        """
        space = x.narrow(dim, 1, x.size(dim)-1)
        time = torch.sqrt(torch.clamp(1.0/self.k() + (space*space).sum(dim=dim, keepdim=True), min=self.eps))
        return torch.cat([time, space], dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        xv = self.inner(x, v, keepdim=True, dim=dim)          # [..., 1]
        return v + xv * (x / self.k())

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

        # alpha = -k * <bp, y>_L  (should be >= 1)
        alpha = (-k * inner).clamp_min(1.0 + 1e-6)

        # direction = y - alpha * bp  (tangent at bp)
        direction = y - alpha.unsqueeze(-1) * bp

        # denom = sqrt(alpha^2 - 1)
        denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=1e-8))

        factor = torch.acosh(alpha) / denom
        v = factor.unsqueeze(-1) * direction

        return v.to(x.dtype)

    def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on the Lorentz hyperboloid at an arbitrary base point.
        """
        if base_point.dim() == 2 and v.dim() == 3:
            base_point = base_point.unsqueeze(1)

        # Compute in float32 for stability (especially under AMP)
        bp = base_point.float()
        vf = v.float()

        k = self.k().float()
        sqrt_k = torch.sqrt(k)

        # Minkowski squared norm <v,v>_L (should be >= 0 for spacelike tangent vectors)
        vv = self.inner(vf, vf, keepdim=True, dim=-1).clamp_min(1e-8)
        norm_v = torch.sqrt(vv)                         # [..., 1]
        arg = sqrt_k * norm_v                           # [..., 1]

        # Coefficients; handle arg ~ 0 with the limit sinh(arg)/arg -> 1
        left_coeff = torch.cosh(arg)
        right_coeff = torch.where(
            arg < 1e-6,
            torch.ones_like(arg),
            torch.sinh(arg) / arg
        )

        out = left_coeff * bp + right_coeff * vf
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
        
        denominator = 1/self.k() - xLy 

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
        """
        Residual connection Exp_x( weight * Log_x(y) ), computed robustly in float32.
        Note: avoid alpha because it's also used in logmap -> use weight instead.
        """
        # x = self.projx(x)
        # y = self.projx(y)

        # Disable autocast for hyperbolic ops
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            yf = y.float()

            v = self.logmap(xf, yf)          # already float32 inside
            v = v * float(weight)

            # Step cap before anything else uses v
            vv = self.inner(v, v, keepdim=True, dim=-1).clamp_min(1e-8)
            norm_v = torch.sqrt(vv)
            max_step = 5.0 / torch.sqrt(self.k().float())      # arg<=5
            scale = (max_step / norm_v).clamp_max(1.0)

            hits = (scale < 1.0).sum()
            total = scale.numel()
            self.cap_hits_geo += hits.to(self.cap_hits_geo.dtype)
            self.cap_total_geo += torch.as_tensor(total, device=scale.device, dtype=self.cap_total_geo.dtype)

            v = v * scale

            out = self.expmap(xf, v)        

        out = out.to(x.dtype)
        # out = self.projx(out)
        return out


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
        # 1. log at origin
        v0 = self.logmap0(y) 

        # 2. parallel transport from 0 to x
        o = torch.zeros_like(x)
        o[..., :1] = (1.0 / torch.sqrt(self.k())).to(dtype=x.dtype, device=x.device)
        vx = self.parallel_transport(base_point=o, tangent_vec=v0, to_point=x)

        # 3. expmap from x in direction vx
        out = self.expmap(base_point=x, v=vx)
        return out

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

    def _is_on_manifold(self, x: torch.Tensor, tol: float = 1e-2, log_details: bool = False) -> bool:
        norm_sq = self._minkowski_norm_sq(x)  # [..., 1]
        target = (-1.0 / self.k().double())
        diff = norm_sq - target               # broadcast over batch dims
        max_diff = diff.abs().max()

        is_valid = bool((max_diff < tol).item())
        rel = diff.abs() / target.abs()

        if log_details:
            kv = float(self.k().detach().cpu())
            tgt = float(target.detach().cpu())
            md = float(max_diff.detach().cpu())

            print(f"[manifold check] k={kv:.6f}")
            print(f"  target Minkowski norm = {tgt:.6f}")
            print(f"  max |<x,x>_L - target| = {md:.6e}")
            print(f"  on manifold (tol={tol:.1e}) = {is_valid}")
            print("   max rel err", float(rel.max().cpu()))

        return is_valid
    
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

    # def geodesic_residual(self, x: torch.Tensor, y: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
    #     with torch.autocast(device_type="cuda", enabled=False):
    #         xf = x.float()
    #         yf = y.float()

    #         self._chk(xf, "geo_res.xf")
    #         self._chk(yf, "geo_res.yf")

    #         v = self.logmap(xf, yf)
    #         self._chk(v, "geo_res.v_after_logmap")
    #         self._stat(v, "geo_res.v_after_logmap")

    #         v = v * float(weight)
    #         self._chk(v, "geo_res.v_after_weight")
    #         self._stat(v, "geo_res.v_after_weight")

    #         # IMPORTANT: check vv before clamp_min; clamp does NOT remove NaNs
    #         vv_raw = self.inner(v, v, keepdim=True, dim=-1)
    #         self._chk(vv_raw, "geo_res.vv_raw_inner(v,v)")
    #         # vv_raw can be negative (timelike) or near zero; you are assuming it should be >=0.
    #         # track if it's often negative/invalid
    #         if (vv_raw <= 0).any():
    #             # not an error necessarily, but a strong signal that v is not a valid spacelike tangent
    #             neg = (vv_raw <= 0).sum().item()
    #             tot = vv_raw.numel()
    #             print(f"[warn] geo_res.vv_raw has {neg}/{tot} <= 0")

    #         vv = vv_raw.clamp_min(1e-8)
    #         self._chk(vv, "geo_res.vv_clamped")

    #         norm_v = torch.sqrt(vv)
    #         self._chk(norm_v, "geo_res.norm_v")

    #         max_step = 5.0 / torch.sqrt(self.k().float())
    #         scale = (max_step / norm_v).clamp_max(1.0)
    #         self._chk(scale, "geo_res.scale")

    #         v = v * scale
    #         self._chk(v, "geo_res.v_after_cap")
    #         self._stat(v, "geo_res.v_after_cap")

    #         out = self.expmap(xf, v)
    #         self._chk(out, "geo_res.out_after_expmap")
    #         self._stat(out, "geo_res.out_after_expmap")

    #     return out.to(x.dtype)

    # def logmap(self, base_point: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #     bp = base_point.float()
    #     y  = x.float()
    #     k  = self.k().float()

    #     if bp.dim() < y.dim():
    #         bp = bp.unsqueeze(-2)

    #     self._chk(bp, "logmap.bp")
    #     self._chk(y,  "logmap.y")

    #     inner = -bp[..., 0] * y[..., 0] + torch.sum(bp[..., 1:] * y[..., 1:], dim=-1)
    #     self._chk(inner, "logmap.inner(<bp,y>_L)")

    #     alpha_raw = (-k * inner)
    #     self._chk(alpha_raw, "logmap.alpha_raw=-k*inner")

    #     # This is the critical diagnostic: if alpha_raw is far below 1,
    #     # your points are not on the hyperboloid (or have wrong time orientation),
    #     # and clamping to 1+eps makes direction not tangent and can explode downstream.
    #     min_alpha = float(alpha_raw.min().detach().cpu())
    #     if min_alpha < 0.9:
    #         print(f"[warn] logmap alpha_raw min={min_alpha:.6f} (<1). Inputs likely off-manifold.")

    #     alpha = alpha_raw.clamp_min(1.0 + 1e-6)
    #     self._chk(alpha, "logmap.alpha_clamped")

    #     direction = y - alpha.unsqueeze(-1) * bp # THIS CAN BE HUGE IF alpha IS HUGE
    #     self._chk(direction, "logmap.direction=y-alpha*bp")
    #     self._stat(direction, "logmap.direction")

    #     denom_arg = alpha * alpha - 1.0
    #     self._chk(denom_arg, "logmap.denom_arg=alpha^2-1") # THIS IS THE PROBLEM

    #     denom = torch.sqrt(torch.clamp(denom_arg, min=1e-8))
    #     self._chk(denom, "logmap.denom=sqrt(clamp(alpha^2-1))")

    #     ac = torch.acosh(alpha)
    #     self._chk(ac, "logmap.acosh(alpha)")

    #     factor = ac / denom
    #     self._chk(factor, "logmap.factor=acosh/denom")
    #     self._stat(factor, "logmap.factor")

    #     v = factor.unsqueeze(-1) * direction
    #     self._chk(v, "logmap.v=factor*direction")
    #     self._stat(v, "logmap.v")

    #     return v.to(x.dtype)
        

    # def expmap(self, base_point: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    #     if base_point.dim() == 2 and v.dim() == 3:
    #         base_point = base_point.unsqueeze(1)

    #     bp = base_point.float()
    #     vf = v.float()
    #     k = self.k().float()
    #     sqrt_k = torch.sqrt(k)

    #     self._chk(bp, "expmap.bp")
    #     self._chk(vf, "expmap.vf")

    #     vv_raw = self.inner(vf, vf, keepdim=True, dim=-1)
    #     self._chk(vv_raw, "expmap.vv_raw_inner(v,v)")

    #     vv = vv_raw.clamp_min(1e-8)
    #     self._chk(vv, "expmap.vv_clamped")

    #     norm_v = torch.sqrt(vv)
    #     self._chk(norm_v, "expmap.norm_v")

    #     arg = sqrt_k * norm_v
    #     self._chk(arg, "expmap.arg=sqrt_k*norm_v")

    #     left_coeff = torch.cosh(arg)
    #     self._chk(left_coeff, "expmap.cosh(arg)")

    #     right_coeff = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
    #     self._chk(right_coeff, "expmap.right_coeff=sinh(arg)/arg")

    #     out = left_coeff * bp + right_coeff * vf
    #     self._chk(out, "expmap.out=left*bp+right*vf")
    #     self._stat(out, "expmap.out")

    #     return out.to(base_point.dtype)