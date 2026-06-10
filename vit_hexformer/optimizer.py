"""RiemannianAdamW optimizer adapted from the HexFormer geoopt fork.

Key changes vs the original:
- Uses installed geoopt's OptimMixin, ManifoldParameter, ManifoldTensor
- Uses geoopt.manifolds.lorentz.math for direct math operations
- Checks isinstance against our local CustomLorentz and geoopt's Lorentz
"""
import math
import torch

from geoopt.optim.mixin import OptimMixin
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Lorentz
from geoopt.manifolds.lorentz.math import (
    expmap0, logmap0, expmap, project_u, project, parallel_transport0back,
)

from .manifold.lorentz import CustomLorentz


class RiemannianAdamW(OptimMixin, torch.optim.AdamW):
    """AdamW with Riemannian updates for ManifoldParameters.

    Adapted from HexFormer (github.com/hhilal/HexFormer).
    Regular (Euclidean) parameters follow standard AdamW weight-decay.
    ManifoldParameters use retraction + parallel transport.
    """

    def _euclid_step(self, point, lr, weight_decay, eps, betas, amsgrad):
        if point.grad is None:
            return point
        point.mul_(1 - lr * weight_decay)
        grad = point.grad
        if grad.is_sparse:
            raise RuntimeError("RiemannianAdamW does not support sparse gradients")
        state = self.state[point]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(point, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(point, memory_format=torch.preserve_format)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(point, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = betas
        state["step"] += 1
        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
        point.addcdiv_(exp_avg, denom, value=-(lr / bc1))
        return point

    def _hyperbolic_step(self, point, lr, weight_decay, eps, betas, amsgrad):
        grad = point.grad
        if grad is None:
            return point
        if grad.is_sparse:
            raise RuntimeError("RiemannianAdamW does not support sparse gradients")
        manifold = point.manifold
        if isinstance(manifold, (CustomLorentz, Lorentz)):
            self._man_params.append([point, manifold.k.clone()])
        state = self.state[point]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(point)
            state["exp_avg_sq"] = torch.zeros_like(point)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(point)
        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        # Hyperbolic weight decay via centroid towards origin
        means = torch.concat(
            (point.unsqueeze(-2), manifold.origin(point.shape).unsqueeze(-2).to(point.device)),
            dim=-2,
        )
        point = manifold.centroid(
            means,
            w=torch.tensor((1 - lr * weight_decay, lr * weight_decay), dtype=means.dtype, device=means.device),
        )
        grad = manifold.egrad2rgrad(point, grad)
        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).add_(manifold.component_inner(point, grad), alpha=1 - betas[1])
        bc1 = 1 - betas[0] ** state["step"]
        bc2 = 1 - betas[1] ** state["step"]
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.div(bc2).sqrt_()
        else:
            denom = exp_avg_sq.div(bc2).sqrt_()
        direction = exp_avg.div(bc1) / denom.add_(eps)
        new_point, exp_avg_new = manifold.retr_transp(point, -lr * direction, exp_avg)
        point.copy_(new_point)
        exp_avg.copy_(exp_avg_new)
        return point

    def step(self, closure=None):
        loss = None
        self._man_params = []
        k_group = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if group["name"] == "k_group":
                    k_group = group
                    continue
                betas = group["betas"]
                wd = group["weight_decay"]
                eps = group["eps"]
                lr = group["lr"]
                amsgrad = group["amsgrad"]
                for point in group["params"]:
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        point.copy_(self._hyperbolic_step(point, lr, wd, eps, betas, amsgrad))
                    else:
                        point.copy_(self._euclid_step(point, lr, wd, eps, betas, amsgrad))
            if k_group is not None:
                betas = k_group["betas"]
                wd = k_group["weight_decay"]
                eps = k_group["eps"]
                lr = k_group["lr"]
                amsgrad = k_group["amsgrad"]
                for point in k_group["params"]:
                    point.copy_(self._euclid_step(point, lr, wd, eps, betas, amsgrad).clamp(0.05, 9))
            self._move_parameters()
        return loss

    @torch.no_grad()
    def _move_parameters(self):
        for point, k_old in self._man_params:
            if k_old == point.manifold.k:
                self._stabilize_param(point)
                continue
            state = self.state[point]
            exp_avg = state["exp_avg"]
            point.copy_(project(point, k=k_old))
            exp_avg.copy_(project_u(point, exp_avg, k=k_old))
            exp_avg = parallel_transport0back(point, exp_avg, k=k_old)
            tangent = logmap0(point, k=k_old, dim=-1)
            point.copy_(point.manifold.expmap0(tangent, dim=-1))
            exp_avg.copy_(point.manifold.transp0(point, exp_avg))
            self._stabilize_param(point)

    @torch.no_grad()
    def _stabilize_param(self, p):
        state = self.state[p]
        if not state:
            return
        manifold = p.manifold
        exp_avg = state["exp_avg"]
        p.copy_(manifold.projx(p))
        exp_avg.copy_(manifold.proju(p, exp_avg))


def build_optimizer(model, cfg) -> RiemannianAdamW:
    """Build RiemannianAdamW with three named param groups (regular / manifold / curvature)."""
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and "scale" not in n
                and not isinstance(p, ManifoldParameter)
                and ".k" not in n
            ],
            "name": "1",
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and isinstance(p, ManifoldParameter)
            ],
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "name": "manifold",
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and ".k" in n
            ],
            "weight_decay": cfg.weight_decay,
            "lr": 1e-1,
            "name": "k_group",
        },
    ]
    return RiemannianAdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, stabilize=1)
