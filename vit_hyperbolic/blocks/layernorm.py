import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LorentzRMSNorm(nn.Module):
    def __init__(self, config, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.manifold = config["manifold"]

        # learned per-channel target hyperbolic norm
        self.gamma = nn.Parameter(torch.ones(1))  # scalar, or per-dim if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # current hyperbolic norm per token: d(o, x)
        # arccosh(-k * <x, o>_L) / sqrt(-k), where <x,o>_L = -x[...,0]/sqrt(-k)
        k = self.manifold.k()  # positive scalar, curvature magnitude
        hyp_norm = torch.acosh(
            torch.clamp(x[..., :1] * k.sqrt(), min=1.0 + self.eps)
        ) / k.sqrt()      # (..., 1)

        # scale factor to bring norm to gamma
        t = self.gamma / (hyp_norm + self.eps)  # (..., 1)
        return self.manifold.gyro_mul(t, x)