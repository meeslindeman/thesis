import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..manifold.lorentz import CustomLorentz


class LorentzMLR(nn.Module):
    """Multinomial logistic regression in the Lorentz model (HexFormer)."""

    def __init__(self, manifold: CustomLorentz, num_features: int, num_classes: int):
        super().__init__()
        self.manifold = manifold
        self.a = nn.Parameter(torch.normal(0, 0.5, (num_classes,)))
        self.z = nn.Parameter(
            F.pad(torch.normal(0, 0.5, (num_classes, num_features - 1)), pad=(1, 0), value=1)
        )
        self.init_weights()

    def forward(self, x, return_distance=False):
        sqrt_mK = 1 / self.manifold.k.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = torch.sinh(sqrt_mK * self.a) * norm_z
        w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
        self.beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)
        self.alpha = -w_t * x.narrow(-1, 0, 1) + (
            torch.cosh(sqrt_mK * self.a) * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z)
        )
        d = self.manifold.k.sqrt() * torch.abs(torch.asinh(sqrt_mK * self.alpha / self.beta))
        logits = torch.sign(self.alpha) * self.beta * d
        if return_distance:
            return logits, d
        return logits

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.z.size(1))
        nn.init.normal_(self.z, 0, stdv)
        nn.init.normal_(self.a, 0, stdv)
