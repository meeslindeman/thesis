import torch
import torch.nn as nn

from geoopt import ManifoldParameter
from ..manifold.lorentz import CustomLorentz


class LorentzLayerNorm(nn.Module):
    """Lorentz Layer Normalization with Centroid and Fréchet variance (HexFormer)."""

    def __init__(self, manifold: CustomLorentz, num_features: int):
        super().__init__()
        self.manifold = manifold
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = nn.Parameter(torch.ones((1,)))
        self.eps = 1e-5

    def forward(self, x):
        mean = self.manifold.centroid(x)
        if len(x.shape) == 3:
            mean = mean.unsqueeze(1)

        x_T = self.manifold.logmap(mean, x)
        x_T = self.manifold.transp0back(mean, x_T)

        var = torch.mean(torch.norm(x_T, dim=-1, keepdim=True), dim=1, keepdim=True)
        x_T = x_T * (self.gamma / (var + self.eps))

        x_T = self.manifold.transp0(self.beta, x_T)
        return self.manifold.expmap(self.beta, x_T)
