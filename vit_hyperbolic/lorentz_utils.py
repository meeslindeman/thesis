import torch
import torch.nn as nn

# insert dropout, layernorm, etc

class LorentzDropout(nn.Module):
    """
    Lorentz dropout (Shi, et al.).
    """
    def __init__(self, config):
        super().__init__()
        self.manifold = config["manifold"]
        self.dropout_rate = config["dropout_rate"]
        self.eps = 1e-6
        self.drop_time = False

    def forward(self, x):
        if (not self.training) or self.dropout_rate == 0.0:
            return x
        if x.size(-1) < 2:
            raise ValueError("LorentzDropout expects last dim to be (1+d) >= 2")

        keep_prob = 1.0 - self.dropout_rate
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        if not self.drop_time:
            mask[..., 0] = 1.0

        x_tilde = x * mask / (keep_prob + self.eps)
        x_dropped = self.manifold.projx(x_tilde, dim=-1)
        return x_dropped


class LorentzLayerNorm(nn.Module):
    """
    Lorentz layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.manifold = config["manifold"]

    def forward(self, x):
        pass