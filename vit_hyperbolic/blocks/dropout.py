import torch
import torch.nn as nn


class ExtrinsicDropout(nn.Module):
    """
    Applies dropout to the spacelike components of a Lorentz vector, 
    then projects back to the manifold.
    """
    def __init__(self, config):
        super().__init__()
        self.manifold = config["manifold"]
        self.dropout_rate = config["dropout_rate"]
        self.eps = 1e-6
        self.drop_time = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.dropout_rate == 0.0:
            return x
        if x.size(-1) < 2:
            raise ValueError("ExtrinsicDropout expects last dim to be (1+d) >= 2")

        keep_prob = 1.0 - self.dropout_rate
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        if not self.drop_time:
            mask[..., 0] = 1.0

        x_tilde = x * mask / (keep_prob + self.eps)
        x_dropped = self.manifold.projx(x_tilde, dim=-1)
        return x_dropped


class IntrinsicDropout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.manifold = config["manifold"]
        self.p = config["dropout_rate"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        D = x.shape[-1] - 1

        mask = torch.bernoulli(
            torch.full((D,), 1.0 - self.p, device=x.device, dtype=x.dtype)
        )

        if not (mask == 0).any():
            return x

        x_spatial = x[..., 1:]  # (..., D)
        x_time = x[..., :1]     # (..., 1)

        # Zero out dropped dimensions
        spatial_masked = x_spatial * mask  # (..., D)

        # Lorentzian norm squared of masked vector
        norm_sq = -x_time**2 + (spatial_masked**2).sum(dim=-1, keepdim=True)  # (..., 1)

        # Scale factor: c = sqrt(kappa * norm_sq) but norm_sq is negative
        # so c = sqrt(-kappa * norm_sq) since ||x||_L^2 = 1/kappa means norm_sq = -1/kappa
        kappa = self.manifold.k().item()
        c = torch.sqrt(-kappa * norm_sq)  # (..., 1)

        return torch.cat([x_time, spatial_masked], dim=-1) / c


class LorentzDropConnect(nn.Module):
    """
    Applies dropout to the weight matrix V of a LorentzFullyConnected layer,
    this is equivalent to randomly dropping entire hyperplanes in the output space."
    """
    def __init__(self, layer: nn.Module, p: float = 0.0):
        super().__init__()
        self.layer = layer
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return self.layer(x)
        
        V = self.layer.create_spacelike_vector()
        mask = torch.bernoulli(
            torch.full((1, V.shape[1]), 1.0 - self.p, device=V.device, dtype=V.dtype)
        )
        V = V * mask / (1.0 - self.p)
        output_space = self.layer.activation(x @ V)
        return self.layer.manifold.projection_space_orthogonal(output_space)
