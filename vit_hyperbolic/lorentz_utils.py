import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Lorentz layer normalization (plain ops version).
    Assumes x has a "set axis" at -2 and Lorentz coords at -1.
    """
    def __init__(self, manifold, dim, eps: float = 1e-6):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.eps = float(eps)

        self.beta = nn.Parameter(torch.zeros(self.dim))   # tangent at origin
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean via lorentz centroid
        mu = self.manifold.lorentz_midpoint(x)      # [B, D]

        # variance via mean squared lorentz distance to mean
        var = self.manifold.lorentz_variance(x, mu, set_dim=-2) # [B, 1]

        mu_exp = mu.unsqueeze(-2)  # [B, D] -> [B, 1, D] for broadcasting
        x_centered = self.manifold.gyro_add(self.manifold.gyro_inv(mu_exp), x)

        gamma = F.softplus(self.gamma_raw) + 1e-6
        scale = gamma * torch.rsqrt(var + self.eps).unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
        x_scaled = self.manifold.gyro_mul(scale, x_centered)

        # use safe expmap for gradient flow with zero norm
        beta_point = self.manifold.expmap0_safe(self.beta.unsqueeze(0)).squeeze(0)
        beta_point = beta_point.view(1, 1, -1)
        out = self.manifold.gyro_add(beta_point, x_scaled)

        return out

class LorentzBatchNorm(nn.Module):
    """
    Lorentz batch normalization (Shi, et al.)
    """
    def __init__(self, manifold, dim, eps: float = 1e-6):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(self.dim - 1))   
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape # [B, N, D+1]

        # Mean via Lorentz centroid
        x_flat = x.reshape(1, B*N, D)               # [1, B*N, D+1]
        mu = self.manifold.lorentz_midpoint(x_flat) # [1, D+1]

        # Variance via mean squared Lorentz distance to mean
        var = self.manifold.lorentz_variance(x_flat, mu, set_dim=-2)

        # Center and scale
        mu_exp = mu.unsqueeze(0)                                                # [1, 1, D+1]
        x_centered = self.manifold.gyro_add(self.manifold.gyro_inv(mu_exp), x)  # [B, N, D+1]

        # Scale
        gamma = F.softplus(self.gamma_raw) + 1e-6
        scale = gamma * torch.rsqrt(var + self.eps)
        x_scaled = self.manifold.gyro_mul(scale, x_centered) 

        # Shift
        beta_point = self.manifold.expmap0_safe(self.beta.unsqueeze(0)).squeeze(0)  # [D+1]
        beta_point = beta_point.view(1, 1, -1)                                      # [1, 1, D+1]
        out = self.manifold.gyro_add(beta_point, x_scaled)                          # [B, N, D+1]
        return out