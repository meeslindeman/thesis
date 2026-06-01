import torch
import torch.nn as nn

from vit_hyperbolic.lorentz import Lorentz


class GyroLorentzBatchNorm(nn.Module):
    """
    GyroLBN following Shi et al.
    """
    def __init__(
        self,
        num_features: int,
        manifold: Lorentz,
        momentum: float = 0.1,
        eps: float = 1e-5,
        normalize_variance: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.normalize_variance = normalize_variance

        self.gamma = nn.Parameter(torch.ones(1))
        self.beta_space = nn.Parameter(torch.zeros(num_features - 1))

        k_val = float(manifold.k())
        origin = torch.zeros(num_features)
        origin[0] = 1.0 / k_val ** 0.5
        self.register_buffer('running_mean', origin)
        self.register_buffer('running_var', torch.ones(1))

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core GyroLBN pass on a flat [N, C] batch.

        All gyro ops (add, inv) run in float32 internally; we cast back to the
        input dtype at the end so the layer is mixed-precision safe.
        """
        in_dtype = x.dtype
        if x.dtype != self.gamma.dtype:
            x = x.to(self.gamma.dtype)

        if self.training:
            # Closed-form Lorentzian centroid (Law et al. 2019)
            mean = self.manifold.lorentz_midpoint(x)            # [C]

            if self.normalize_variance:
                var = self.manifold.lorentz_distance2(x, mean).mean()  # scalar
            else:
                var = None

            with torch.no_grad():
                pair = torch.stack([self.running_mean.to(mean.dtype), mean.detach()])
                w    = x.new_tensor([1.0 - self.momentum, self.momentum])
                self.running_mean.copy_(
                    self.manifold.lorentz_midpoint(pair, weights=w).to(self.running_mean.dtype)
                )
                if self.normalize_variance and var is not None:
                    self.running_var.copy_(
                        (1.0 - self.momentum) * self.running_var + self.momentum * var.detach()
                    )
        else:
            mean = self.running_mean.to(x.dtype)                # [C]
            var  = self.running_var.to(x.dtype) if self.normalize_variance else None

        # 1. Center:  ⊖μ ⊕ x
        inv_mean   = self.manifold.gyro_inv(mean.unsqueeze(0))  # [1, C]
        x_centered = self.manifold.gyro_add(inv_mean, x)        # [N, C]

        # 2. Scale:  γ / (√σ² + ε) ⊙ x_centered  (or just γ when mean-only)
        scale    = self.gamma / (var.sqrt() + self.eps) if var is not None else self.gamma
        x_scaled = self.manifold.gyro_mul(scale, x_centered)    # [N, C]

        # 3. Bias:  β ⊕ x_scaled
        beta = self.manifold.projection_space_orthogonal(
            self.beta_space.unsqueeze(0)
        ).squeeze(0)                                            # [C]
        out = self.manifold.gyro_add(beta.unsqueeze(0), x_scaled)  # [N, C]

        return out.to(in_dtype)


class GyroLorentzBatchNorm1d(GyroLorentzBatchNorm):
    """
    GyroLBN for sequence data.

    Input shape: [B, L, C] where C = num_features (time + spatial components).
    """

    def __init__(self, num_features: int, manifold: Lorentz, momentum: float = 0.1,
                 eps: float = 1e-5, normalize_variance: bool = True):
        super().__init__(num_features, manifold, momentum, eps, normalize_variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        out = self._forward_flat(x.reshape(-1, C))
        return out.view(B, L, C)


class GyroLorentzBatchNorm2d(GyroLorentzBatchNorm):
    """
    GyroLBN for spatial feature maps (channel-last).

    Input shape: [B, H, W, C] where C = num_features (time + spatial components).
    """

    def __init__(self, num_features: int, manifold: Lorentz, momentum: float = 0.1,
                 eps: float = 1e-5, normalize_variance: bool = True):
        super().__init__(num_features, manifold, momentum, eps, normalize_variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        out = self._forward_flat(x.reshape(-1, C))
        return out.view(B, H, W, C)
