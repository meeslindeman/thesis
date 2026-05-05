import torch
import torch.nn as nn

from vit_hyperbolic.lorentz import Lorentz


class LorentzGyroBN(nn.Module):
    """
    GyroBN-style batch normalization on the Lorentz model (Bdeir et al.).

    Normalizes via three hyperbolic operations:
      1. Gyro-center  : x ← gyro_inv(mean) ⊕ x
      2. Gyro-scale   : x ← gyro_mul(shift / sqrt(var + eps), x)
      3. Gyro-bias    : x ← bias ⊕ x   where bias = expmap0(weight)

    Mean and variance follow the "wai yun" tangent-space approach:
      mean = lorentz_midpoint(x)
      var  = E[‖PT_{mean→0}(log_mean(x))‖]
    """

    def __init__(
        self,
        num_features: int,
        manifold: Lorentz,
        momentum: float = 0.1,
        eps: float = 1e-6,
        track_running_stats: bool = True,
        normalize_variance: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.normalize_variance = normalize_variance

        # Learned bias in tangent space at origin (space components only).
        # expmap0(zeros) == origin, so bias starts at the origin.
        self.weight = nn.Parameter(torch.zeros(num_features - 1))
        # Learned scale; initialised to 0.5 following Bdeir et al.
        self.shift = nn.Parameter(torch.ones(1) * 0.5)

        if track_running_stats:
            origin = self._make_origin(num_features, manifold)
            self.register_buffer('running_mean', origin)
            self.register_buffer('running_var', torch.ones(1))

    @staticmethod
    def _make_origin(num_features: int, manifold: Lorentz) -> torch.Tensor:
        o = torch.zeros(num_features)
        o[0] = 1.0 / float(manifold.k()) ** 0.5
        return o

    def _origin(self, like: torch.Tensor) -> torch.Tensor:
        o = torch.zeros(self.num_features, dtype=like.dtype, device=like.device)
        o[0] = 1.0 / self.manifold.k().to(like.dtype).sqrt()
        return o

    def _compute_mean_var(self, x: torch.Tensor):
        """x: [N, C].  Returns (mean [C], var scalar)."""
        mean = self.manifold.lorentz_midpoint(x)             # [C]
        x_T = self.manifold.logmap(mean, x)                  # [N, C]
        origin = self._origin(x)
        x_T0 = self.manifold.parallel_transport(mean, x_T, origin)  # [N, C]
        var = torch.mean(x_T0.norm(dim=-1), dim=0)           # scalar
        return mean, var

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, var) -> torch.Tensor:
        """x: [N, C], mean: [C], var: scalar."""
        bias = self.manifold.expmap0(self.weight)                          # [C]

        # Gyro-center: (-mean) ⊕ x
        inv_mean = self.manifold.gyro_inv(mean)                            # [C]
        x_c = self.manifold.gyro_add(inv_mean, x)                         # [N, C]
        x_c = self.manifold.projx(x_c)                                    # re-project

        # Gyro-scale
        if self.normalize_variance:
            factor = torch.clamp(self.shift, min=-2.5, max=2.0) / (var + self.eps).sqrt()
        else:
            factor = torch.clamp(self.shift, min=-2.5, max=2.0)
        x_s = self.manifold.gyro_mul(factor, x_c)                         # [N, C]
        x_s = self.manifold.projx(x_s)                                    # re-project

        # Gyro-bias: bias ⊕ x_s
        return self.manifold.gyro_add(bias, x_s)                          # [N, C]

    def _update_running_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        m = self.momentum
        rm = self.running_mean.to(mean.dtype)
        means = torch.stack([rm, mean.detach()])
        w = torch.tensor([1.0 - m, m], dtype=means.dtype, device=means.device)
        self.running_mean.copy_(
            self.manifold.lorentz_midpoint(means, weights=w).to(self.running_mean.dtype)
        )
        self.running_var.copy_((1.0 - m) * self.running_var + m * var.detach())

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C]  →  normalised [N, C]."""
        orig_dtype = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        need_var = self.normalize_variance or self.track_running_stats

        if self.training:
            mean, var = self._compute_mean_var(x)
            if self.track_running_stats:
                with torch.no_grad():
                    self._update_running_stats(mean, var)
        elif self.track_running_stats:
            mean = self.running_mean.to(x.dtype)
            var = self.running_var.to(x.dtype)
        else:
            if need_var:
                mean, var = self._compute_mean_var(x)
            else:
                mean = self.manifold.lorentz_midpoint(x)
                var = None

        out = self._normalize(x, mean, var)
        return out.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LorentzGyroBN1d(LorentzGyroBN):
    """GyroBN for sequence data.  Input: [B, L, C]"""

    def __init__(self, num_features: int, manifold: Lorentz, **kwargs):
        super().__init__(num_features, manifold, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        out = self._forward_flat(x.reshape(-1, C))
        return out.view(B, L, C)


class LorentzGyroBN2d(LorentzGyroBN):
    """GyroBN for image data in channel-last layout.  Input: [B, H, W, C]"""

    def __init__(self, num_features: int, manifold: Lorentz, **kwargs):
        super().__init__(num_features, manifold, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        out = self._forward_flat(x.reshape(-1, C))
        return out.view(B, H, W, C)
