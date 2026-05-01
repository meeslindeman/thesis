import torch
import torch.nn as nn


class LorentzBatchNorm(nn.Module):
    """
    Lorentz Batch Normalization following Bdeir et al.

    Normalization happens in tangent space at the batch centroid:
    1. Compute batch centroid via lorentz_midpoint (Fréchet mean approximation)
    2. Log-map all points to tangent space at the centroid
    3. Parallel-transport tangent vectors to the origin
    4. Scale by gamma (/ std if normalize_variance=True)
    5. Parallel-transport to learned center beta
    6. Exp-map back to the manifold
    """

    def __init__(
        self,
        num_features: int,
        manifold,
        momentum: float = 0.1,
        eps: float = 1e-5,
        fix_gamma: bool = False,
        clamp_scale: bool = False,
        normalize_variance: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.fix_gamma = fix_gamma
        self.clamp_scale = clamp_scale
        self.normalize_variance = normalize_variance

        if fix_gamma:
            self.register_buffer('gamma', torch.ones((1,)))
        else:
            self.gamma = nn.Parameter(torch.ones((1,)))

        # Beta stored as spatial components; manifold point recomputed each forward
        # via projection_space_orthogonal so it always lies on the hyperboloid.
        self.beta_space = nn.Parameter(torch.zeros(num_features - 1))

        # Running mean stored as a full manifold point, initialised to the origin
        # (1/sqrt(k), 0, ..., 0).
        k_val = float(manifold.k())
        origin_vec = torch.zeros(num_features)
        origin_vec[0] = 1.0 / k_val ** 0.5
        self.register_buffer('running_mean', origin_vec)
        self.register_buffer('running_var', torch.ones(1))

    def _origin(self, like: torch.Tensor) -> torch.Tensor:
        """Return the manifold origin (1/sqrt(k), 0, ..., 0) on the same device/dtype."""
        o = torch.zeros(like.shape[-1], dtype=like.dtype, device=like.device)
        o[0] = 1.0 / self.manifold.k().to(like.dtype).sqrt()
        return o

    def _transport_to_origin(self, base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel-transport v from base to the manifold origin."""
        return self.manifold.parallel_transport(base, v, self._origin(base))

    def _transport_from_origin(self, v: torch.Tensor, to: torch.Tensor) -> torch.Tensor:
        """Parallel-transport v from the manifold origin to point `to`."""
        return self.manifold.parallel_transport(self._origin(to), v, to)

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _to_spatial(self, x_flat: torch.Tensor, *shape_info) -> torch.Tensor:
        raise NotImplementedError

    def _forward_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_flat: [N, C]  where N = batch × spatial dims, C = num_features
        Returns:
            normalised output [N, C]
        """
        if self.training:
            # Fréchet mean approximation via Lorentz midpoint
            mean = self.manifold.lorentz_midpoint(x_flat).unsqueeze(0)  # [1, C]

            if self.normalize_variance:
                var = (self.manifold.lorentz_distance(x_flat, mean) ** 2).mean()
                div_factor = torch.sqrt(var + self.eps)
            else:
                var = None
                div_factor = 1.0

            xT_at_mu = self.manifold.logmap(mean, x_flat)               # [N, C]
            xT_at_origin = self._transport_to_origin(mean, xT_at_mu)    # [N, C]

            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

            with torch.no_grad():
                means = torch.stack([self.running_mean, mean[0].detach()])  # [2, C]
                w = torch.tensor(
                    [1.0 - self.momentum, self.momentum],
                    dtype=means.dtype, device=means.device,
                )
                self.running_mean.copy_(
                    self.manifold.lorentz_midpoint(means, weights=w)
                )
                if self.normalize_variance and var is not None:
                    self.running_var.copy_(
                        (1.0 - self.momentum) * self.running_var
                        + self.momentum * var.detach()
                    )
        else:
            xT_at_running_mu = self.manifold.logmap(self.running_mean, x_flat)
            xT_at_origin = self._transport_to_origin(self.running_mean, xT_at_running_mu)

            div_factor = (
                torch.sqrt(self.running_var + self.eps)
                if self.normalize_variance else 1.0
            )
            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

        beta = self.manifold.projection_space_orthogonal(
            self.beta_space.unsqueeze(0)
        ).squeeze(0)                                                    # [C]

        xT_at_beta = self._transport_from_origin(xT_at_origin, beta)   # [N, C]
        return self.manifold.expmap(beta, xT_at_beta)                   # [N, C]


class LorentzBatchNorm1d(LorentzBatchNorm):
    """
    1D Lorentz Batch Normalization for sequence data.

    Input shape: [B, L, C] where C = num_features (including time component)
    """

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] -> [B*L, C]"""
        return x.reshape(-1, x.shape[-1])

    def _to_spatial(self, x_flat: torch.Tensor, batch: int, L: int) -> torch.Tensor:
        """[B*L, C] -> [B, L, C]"""
        return x_flat.view(batch, L, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, L, C = x.shape
        x_flat = self._to_flat(x)
        output_flat = self._forward_flat(x_flat)
        return self._to_spatial(output_flat, batch, L)


class LorentzBatchNorm2d(LorentzBatchNorm):
    """
    2D Lorentz Batch Normalization for image data.

    Input shape: [B, C, H, W] where C = num_features (including time component)
    """

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B*H*W, C]"""
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def _to_spatial(self, x_flat: torch.Tensor, batch: int, H: int, W: int) -> torch.Tensor:
        """[B*H*W, C] -> [B, C, H, W]"""
        return x_flat.view(batch, H, W, -1).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        x_flat = self._to_flat(x)
        output_flat = self._forward_flat(x_flat)
        return self._to_spatial(output_flat, batch, H, W)