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

    # ORIGINAL VERSION, drops last spatial dimension and renormalizes
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if not self.training or self.p == 0.0:
    #         return x
        
    #     # Drop last spatial dimension
    #     x_dropped = x[..., :-1]

    #     # Lorentzian norm of x_dropped
    #     # ||x_dropped||_L^2 = -x0^2 + x1^2 + ... + x_{D-1}^2
    #     norm_sq = self.manifold.inner(x_dropped, x_dropped)  # (...,)
    #     norm = torch.sqrt(-norm_sq).unsqueeze(-1)  # (..., 1)
        
    #     # Rescale to land on L^{D-1}_kappa
    #     x_projected = x_dropped / (norm * self.manifold.k().sqrt())
        
    #     # Embed back: reinsert zero at last spatial position
    #     zeros = torch.zeros_like(x[..., :1])
    #     x_out = torch.cat([x_projected, zeros], dim=-1)
        
    #     return x_out

    # DROPS RANDOM SPATIAL DIMENSIONS, renormalizes after each drop -> expensive/slow
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if not self.training or self.p == 0.0:
    #         return x

    #     D = x.shape[-1] - 1  # number of spatial dimensions
    #     k = int(D * self.p)  # number of dimensions to drop
    #     if k == 0:
    #         return x

    #     # Randomly select k spatial dimensions to drop (1-indexed, 0 is time)
    #     dims_to_drop = torch.randperm(D)[:k] + 1  # +1 to skip time component

    #     for i in dims_to_drop:
    #         # Swap dimension i with last spatial dimension
    #         x = x.clone()
    #         x[..., i], x[..., -1] = x[..., -1].clone(), x[..., i].clone()

    #         # Drop last spatial dimension and renormalize
    #         x_dropped = x[..., :-1]
    #         norm_sq = -x_dropped[..., 0]**2 + (x_dropped[..., 1:]**2).sum(-1)
    #         norm = torch.sqrt(-norm_sq).unsqueeze(-1)
    #         x_projected = x_dropped / (norm * self.manifold.k().sqrt())

    #         # Embed back: reinsert zero at last position, then swap back
    #         zeros = torch.zeros_like(x[..., :1])
    #         x = torch.cat([x_projected, zeros], dim=-1)

    #         # Swap back
    #         x = x.clone()
    #         x[..., i], x[..., -1] = x[..., -1].clone(), x[..., i].clone()

    #     return x
    
    # ALTERNATIVE VERSION: drops random spatial dimensions all at once, then renormalizes once
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

# class LorentzDropConnect(nn.Module):
#     def __init__(self, layer: nn.Module, p: float = 0.0, 
#                  mode: str = "column", scale: bool = True):
#         """
#         mode:  "column" — mask entire columns of V (full hyperplanes)
#                "entry"  — mask individual entries of V
#         scale: whether to apply 1/(1-p) correction
#         """
#         super().__init__()
#         self.layer = layer
#         self.p = p
#         self.mode = mode
#         self.scale = scale

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if not self.training or self.p == 0.0:
#             return self.layer(x)
        
#         V = self.layer.create_spacelike_vector()
        
#         if self.mode == "column":
#             mask = torch.bernoulli(
#                 torch.full((1, V.shape[1]), 1.0 - self.p, 
#                            device=V.device, dtype=V.dtype)
#             )
#         elif self.mode == "entry":
#             mask = torch.bernoulli(
#                 torch.full(V.shape, 1.0 - self.p, 
#                            device=V.device, dtype=V.dtype)
#             )
#         else:
#             raise ValueError(f"Unknown mode: {self.mode}")
        
#         V = V * mask
#         if self.scale:
#             V = V / (1.0 - self.p)
        
#         output_space = self.layer.activation(x @ V)
#         return self.layer.manifold.projection_space_orthogonal(output_space)