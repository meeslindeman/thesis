from vit_hyperbolic.lorentz import Lorentz
import torch
import torch.nn as nn

manifold = Lorentz(k=1.0)

def inner(x, verbose=False):
    # <x,x>_L = -t^2 + ||space||^2
    time = x[..., :1]
    space = x[..., 1:]
    result = -time * time + (space * space).sum(dim=-1, keepdim=True)
    if verbose:
        print(f"Inner product: {result.item()}")
    return result

def lorentz_norm(x, verbose=False):
    inner = -x[..., :1]**2 + (x[..., 1:]**2).sum(dim=-1, keepdim=True)
    norm = inner.abs().sqrt()
    if verbose:
        print(f"Lorentz norm: {norm.item()}")
    return norm

x = torch.randn(5)
# print(x)

x_lor = manifold.expmap0(x)

class IntrinsicDropout(nn.Module):
    def __init__(self, manifold, p):
        super().__init__()
        self.manifold = manifold
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        D = x.shape[-1] - 1  # number of spatial dimensions
        k = int(D * self.p)  # number of dimensions to drop
        if k == 0:
            return x

        # Randomly select k spatial dimensions to drop (1-indexed, 0 is time)
        dims_to_drop = torch.randperm(D)[:k] + 1  # +1 to skip time component

        for i in dims_to_drop:
            # Swap dimension i with last spatial dimension
            x = x.clone()
            x[..., i], x[..., -1] = x[..., -1].clone(), x[..., i].clone()

            # Drop last spatial dimension and renormalize
            x_dropped = x[..., :-1]
            norm_sq = -x_dropped[..., 0]**2 + (x_dropped[..., 1:]**2).sum(-1)
            norm = torch.sqrt(-norm_sq).unsqueeze(-1)
            x_projected = x_dropped / (norm * self.manifold.k().sqrt())

            # Embed back: reinsert zero at last position, then swap back
            zeros = torch.zeros_like(x[..., :1])
            x = torch.cat([x_projected, zeros], dim=-1)

            # Swap back
            x = x.clone()
            x[..., i], x[..., -1] = x[..., -1].clone(), x[..., i].clone()

        return x

class LorentzProjectionDropout(nn.Module):
    def __init__(self, manifold: Lorentz, p: float = 0.0):
        super().__init__()
        self.manifold = manifold
        self.p = p

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

dropout = LorentzProjectionDropout(manifold, p=0.9)
dropout.train()

# Generate random points on the manifold
x_eucl = torch.randn(100, 6)
x = manifold.expmap0(x_eucl)  # 100 points on L^5

D = x.shape[-1] - 1  # 6 spatial dims

# Test 1: check c is always positive
x_spatial = x[..., 1:]
x_time = x[..., :1]
mask = torch.bernoulli(torch.full((D,), 0.1))  # p=0.9 so keep 10%
spatial_masked = x_spatial * mask
norm_sq = -x_time**2 + (spatial_masked**2).sum(dim=-1, keepdim=True)
c = torch.sqrt(-1.0 * norm_sq)
print("c min:", c.min().item())
print("c has nan:", c.isnan().any().item())

# Test 2: high dropout
x_single = x[0]
print("Original:", x_single)
x_dropped = dropout(x_single.unsqueeze(0)).squeeze(0)
print("Dropped:", x_dropped)
inner(x_dropped, verbose=True)

# Test 3: manually drop all spatial dims
x_time = x_single[:1]
spatial_masked = torch.zeros(D)
norm_sq = -x_time**2
c = torch.sqrt(-1.0 * norm_sq)
x_out = torch.cat([x_time, spatial_masked]) / c
print("All dropped:", x_out)
print("Expected origin:", torch.tensor([1/manifold.k().sqrt().item()] + [0]*D))

