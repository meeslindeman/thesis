import torch

def projx_1(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        space = x.narrow(dim, 1, x.size(dim)-1)
        time = torch.sqrt(torch.clamp(1.0/1.0 + (space*space).sum(dim=dim, keepdim=True), min=1e-16))
        return torch.cat([time, space], dim=dim)

def projx_2(x, *, dim=-1):
    dn = x.size(dim) - 1
    left_ = torch.sqrt(
        1.0 + torch.norm(x.narrow(dim, 1, dn), p=2, dim=dim) ** 2
    ).unsqueeze(dim)
    right_ = x.narrow(dim, 1, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj

def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
    # <x,x>_L = -t^2 + ||space||^2
    time = x[..., :1]
    space = x[..., 1:]
    return -time * time + (space * space).sum(dim=-1, keepdim=True)

x = torch.randn(1, 2)
print(x)

xprojx_1 = projx_1(x)
print(xprojx_1)

xprojx_2 = projx_2(x)
print(xprojx_2)

print(_minkowski_norm_sq(xprojx_1))
print(_minkowski_norm_sq(xprojx_2))

def _lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_t, x_x = x[..., :1], x[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]
    y_t, y_x = y[..., :1], y[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]

    time = torch.matmul(x_t, y_t.transpose(-1, -2))    # [B, H, N, 1] @ [B, H, 1, N] -> [B, H, N, N]
    space = torch.matmul(x_x, y_x.transpose(-1, -2))   # [B, H, N, head_spatial_dim] @ [B, H, head_spatial_dim, N] -> [B, H, N, N]

    return -time + space 

print(_lorentz_inner(xprojx_1, xprojx_1))
print(_lorentz_inner(xprojx_2, xprojx_2))

print(1/1.0)


