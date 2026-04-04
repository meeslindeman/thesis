import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class LorentzLayerNorm(nn.Module):
#     """
#     Lorentz layer normalization (plain ops version).
#     Assumes x has a "set axis" at -2 and Lorentz coords at -1.
#     """
#     def __init__(self, manifold, dim, eps: float = 1e-6):
#         super().__init__()
#         self.manifold = manifold
#         self.dim = dim
#         self.eps = float(eps)

#         self.beta = nn.Parameter(torch.zeros(self.dim))   # tangent at origin
#         self.gamma_raw = nn.Parameter(torch.tensor(0.0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # mean via lorentz centroid
#         mu = self.manifold.lorentz_midpoint(x)      # [B, D]

#         # variance via mean squared lorentz distance to mean
#         var = self.manifold.lorentz_variance(x, mu, set_dim=-2) # [B, 1]

#         mu_exp = mu.unsqueeze(-2)  # [B, D] -> [B, 1, D] for broadcasting
#         x_centered = self.manifold.gyro_add(self.manifold.gyro_inv(mu_exp), x)

#         gamma = F.softplus(self.gamma_raw) + 1e-6
#         scale = gamma * torch.rsqrt(var + self.eps).unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
#         x_scaled = self.manifold.gyro_mul(scale, x_centered)

#         # use safe expmap for gradient flow with zero norm
#         beta_point = self.manifold.expmap0_safe(self.beta.unsqueeze(0)).squeeze(0)
#         beta_point = beta_point.view(1, 1, -1)
#         out = self.manifold.gyro_add(beta_point, x_scaled)

#         return out

class LorentzLayerNormTheirs(nn.Module):
    """
    Lorentz Layer Normalization with hyperbolic centroid and Fréchet variance.
    """
    def __init__(self, manifold, num_features: int):
        super().__init__()
        self.manifold = manifold
        self.eps = 1e-5

        with torch.no_grad():
            origin = torch.zeros(num_features)
            origin_lifted = manifold.expmap0(origin)  # (1+D,)

        self.beta = nn.Parameter(origin_lifted)
        self.gamma = nn.Parameter(torch.ones(1))

    def _origin(self, dtype, device) -> torch.Tensor:
        """Return the origin point (1/sqrt(k), 0, ..., 0)."""
        k = self.manifold.k().to(dtype=dtype, device=device)
        o = torch.zeros(self.beta.shape, dtype=dtype, device=device)
        o[..., 0] = 1.0 / torch.sqrt(k)
        return o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 1. Compute hyperbolic centroid over token dim ---
        if x.dim() == 2:
            # single token: unsqueeze N dim, compute mean, squeeze back
            mean = self.manifold.lorentz_midpoint(x.unsqueeze(-2))  # (B, D+1)
        else:
            mean = self.manifold.lorentz_midpoint(x)  # (B, D+1)
        
        # Expand mean for broadcasting over token dim
        if x.dim() == 3:
            mean_exp = mean.unsqueeze(1)  # (B, 1, D+1)
        else:
            mean_exp = mean

        # --- 2. Log map: lift x to tangent space at mean ---
        x_T = self.manifold.logmap(mean_exp, x)  # (..., N, D+1), tangent at mean
        self.manifold._check(x_T, "logmap output")  

        # --- 3. Parallel transport tangent vectors to origin ---
        origin = self._origin(x.dtype, x.device)
        # transport from mean -> origin for each token
        x_T = self.manifold.parallel_transport(mean_exp, x_T, origin)
        self.manifold._check(x_T, "parallel transport output")
        # x_T is now a tangent vector at the origin

        # --- 4. Compute Fréchet variance: mean norm over token dim ---
        # norm of tangent vectors at origin
        norms = torch.norm(x_T, dim=-1, keepdim=True)  # (..., N, 1)
        if x.dim() == 3:
            var = norms.mean(dim=-2, keepdim=True)      # (B, 1, 1)
        else:
            var = norms.mean(dim=-2, keepdim=True)

        # --- 5. Rescale ---
        x_T = x_T * (self.gamma / (var + self.eps))

        # --- 6. Parallel transport rescaled vectors from origin to beta ---
        beta = self.beta
        while beta.dim() < x_T.dim():
            beta = beta.unsqueeze(0)

        x_T = self.manifold.parallel_transport(origin, x_T, beta)
        self.manifold._check(x_T, "parallel transport to beta output")
        # x_T is now a tangent vector at beta

        # --- 7. Exp map at beta to return to manifold ---
        output = self.manifold.expmap(beta, x_T)
        self.manifold._check(output, "final output")

        return output

class LorentzBatchNorm(nn.Module):
    """
    Lorentz batch normalization.
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


class LorentzSoftNorm(nn.Module):
    """
    Lorentz soft normalization: f(x) = (cosh(c * tanh(rho/c)), sinh(c * tanh(rho/c)) * u_hat)
    where rho is the geodesic distance to the origin and u_hat is the spatial direction.
    """
    def __init__(self, dim: int, init_c: float = None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        init_c = init_c if init_c is not None else math.sqrt(dim)
        self.log_c = nn.Parameter(torch.tensor(math.log(init_c)))
    
    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[..., 0]    
        x_sp = x[..., 1:]

        # geodesic distance to origin
        x0_safe = x0.clamp(min=1.0 + self.eps)
        rho = torch.acosh(x0_safe)

        # spatial direction
        sp_norm = x_sp.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        u_hat = x_sp / sp_norm

        c = self.c
        f_rho = c * torch.tanh(rho / c)

        new_x0  = torch.cosh(f_rho)                         
        new_xsp = torch.sinh(f_rho).unsqueeze(-1) * u_hat

        return torch.cat([new_x0.unsqueeze(-1), new_xsp], dim=-1)


class LorentzSplitNorm(nn.Module):
    """
    Lorentz split normalization: apply LorentzSoftNorm to the first half of the dimensions and leave the rest unchanged.
    """
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        self.log_c = nn.Parameter(torch.tensor(math.log(math.sqrt(dim))))
        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta  = nn.Parameter(torch.zeros(dim))

    @property
    def c(self):
        return self.log_c.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0   = x[..., 0]
        x_sp = x[..., 1:]

        # radial soft norm
        rho = torch.acosh(x0.clamp(min=1.0 + self.eps))
        c = self.c
        f_rho = c * torch.tanh(rho / c)

        # directional: per token LN
        u_hat = x_sp / x_sp.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        mu = u_hat.mean(dim=-1, keepdim=True)
        sigma = (u_hat.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        u_std = (u_hat - mu) / sigma

        if self.affine:
            u_std = self.gamma * u_std + self.beta

        # re-normalize to unit spatial norm
        u_prime = u_std / u_std.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        new_x0 = torch.cosh(f_rho)
        new_xsp = torch.sinh(f_rho).unsqueeze(-1) * u_prime
        return torch.cat([new_x0.unsqueeze(-1), new_xsp], dim=-1)

class LorentzNorm(nn.Module):
    """
    Hyperbolic LayerNorm analog — fully geodesic, no tangent space excursions.

    Steps:
      1. Lorentz centroid (closed-form barycenter)
      2. Geodesic distances to centroid
      3. Per-point interpolation factor t from distance distribution
      4. Geodesic interpolation via gyro-scalar multiplication:
             x_norm = centroid ⊕ r * ((-centroid) ⊕ x)
         where r = (1 - t) scales the geodesic offset from centroid
      5. Learnable geodesic scaling via gyro-scalar (gamma) + gyro-translation (beta)
    """

    def __init__(
        self,
        manifold,
        affine: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.eps = eps

        if affine:
            # Scalar in (0, inf): scales geodesic distance from centroid
            self.gamma = nn.Parameter(torch.ones(1))
            # Point on manifold: geodesic bias
            # Initialized at origin of hyperboloid
        else:
            self.gamma = None

    def _centroid(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N, d+1) -> (..., d+1)"""
        return self.manifold.lorentz_midpoint(x)

    def _geodesic_dist(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        x:  (..., N, d+1)
        mu: (..., d+1)
        returns: (..., N, 1)
        """
        mu_exp = mu.unsqueeze(-2).expand_as(x)
        return self.manifold.lorentz_distance(x, mu_exp)

    def _interp_factor(self, dists: torch.Tensor) -> torch.Tensor:
        """
        t in [0,1]: how much to pull each point toward centroid.
        Outliers (large dist) get t closer to 1 -> pulled harder.
        """
        mu_d  = dists.mean(dim=-2, keepdim=True).clamp(min=self.eps)
        std_d = dists.std(dim=-2, keepdim=True).clamp(min=self.eps)
        return 1.0 - torch.sigmoid((dists - mu_d) / std_d)   # (..., N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N, d+1) on hyperboloid"""
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()

            # 1. Centroid
            mu = self._centroid(xf)                              # (..., d+1)
            mu_exp = mu.unsqueeze(-2).expand_as(xf)             # (..., N, d+1)

            # 2 & 3. Distances + interpolation factor
            dists = self._geodesic_dist(xf, mu)                  # (..., N, 1)
            t     = self._interp_factor(dists)                   # (..., N, 1)

            # 4. Purely geodesic interpolation via gyro operations:
            #    - translate x into centroid's frame: delta = (-mu) ⊕ x
            #    - scale the geodesic offset: r * delta  (r = 1-t)
            #    - translate back: mu ⊕ (r * delta)
            delta   = self.manifold.gyro_add(
                self.manifold.gyro_inv(mu_exp), xf
            )                                                    # (..., N, d+1)
            r       = (1.0 - t)                                  # (..., N, 1)
            scaled  = self.manifold.gyro_mul(r, delta)           # (..., N, d+1)
            out     = self.manifold.gyro_add(mu_exp, scaled)     # (..., N, d+1)

            # 5. Learnable geodesic scaling (optional)
            #    Scales each point's geodesic distance from centroid by gamma
            if self.gamma is not None:
                delta2  = self.manifold.gyro_add(
                    self.manifold.gyro_inv(mu_exp), out
                )
                out = self.manifold.gyro_add(
                    mu_exp, self.manifold.gyro_mul(self.gamma, delta2)
                )

        return out.to(x.dtype)