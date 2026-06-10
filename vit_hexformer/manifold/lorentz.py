import torch

from geoopt.manifolds import Lorentz
from geoopt.manifolds.lorentz import math

MIN_NORM = 1e-8


class CustomLorentz(Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k=k, learnable=learnable)
        self.max_dist = torch.sqrt(self.k) * torch.arccosh(2e3 / self.k).to(device=self.k.device)
        self.tanh_factor = torch.atanh(torch.tensor(0.99, device=self.k.device)) / (self.max_dist * 2)
        self.learnable_factor = torch.nn.Parameter(torch.randn(1, device=self.k.device))

    @torch.no_grad()
    def update_limits(self):
        self.max_dist = torch.sqrt(self.k) * torch.arccosh(2e3 / self.k).to(device=self.k.device)
        self.tanh_factor = torch.atanh(torch.tensor(0.99, device=self.k.device)) / (self.max_dist * 2)

    def scale_hyperbolic_vector(self, vector, factor=None, norm=None):
        if norm is None:
            norm = self.dist0(vector, dim=-1, keepdim=True)
        norm = norm / torch.sqrt(self.k)
        if factor is None:
            factor = self.learnable_factor
        out = vector * (torch.exp(factor * norm) - torch.exp(-factor * norm)) / (
            torch.exp(norm) - torch.exp(-norm) + MIN_NORM
        )
        return self.projx(out)

    def calculate_max_norm(self, vector, norm=None):
        if norm is None:
            norm = self.dist0(vector, dim=-1, keepdim=True)
        return self.max_dist * torch.tanh(self.tanh_factor.to(device=self.k.device) * norm)

    def rescale_to_max(self, hyperbolic_vector):
        x_norm = self.dist0(hyperbolic_vector, dim=-1, keepdim=True)
        new_norms = self.calculate_max_norm(hyperbolic_vector, x_norm)
        factor = new_norms / (x_norm + 1e-6)
        return self.scale_hyperbolic_vector(hyperbolic_vector, factor, x_norm)

    def sqdist(self, x, y, dim=-1):
        return (-2 * self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)).clamp_min(0)

    def add_time(self, space):
        return torch.cat([self.calc_time(space), space], dim=-1)

    def calc_time(self, space):
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True) ** 2 + self.k)

    def centroid(self, x, w=None, eps=1e-8, dim=-2):
        avg = w.matmul(x) if w is not None else x.mean(dim=dim)
        denom = (-self.inner(avg, avg, keepdim=True)).abs().clamp_min(eps).sqrt()
        return torch.sqrt(self.k) * avg / denom

    def csqdist(self, x, y):
        return -2 * self.k - 2 * self.cinner(x, y)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def switch_man(self, x, manifold_in):
        return self.expmap0(manifold_in.logmap0(x))
