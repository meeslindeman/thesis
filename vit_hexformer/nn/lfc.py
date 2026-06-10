import torch
import torch.nn as nn

from ..manifold.lorentz import CustomLorentz


class LorentzFullyConnected(nn.Module):
    """Modified Lorentz fully connected layer (Chen et al., 2022 / HexFormer)."""

    def __init__(
        self,
        manifold: CustomLorentz,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
        activation=None,
        dropout=0.0,
        nheads=1,
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.init_std = 0.02

        self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(
            torch.ones(()) * (init_scale if init_scale is not None else 2.3),
            requires_grad=learn_scale,
        )
        self.reset_parameters()

    def forward(self, x):
        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)

        if self.nheads > 1:
            x_space = x_space.view(
                x_space.size(0), x_space.size(1), self.nheads, self.out_features // self.nheads
            ).transpose(1, 2)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length
            x_time = torch.sqrt(scale ** 2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(square_norm <= 1e-10, self.manifold.k.sqrt())
        else:
            time = (
                x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
                + (self.manifold.k.sqrt().reciprocal() + 0.5)
            )
            if self.nheads > 1:
                batch_size, nheads, seq_len, head_dim = x_space.shape
                x_space_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                time = time.unsqueeze(1).expand(batch_size, nheads, seq_len, 1)
                scale = (time * time - self.manifold.k.reciprocal()) / x_space_norm
                x_space = x_space * scale.clamp_min(1e-8).sqrt()
                x_time = time
            else:
                x_space_norm = (x_space * x_space).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = (time * time - self.manifold.k.reciprocal()) / x_space_norm
                x_space = x_space * scale.clamp_min(1e-8).sqrt()
                x_time = time

        return torch.cat([x_time, x_space], dim=-1)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
