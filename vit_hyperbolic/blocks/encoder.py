import torch
import torch.nn as nn

from vit_hyperbolic.blocks.attention import LorentzMHA
from layers.LLinear import LorentzFullyConnected


class LorentzMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1, 
            out_features=config["mlp_dim"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            activation=nn.functional.gelu
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            activation=nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.skip_connection = config["skip_connection"]
        self.use_midpoint = config["use_midpoint"]

        self.attention = LorentzMHA(
            input_dim=config["hidden_size"] + 1,                                   
            curvature=config["curvature"],
            num_heads=config["num_attention_heads"],
            compute_scores=config["compute_scores"],
            head_mode=config["head_mode"],
            concat_mode=config["concat_mode"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            attn_mask=config.get("attn_mask", None),
            dropout_rate=config["dropout_rate"]
        )

        self.mlp = LorentzMLP(config)

        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_output = self.attention(x)

        if self.skip_connection:
            if self.use_midpoint:
                alpha = float(self.res_scale_att)
                xs2 = torch.stack([x, att_output], dim=-2)
                w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
                x = self.manifold.lorentz_midpoint(xs2, w2)
            else:
                x = self.manifold.geodesic_residual(x, att_output, weight=self.res_scale_att)
        else:
            x = att_output

        mlp_output = self.mlp(x)

        if self.skip_connection:
            if self.use_midpoint:
                alpha = float(self.res_scale_mlp)
                xs2 = torch.stack([x, mlp_output], dim=-2)
                w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
                out = self.manifold.lorentz_midpoint(xs2, w2)
            else:
                out = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)
        else:
            out = mlp_output

        return out


class LorentzEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [LorentzTransformerBlock(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x