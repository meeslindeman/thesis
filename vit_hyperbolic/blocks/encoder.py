import torch
import torch.nn as nn

from vit_hyperbolic.blocks.attention import LorentzMHA
from vit_hyperbolic.blocks.layernorm import LorentzNorm
from vit_hyperbolic.blocks.dropout import LorentzDropConnect, IntrinsicDropout
from layers.LLinear import LorentzFullyConnected


class LorentzMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]

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
        # self.fc2 = LorentzDropConnect(
        #     LorentzFullyConnected(
        #         in_features=config["mlp_dim"] + 1, 
        #         out_features=config["hidden_size"] + 1,
        #         manifold=config["manifold"],
        #         reset_params=config["reset_params"],
        #         a_default=config["a_default"],
        #         activation=nn.Identity()
        #     ),
        #     p=config["dropout_rate"]
        # )

        self.dropout = IntrinsicDropout(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.dropout(x)
        return x
    

class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.residual = config["residual"]
        self.layernorm = config["layernorm"]
        self.dropout = config["dropout"]

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
        self.layernorm1 = LorentzNorm(config["manifold"])
        self.layernorm2 = LorentzNorm(config["manifold"])

        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernorm:
            x = self.layernorm1(x)
        att_output = self.attention(x)
        # INSERT DROPOUT

        if self.residual == "none":
            x = att_output
        if self.residual == "geodesic":
            x = self.manifold.geodesic_residual(x, att_output, weight=self.res_scale_att)
        elif self.residual == "midpoint":
            alpha = float(self.res_scale_att)
            xs2 = torch.stack([x, att_output], dim=-2)
            w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
            x = self.manifold.lorentz_midpoint(xs2, w2)
            
        if self.layernorm:
            x = self.layernorm2(x)
        mlp_output = self.mlp(x)
        # INSERT DROPOUT -> done in mlp via DropConnect

        if self.residual == "none":
            out = mlp_output
        if self.residual == "geodesic":
            out = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)
        elif self.residual == "midpoint":
            alpha = float(self.res_scale_mlp)
            xs2 = torch.stack([x, mlp_output], dim=-2)
            w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
            out = self.manifold.lorentz_midpoint(xs2, w2)

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