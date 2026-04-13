import torch
import torch.nn as nn

from vit_hyperbolic.blocks.attention import LorentzMHA
from vit_hyperbolic.blocks.layernorm import LorentzRMSNorm
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
            activation=nn.functional.relu,
            use_weight_norm=config["use_weight_norm"]
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            activation=nn.Identity(),
            use_weight_norm=config["use_weight_norm"]
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
            use_weight_norm=config["use_weight_norm"],
            attn_mask=config.get("attn_mask", None),
            dropout_rate=config["dropout_rate"]
        )
        self.mlp = LorentzMLP(config)
        self.layernorm1 = LorentzRMSNorm(config)
        self.layernorm2 = LorentzRMSNorm(config)

        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_block_in = x

        residual = x 
        if self.layernorm:
            x_normed = self.layernorm1(x)
        else:
            x_normed = x
        x_att_base = x

        att_output = self.attention(x_normed)

        if self.residual == "none":
            x = att_output
        elif self.residual == "tangent":
            x = self.manifold.tangent_residual(residual, att_output, weight=self.res_scale_att)
        elif self.residual == "geodesic":
            x = self.manifold.geodesic_residual(residual, att_output, weight=self.res_scale_att)
        elif self.residual == "gyroadd":
            x = self.manifold.gyro_add(residual, att_output)
            # x = self.layernorm1(x)
        elif self.residual == "midpoint":
            xs2 = torch.stack([residual, att_output], dim=-2)
            alpha = torch.sigmoid(self.res_scale_att)  # enforce (0,1)
            w2 = torch.stack([1.0 - alpha, alpha])
            x = self.manifold.lorentz_midpoint(xs2)
        else:
            raise ValueError(f"Invalid residual type: {self.residual}")
            
        residual = x
        if self.layernorm:
            x_normed = self.layernorm2(x)
        else:
            x_normed = x
        x_mlp_base = x

        mlp_output = self.mlp(x_normed)

        if self.residual == "none":
            out = mlp_output
        elif self.residual == "tangent":
            out = self.manifold.tangent_residual(residual, mlp_output, weight=self.res_scale_mlp)
        elif self.residual == "geodesic":
            out = self.manifold.geodesic_residual(residual, mlp_output, weight=self.res_scale_mlp)
        elif self.residual == "gyroadd":
            out = self.manifold.gyro_add(residual, mlp_output)
            # out = self.layernorm1(out)
        elif self.residual == "midpoint":
            xs2 = torch.stack([residual, mlp_output], dim=-2)
            alpha = torch.sigmoid(self.res_scale_att)  # enforce (0,1)
            w2 = torch.stack([1.0 - alpha, alpha])
            out = self.manifold.lorentz_midpoint(xs2)
        else:
            raise ValueError(f"Invalid residual type: {self.residual}")

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