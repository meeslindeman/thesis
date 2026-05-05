import torch
import torch.nn as nn

from vit_hyperbolic.blocks.attention import LorentzMHA
from vit_hyperbolic.blocks.batchnorm import LorentzBatchNorm1d
from vit_hyperbolic.blocks.batchnorm_gyro import LorentzGyroBN1d
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.layernorm = config["layernorm"]

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
            dropout_rate=config["dropout_rate"],
            attn_dropout_rate=config["attn_dropout_rate"],
            use_weight_norm=config["use_weight_norm"]
        )
        self.mlp = LorentzMLP(config)

        self.batchnorm1 = LorentzBatchNorm1d(
            num_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            normalize_variance=False,
        )
        self.batchnorm2 = LorentzBatchNorm1d(
            num_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            normalize_variance=False,
        )

        # self.batchnorm1 = LorentzGyroBN1d(
        #     num_features=config["hidden_size"] + 1,
        #     manifold=config["manifold"],
        #     normalize_variance=False
        # )
        # self.batchnorm2 = LorentzGyroBN1d(
        #     num_features=config["hidden_size"] + 1,
        #     manifold=config["manifold"],
        #     normalize_variance=False
        # )

        # geodesic: weight is t passed through sigmoid, init 0.0 → sigmoid(0)=0.5 (midpoint)
        # lorentz_residual / others: weight is wy used with abs(), init 1.0 (equal-weight midpoint)
        res_scale_init = config["residual_weight_init"]
        self.res_scale_att = nn.Parameter(torch.tensor(res_scale_init))
        self.res_scale_mlp = nn.Parameter(torch.tensor(res_scale_init))
        
        init_delta = config["residual_delta_init"]
        self.delta = nn.Parameter(torch.tensor(float(init_delta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x_normed = self.batchnorm1(x) if self.layernorm else x
        att_output = self.attention(x_normed)

        x = self.manifold.geodesic_residual(residual, att_output, weight=torch.sigmoid(self.res_scale_att))
        x = self.manifold.gyro_scale(x, self.delta)

        residual = x
        x_normed = self.batchnorm2(x) if self.layernorm else x
        mlp_output = self.mlp(x_normed)


        out = self.manifold.geodesic_residual(residual, mlp_output, weight=torch.sigmoid(self.res_scale_mlp))
        out = self.manifold.gyro_scale(out, self.delta)

        return out


class LorentzEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config["num_hidden_layers"]
        self.manifold = config["manifold"]

        self.blocks = nn.ModuleList(
            [LorentzTransformerBlock(config) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
