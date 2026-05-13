import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vit_hyperbolic.linear import LorentzFullyConnected
from vit_hyperbolic.blocks.dropout import IntrinsicDropout


class LorentzPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        self.manifold = config["manifold"]

        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        patch_dim = self.patch_size * self.patch_size * self.num_channels

        self.projection = LorentzFullyConnected(
            in_features=patch_dim + 1,
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            use_weight_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]

        x_lor = self.manifold.projection_space_orthogonal(x)
        x_lor = self.projection(x_lor)
        return x_lor


class LorentzEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = LorentzPatchEmbeddings(config)
        self.final_op = config["final_op"]
        self.manifold = config["manifold"]

        if self.final_op == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"] + 1))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

        pos = torch.randn(1, n_tokens, config["hidden_size"]) / config["hidden_size"]**0.5
        self.pos_space = nn.Parameter(pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

        pos_lor = self.manifold.expmap_origin(self.pos_space)
        x = self.manifold.gyro_add(x, pos_lor)
        return x