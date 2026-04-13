import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.LLinear import LorentzFullyConnected
from vit_hyperbolic.blocks.dropout import IntrinsicDropout


class LorentzPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.normalization = config["normalization"]
        self.pre_exp_scale = config["pre_exp_scale"]

        self.manifold = config["manifold"]

        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        patch_dim = self.patch_size * self.patch_size * self.num_channels

        self.log_scale = nn.Parameter(torch.tensor(math.log(self.pre_exp_scale), dtype=torch.float32)) 

        self.projection = LorentzFullyConnected(
            in_features=patch_dim + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"]
        )

    def normalize_pre_exp(self, x):
        if self.normalization == "none":
            return x # will break
        elif self.normalization == "euclidean_unit":
            return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        elif self.normalization == "euclidean_rms":
            return x / (self.hidden_size ** 0.5)
        elif self.normalization == "euclidean_scale":
            return self.log_scale.exp() * x
        elif self.normalization == "projection":
            return x  # handled separately, no pre-exp normalization
        else:
            raise ValueError(f"Invalid normalization: {self.normalization}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]

        x_euc = self.normalize_pre_exp(x)

        if self.normalization == "projection":
            x_lor = self.manifold.projection_space_orthogonal(x)
        else:
            x_lor = self.manifold.expmap0(x_euc)   

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

        self.dropout = IntrinsicDropout(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

        pos_lor = self.manifold.expmap0(self.pos_space)
        # x = self.dropout(x)
        return x


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed_type = config["patch_embed_type"] 

        if self.patch_embed_type == "conv":
            self.projection = nn.Conv2d(
                self.num_channels,
                self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=True,
            )
        elif self.patch_embed_type == "linear":
            patch_dim = self.patch_size * self.patch_size * self.num_channels
            # insert time component
            self.projection = nn.Linear(patch_dim, self.hidden_size, bias=True)
        else:
            raise ValueError(f"Invalid patch_embed_type: {self.patch_embed_type}. Use 'conv' or 'linear'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        if self.patch_embed_type == "conv":
            x = self.projection(x)              # [B, D, Gh, Gw]
            x = x.flatten(2).transpose(1, 2)    # [B, N, D]
            return x

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]
        x = self.projection(x)                                      # [B, N, D]
        return x
    

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.final_op = config["final_op"] 

        if self.final_op == "cls":
            # create learnable cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

        # create learnable position embeddings with controlled radius
        radius = 1.0
        pos = torch.randn(1, n_tokens, config["hidden_size"])
        pos = pos / pos.norm(dim=-1, keepdim=True) * radius
        self.position_embeddings = nn.Parameter(pos)

        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)   # [B, 1, d]
            x = torch.cat((cls, x), dim=1)           # [B, 1+P, d]

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x