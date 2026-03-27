import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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