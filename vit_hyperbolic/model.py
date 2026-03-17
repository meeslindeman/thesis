import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.attention import LorentzMHA
from vit_hyperbolic.lorentz_utils import LorentzDropout, LorentzLayerNorm
from layers.LLinear import LorentzFullyConnected


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
            out_features=self.hidden_size + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]

        # compute time component
        x_lor = self.manifold.projection_space_orthogonal(x, manifold_dim=-1)   # [B, N, patch_dim+1]

        # project to hidden size
        x_lor = self.projection(x_lor)              # [B, N, hidden_size+1]
        return x_lor


class LorentzEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = LorentzPatchEmbeddings(config)
        self.final_op = config["final_op"]
        self.manifold = config["manifold"]
        self.init_mode = config["init_mode"]

        if self.final_op == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"] + 1))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

        # initialize with random points on the unit sphere in the tangent space
        pos = torch.randn(1, n_tokens, config["hidden_size"])
        
        if self.init_mode == "normalized":
            radius = 1.0
            pos = pos / pos.norm(dim=-1, keepdim=True) * radius
        elif self.init_mode == "random":
            pass  # keep as is
        else:
            raise ValueError(f"Invalid init_mode: {self.init_mode}. Must be 'normalized' or 'random'.")

        self.pos_space = nn.Parameter(pos)
        self.dropout = LorentzDropout(config)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)     # [B, 1, D]
            x = torch.cat((cls, x), dim=1)                      # [B, 1+P, D]

        # project position embeddings to manifold
        pos_lor = self.manifold.expmap0(self.pos_space)

        # add position embeddings via gyro_add
        x = self.manifold.gyro_add(x, pos_lor)                  # [B, N, D]
        # x = self.dropout(x)
        return x


# TODO: move to separate file
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

    def forward(self, x):
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

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)   # [B, 1, d]
            x = torch.cat((cls, x), dim=1)           # [B, 1+P, d]

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class LorentzMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1, 
            out_features=config["mlp_dim"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"]
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"]
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.skip_connection = config["skip_connection"]
        self.use_midpoint = config["use_midpoint"]
        self.use_layer_norm = config["use_layer_norm"]
        
        d = config["hidden_size"]          # spatial dim
        D = d + 1                          # lorentz dim

        self.attention = LorentzMHA(
            input_dim=D,                                   
            curvature=config["curvature"],
            num_heads=config["num_attention_heads"],
            compute_scores=config["compute_scores"],
            head_mode=config["head_mode"],
            concat_mode=config["concat_mode"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            attn_mask=config.get("attn_mask", None)
        )

        self.mlp = LorentzMLP(config)
        self.layernorm1 = LorentzLayerNorm(manifold=self.manifold, dim=d)
        self.layernorm2 = LorentzLayerNorm(manifold=self.manifold, dim=d)

        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_norm:
            x = self.layernorm1(x)

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

        if self.use_layer_norm:
            x = self.layernorm2(x)
        mlp_output = self.mlp(x)

        if self.skip_connection:
            if self.use_midpoint:
                alpha = float(self.res_scale_mlp)
                xs2 = torch.stack([x, mlp_output], dim=-2)
                w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
                mlp_output = self.manifold.lorentz_midpoint(xs2, w2)
            else:
                mlp_output = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)

        return mlp_output


class LorentzEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = (LorentzTransformerBlock(config))
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    

class LorentzViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LorentzEncoder(config)
        self.manifold = config["manifold"]
        self.final_op = config["final_op"] 
        self.embedding_space = config["embeddings"]

        if self.embedding_space == "euclidean":
            self.embedding = Embeddings(config)
        elif self.embedding_space == "lorentz":
            self.embedding = LorentzEmbeddings(config)

        self.classifier = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,      
            out_features=config["num_classes"] + 1,    
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            do_mlr=True
        )

        self.k = config["curvature"]
        print(f"Initialized LorentzViT with curvature k={self.k}")

        # euclidean scaling and normalization before expmap to help with stability
        self.pre_scale = nn.Parameter(torch.tensor(0.3))
        self.r_logit = nn.Parameter(torch.tensor(1.0))

    def learned_radius(self):
        return F.softplus(self.r_logit) + 1e-6

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_op == "cls":
            patches = x[:, 1:, :]   # [B, P, D]
            cls = x[:, 0, :]        # [B, D]
            return cls  
        elif self.final_op == "pooling":
            return self.manifold.lorentz_midpoint(x)  # [B, D]
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

    @staticmethod
    def clamp_euclidean_radius(x: torch.Tensor, r_max: float, eps: float = 1e-6):
        norm = x.norm(dim=-1, keepdim=True)
        scale = (r_max / norm.clamp_min(eps)).clamp_max(1.0)
        return x * scale

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        metrics = {} if return_metrics else None

        if self.embedding_space == "euclidean":
            x_euc = self.embedding(x)                   # [B, N, d]

            # clamp radius (optional to just set to any number or keep learned)
            r_max = self.learned_radius()
            x_euc = self.clamp_euclidean_radius(x_euc, r_max) 

            x_lor0 = self.manifold.expmap0(x_euc)       # [B, N, D]

        elif self.embedding_space == "lorentz":
            x_lor0 = self.embedding(x)                  # [B, N, D]

        x_lor = self.encoder(x_lor0)                    # [B, N, D]

        pooled = self.pool_tokens(x_lor)                # [B, D]

        logits = self.classifier(pooled)    
         
        return (logits, metrics) if return_metrics else (logits, None)


        