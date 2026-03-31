import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from vit_hyperbolic.patches import Embeddings

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.attention import LorentzMHA
from vit_hyperbolic.layernorm import LorentzBatchNorm, LorentzSplitNorm
from vit_hyperbolic.dropout import LorentzDropout
from layers.LLinear import LorentzFullyConnected


class LorentzPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.normalization = config["normalization"]

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
        self.batchnorm = LorentzBatchNorm(manifold=config["manifold"], dim=self.hidden_size + 1)
        self.batchnorm1d = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]

        if self.normalization == "euclidean":
            # x = x / self.hidden_size**0.5  # option 1
            x = x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-6))    # option 2
            x_lor = self.manifold.expmap0(x)                                       

        elif self.normalization == "lorentz":
            x_lor = self.manifold.expmap0(x)                                        # [B, N, patch_dim+1]
            x_lor = self.batchnorm(x_lor)                                           # [B, N, patch_dim+1]

        x_lor = self.projection(x_lor)                                          # [B, N, hidden_size+1]
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

        d = config["hidden_size"]
        pos = torch.randn(1, n_tokens, d) / d**0.5
        
        self.pos_space = nn.Parameter(pos)
        self.dropout = LorentzDropout(config)

    def forward(self, x: torch.Tensor, return_metrics: bool = False) -> torch.Tensor:
        metrics = {} if return_metrics else None

        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

        pos_lor = self.manifold.expmap0(self.pos_space)
        x = self.manifold.gyro_add(x, pos_lor)

        if return_metrics:
            r = self.manifold.radius_metrics(x)
            metrics["embed/r_mean"] = r["mean"]
            metrics["embed/r_max"] = r["max"]

            pos_norm = self.pos_space.detach().norm(dim=-1)
            metrics["params/pos_space_norm_mean"] = float(pos_norm.mean().cpu())
            metrics["params/pos_space_norm_max"] = float(pos_norm.max().cpu())

        x = self.dropout(x)
        return x, metrics


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
        self.dropout = LorentzDropout(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.skip_connection = config["skip_connection"]
        self.use_midpoint = config["use_midpoint"]
        self.use_layernorm = config["use_layernorm"]
        
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
            attn_mask=config.get("attn_mask", None),
            dropout_rate=config["dropout_rate"]
        )

        self.mlp = LorentzMLP(config)

        self.batchnorm1 = LorentzBatchNorm(manifold=self.manifold, dim=D)
        self.batchnorm2 = LorentzBatchNorm(manifold=self.manifold, dim=D)

        self.layernorm1 = LorentzSplitNorm(dim=d)
        self.layernorm2 = LorentzSplitNorm(dim=d)

        self.dropout = LorentzDropout(config)

        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, return_metrics: bool = False, block_idx: int = None) -> torch.Tensor:
        prefix = f"block{block_idx}" if block_idx is not None else "block"
        metrics = {} if return_metrics else None

        if return_metrics:
            r_in = self.manifold.radius_metrics(x)
            metrics[f"{prefix}/r_in_mean"] = r_in["mean"]
            metrics[f"{prefix}/r_in_max"] = r_in["max"]

        if self.use_layernorm:
            x = self.layernorm1(x)

        att_output = self.attention(x)
        att_output = self.dropout(att_output)

        if self.skip_connection:
            if self.use_midpoint:
                alpha = float(self.res_scale_att)
                xs2 = torch.stack([x, att_output], dim=-2)
                w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
                x = self.manifold.lorentz_midpoint(xs2, w2)
                if return_metrics:
                    metrics[f"{prefix}/att_update_mean"] = float("nan")
                    metrics[f"{prefix}/att_update_max"] = float("nan")
            else:
                x = self.manifold.geodesic_residual(x, att_output, weight=self.res_scale_att)
                if return_metrics:
                    v_att = self.manifold.logmap(x.float(), att_output.float()).float()
                    v_att = self.manifold.proju(x.float(), v_att, dim=-1)
                    v_att = v_att * self.res_scale_att.float() if self.res_scale_att is torch.nn.Parameter else v_att * self.res_scale_att
                    upd = self.manifold.update_metrics(v_att)
                    metrics[f"{prefix}/att_update_mean"] = upd["mean"]
                    metrics[f"{prefix}/att_update_max"] = upd["max"]
                    metrics[f"{prefix}/res_scale_att"] = float(self.res_scale_att)
        else:
            x = att_output
            if return_metrics:
                metrics[f"{prefix}/att_update_mean"] = float("nan")
                metrics[f"{prefix}/att_update_max"] = float("nan")

        if self.use_layernorm:
            x = self.layernorm2(x)

        mlp_output = self.mlp(x)

        if self.skip_connection:
            if self.use_midpoint:
                alpha = float(self.res_scale_mlp)
                xs2 = torch.stack([x, mlp_output], dim=-2)
                w2 = torch.tensor([1.0 - alpha, alpha], device=x.device, dtype=x.dtype)
                out = self.manifold.lorentz_midpoint(xs2, w2)

                if return_metrics:
                    metrics[f"{prefix}/mlp_update_mean"] = float("nan")
                    metrics[f"{prefix}/mlp_update_max"] = float("nan")
            else:
                if return_metrics:
                    v_mlp = self.manifold.logmap(x.float(), mlp_output.float()).float()
                    v_mlp = self.manifold.proju(x.float(), v_mlp, dim=-1)
                    v_mlp = v_mlp * self.res_scale_mlp.float() if self.res_scale_mlp is torch.nn.Parameter else v_mlp * self.res_scale_mlp
                    upd = self.manifold.update_metrics(v_mlp)
                    metrics[f"{prefix}/mlp_update_mean"] = upd["mean"]
                    metrics[f"{prefix}/mlp_update_max"] = upd["max"]
                    metrics[f"{prefix}/res_scale_mlp"] = float(self.res_scale_mlp)
                out = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)
        else:
            out = mlp_output
            if return_metrics:
                metrics[f"{prefix}/mlp_update_mean"] = float("nan")
                metrics[f"{prefix}/mlp_update_max"] = float("nan")

        if return_metrics:
            r_out = self.manifold.radius_metrics(out)
            metrics[f"{prefix}/r_out_mean"] = r_out["mean"]
            metrics[f"{prefix}/r_out_max"] = r_out["max"]

        return out, metrics


class LorentzEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [LorentzTransformerBlock(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        metrics = {} if return_metrics else None

        for idx, block in enumerate(self.blocks):
            x, block_metrics = block(x, return_metrics=return_metrics, block_idx=idx)
            if return_metrics:
                metrics.update(block_metrics)

        return x, metrics
    

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
            r_max = self.learned_radius()
            x_euc = self.clamp_euclidean_radius(x_euc, r_max) 
            x_lor0 = self.manifold.expmap0(x_euc)       # [B, N, D]

            if return_metrics:
                r = self.manifold.radius_metrics(x_lor0)
                metrics["embed/r_mean"] = r["mean"]
                metrics["embed/r_max"] = r["max"]

        elif self.embedding_space == "lorentz":
            x_lor0, emb_metrics = self.embedding(x, return_metrics=return_metrics)                 
            if return_metrics:
                metrics.update(emb_metrics)

        x_lor, enc_metrics = self.encoder(x_lor0, return_metrics=return_metrics)

        if return_metrics:
            metrics.update(enc_metrics)

            r_enc = self.manifold.radius_metrics(x_lor)
            metrics["model/r_encoder_mean"] = r_enc["mean"]
            metrics["model/r_encoder_max"] = r_enc["max"]

        pooled = self.pool_tokens(x_lor)

        if return_metrics:
            r_pool = self.manifold.radius_metrics(pooled)
            metrics["model/r_pool_mean"] = r_pool["mean"]
            metrics["model/r_pool_max"] = r_pool["max"]

        logits = self.classifier(pooled)

        if return_metrics:
            metrics["model/logits_abs_max"] = float(logits.detach().abs().max().cpu())

        return (logits, metrics) if return_metrics else (logits, None)


        