import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hyperbolic.lorentz import Lorentz
from hyperbolic.attention import LorentzMHA, LorentzMHAEfficient
from layers.LLinear import LorentzFullyConnected


def _check_finite(t: torch.Tensor, tag: str):
    if not torch.isfinite(t).all():
        finite = t[torch.isfinite(t)]
        fin_min = float(finite.min()) if finite.numel() else float("nan")
        fin_max = float(finite.max()) if finite.numel() else float("nan")
        raise RuntimeError(
            f"NON-FINITE at {tag}: shape={tuple(t.shape)} "
            f"finite_min={fin_min} finite_max={fin_max} "
            f"bad_count={(~torch.isfinite(t)).sum().item()}"
        )

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # projection layer
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.final_op = config.get("final_op", "cls") # "cls" or "mean"

        if self.final_op == "cls":
            # create learnable cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "midpoint":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'midpoint'.")

        # create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, n_tokens, config["hidden_size"]))
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
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0)
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0)
        )

        # TODO: lorentz dropout, activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.skip_connection = config["skip_connection"]
        
        d = config["hidden_size"]          # spatial dim
        D = d + 1                          # lorentz dim

        self.attention = LorentzMHA(
            input_dim=D,                                   # Lorentz dim
            curvature=config.get("curvature", 0.1),
            num_heads=config["num_attention_heads"],
            compute_scores=config.get("compute_scores", "lorentz_inner"),
            head_mode=config.get("head_mode", "full_midpoint"),
            concat_mode=config.get("concat_mode", "direct"),
            out_dim=d,                                     # spatial out_dim; output is 1+out_dim = D
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0),
            attn_mask=config.get("attn_mask", None),
        )

        self.mlp = LorentzMLP(config)
        self.res_scale_att = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))

        self._debug = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._debug: _check_finite(x, "block.in")

        att_output = self.attention(x)
        if self._debug: _check_finite(att_output, "block.attention.out")

        if self.skip_connection:
            x2 = self.manifold.geodesic_residual(x, att_output, weight=self.res_scale_att)
            if self._debug: _check_finite(x2, "block.residual1(out of geodesic_residual)")
            x = x2
        else:
            x = att_output

        if self._debug: _check_finite(x, "block.after_skip1")

        mlp_output = self.mlp(x)
        if self._debug: _check_finite(mlp_output, "block.mlp.out")

        if self.skip_connection:
            y = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)
            # TODO: maybe look at step size here
            if self._debug: _check_finite(y, "block.residual2(out of geodesic_residual)")
            mlp_output = y

        if self._debug: _check_finite(mlp_output, "block.out")
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
        self.embedding = Embeddings(config)
        self.encoder = LorentzEncoder(config)
        self.manifold = config["manifold"]
        self.final_op = config.get("final_op", "cls") # "cls" or "midpoint"
        self.classifier = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,      
            out_features=config["num_classes"] + 1,    
            manifold=config["manifold"],
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0),
            do_mlr=True
        )

        self.k = config.get("curvature", 0.1)
        print(f"Initialized LorentzViT with curvature k={self.k}")

        self.pre_ln = nn.LayerNorm(config["hidden_size"], elementwise_affine=True)
        self.pre_scale = nn.Parameter(torch.tensor(0.1))

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_op == "cls":
            patches = x[:, 1:, :]   # [B, P, D]
            cls = x[:, 0, :]        # [B, D]
            return cls  
        elif self.final_op == "midpoint":
            return self.manifold.lorentz_midpoint(x)  # [B, D]
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'midpoint'.")

    @staticmethod
    def _lorentz_util_metrics(x_lor: torch.Tensor, k: float, eps: float = 1e-6) -> dict:
        """
        x_lor: [B, N, D] or [B, D] with time coordinate at index 0.
        Returns scalar metrics indicating how far points are from origin.
        """
        k = float(k)
        with torch.no_grad():
            if x_lor.dim() == 2:
                x0 = x_lor[:, 0]
                xs = x_lor[:, 1:]
            else:
                x0 = x_lor[..., 0]
                xs = x_lor[..., 1:]

            sqrt_k = float(k) ** 0.5

            # Correct acosh argument
            arg = (sqrt_k * x0).clamp_min(1.0 + eps)

            r = torch.acosh(arg) / sqrt_k

            xs_norm = torch.linalg.norm(xs, dim=-1)

            return {
                "r_mean": r.mean().item(),
                "r_std": r.std(unbiased=True).item(),
                "xs_norm_mean": xs_norm.mean().item(),
                "xs_norm_std": xs_norm.std(unbiased=True).item(),
                "r_p95": torch.quantile(r.flatten(), 0.95).item(),
            }

    def print_stats(self, name, t):
        t = t.detach()
        print(f"{name}: mean={t.mean().item():.4f}, std={t.std(unbiased=True).item():.4f}, "
            f"p95={torch.quantile(t.flatten(), 0.95).item():.4f}, max={t.max().item():.4f}")

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        metrics = {} if return_metrics else None

        x_euc = self.embedding(x)                    # [B, N, d]

        # x_euc = self.pre_ln(x_euc) * self.pre_scale  # [B, N, d]

        # L2 normalization
        x_euc = x_euc / (x_euc.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        # t = x_euc.norm(dim=-1) 
        # self.print_stats("pre_expmap0_token_norm", t)

        x_lor0 = self.manifold.expmap0(x_euc)        # [B, N, D]
        if return_metrics:
            metrics["after_expmap0"] = self._lorentz_util_metrics(x_lor0, self.k)

        x_lor = self.encoder(x_lor0)                 # [B, N, D]
        if return_metrics:
            metrics["after_encoder"] = self._lorentz_util_metrics(x_lor, self.k)

        pooled = self.pool_tokens(x_lor)             # [B, D]
        if return_metrics:
            metrics["after_pool"] = self._lorentz_util_metrics(pooled, self.k)

        logits = self.classifier(pooled)             # [B, C(+1 depending on your head)]
        return (logits, metrics) if return_metrics else (logits, None)


        