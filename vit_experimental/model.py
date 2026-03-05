import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vit_experimental.lorentz import Lorentz
from vit_experimental.attention import LorentzMHA
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
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected {(self.image_size, self.image_size)} but got {(H, W)}")

        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)   # [B, C, Gh, p, Gw, p]
        x = x.permute(0, 2, 4, 3, 5, 1)                             # [B, Gh, Gw, p, p, C]
        x = x.reshape(B, self.num_patches, p * p * C)               # [B, N, patch_dim]

        # add time component for Lorentz input
        time = torch.zeros_like(x[..., :1])         # [B, N, 1]
        x_amb = torch.cat((time, x), dim=-1)        # [B, N, patch_dim+1]
        x_lor = self.manifold.projx(x_amb, dim=-1)  # [B, N, patch_dim+1]
        # print(f"Patch embedding input: mean={x_lor.mean().item():.4f}, std={x_lor.std(unbiased=True).item():.4f}")

        x_lor2 = self.manifold.projection_space_orthogonal(x, )
        # print(f"Patch embedding orthogonal projection: mean={x_lor2.mean().item():.4f}, std={x_lor2.std(unbiased=True).item():.4f}")
        # self.manifold._is_on_manifold(x_lor, "patch embedding input")

        # project to hidden size
        x_lor = self.projection(x_lor)              # [B, N, hidden_size+1]
        # self.manifold._is_on_manifold(x_lor, "patch embedding after projection")
        return x_lor


class LorentzEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = LorentzPatchEmbeddings(config)
        self.final_op = config.get("final_op", "pooling") # "cls" or "pooling"

        self.manifold = config["manifold"]

        if self.final_op == "cls":
            # create learnable cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"] + 1))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

        # init = 3e-3
        # self.pos_space = nn.Parameter(init * torch.randn(1, n_tokens, config["hidden_size"]))

        pos = torch.randn(1, n_tokens, config["hidden_size"])
        pos = pos / pos.norm(dim=-1, keepdim=True)
        self.pos_space = nn.Parameter(pos)  # [1, N, d]

        self.dropout = LorentzDropout(config)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        if self.final_op == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)   # [B, 1, D]
            x = torch.cat((cls, x), dim=1)           # [B, 1+P, D]

        # move pos emb to manifold
        # time = torch.zeros_like(self.pos_space[..., :1])        # [1, N, 1]
        # pos_amb = torch.cat((time, self.pos_space), dim=-1)     # [1, N, hidden_size+1]
        # pos_lor = self.manifold.projx(pos_amb, dim=-1)          # [1, N, hidden_size+1]
        pos_lor = self.manifold.expmap0(self.pos_space)
        # self.manifold._is_on_manifold(pos_lor, "position embedding on manifold")

        x = self.manifold.gyro_add(x, pos_lor)                  # [B, N, D]
        # self.manifold._is_on_manifold(x, "after adding pos emb")
        # x = self.dropout(x)
        return x

class LorentzDropout(nn.Module):
    """
    Lorentz dropout via 'mask & project'.
    Shi, et al. (Intrinsic)
    """
    def __init__(self, config):
        super().__init__()
        self.manifold = config["manifold"]
        self.dropout_rate = config["dropout_rate"]
        self.eps = 1e-6
        self.drop_time = False

    def forward(self, x):
        if (not self.training) or self.dropout_rate == 0.0:
            return x
        if x.size(-1) < 2:
            raise ValueError("LorentzDropout expects last dim to be (1+d) >= 2")

        keep_prob = 1.0 - self.dropout_rate
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        if not self.drop_time:
            mask[..., 0] = 1.0

        x_tilde = x * mask / (keep_prob + self.eps)
        x_dropped = self.manifold.projx(x_tilde, dim=-1)
        return x_dropped

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

        self.patch_embed_type = config.get("patch_embed_type", "linear")  # "conv" or "linear"

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
        self.final_op = config.get("final_op", "pooling") # "cls" or "pooling"

        if self.final_op == "cls":
            # create learnable cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
            n_tokens = self.patch_embeddings.num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = self.patch_embeddings.num_patches
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

        # create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, n_tokens, config["hidden_size"]))
        # change to manifold aware
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

        self._debug = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._debug: 
            _check_finite(x, "block.in")


        # self.manifold._is_on_manifold(x, "input x")

        att_output = self.attention(x)
        if self._debug: 
            _check_finite(att_output, "block.attention.out")

        if self.skip_connection:
            # print("===== ATTENTION RESIDUAL =====")
            # self.manifold._is_on_manifold(att_output, "att_output")
            x2 = self.manifold.geodesic_residual(x, att_output, weight=self.res_scale_att)
            if self._debug: 
                _check_finite(x2, "block.residual1(out of geodesic_residual)")
            x = x2
        else:
            x = att_output

        if self._debug: _check_finite(x, "block.after_skip1")

        mlp_output = self.mlp(x)
        if self._debug: _check_finite(mlp_output, "block.mlp.out")

        if self.skip_connection:
            # print("===== MLP RESIDUAL =====")    
            # self.manifold._is_on_manifold(att_output, "mlp_output")
            y = self.manifold.geodesic_residual(x, mlp_output, weight=self.res_scale_mlp)
            if self._debug: 
                _check_finite(y, "block.residual2(out of geodesic_residual)")
            mlp_output = y

        if self._debug: 
            _check_finite(mlp_output, "block.out")
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
        self.final_op = config.get("final_op", "pooling") # "cls" or "pooling"
        self.embedding_space = config.get("embeddings", "lorentz")

        if self.embedding_space == "euclidean":
            self.embedding = Embeddings(config)
        elif self.embedding_space == "lorentz":
            self.embedding = LorentzEmbeddings(config)

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

    @staticmethod
    def print_stats(name, t):
        t = t.detach()
        print(f"{name}: mean={t.mean().item():.4f}, std={t.std(unbiased=True).item():.4f}, "
            f"p95={torch.quantile(t.flatten(), 0.95).item():.4f}, max={t.max().item():.4f}")
    
    @staticmethod
    def clamp_euclidean_radius(x: torch.Tensor, r_max: float, eps: float = 1e-6):
        norm = x.norm(dim=-1, keepdim=True)
        scale = (r_max / norm.clamp_min(eps)).clamp_max(1.0)
        return x * scale

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        metrics = {} if return_metrics else None

        if self.embedding_space == "euclidean":
            x_euc = self.embedding(x)                    # [B, N, d]

            # clamp radius (optional to just set to any number)
            # r_max = self.learned_radius()
            # x_euc = self.clamp_euclidean_radius(x_euc, r_max) 

            # pre-layernorm and scaling
            # x_euc = self.pre_ln(x_euc) * self.pre_scale  # [B, N, d]

            # t = x_euc.norm(dim=-1) 
            # self.print_stats("pre_expmap0_token_norm", t)

            x_lor0 = self.manifold.expmap0(x_euc)        # [B, N, D]

            if return_metrics:
                metrics["after_expmap0"] = self._lorentz_util_metrics(x_lor0, self.k)

        elif self.embedding_space == "lorentz":
            x_lor0 = self.embedding(x)                    # [B, N, D]

            if return_metrics:
                metrics["after_embedding"] = self._lorentz_util_metrics(x_lor0, self.k)
        
        # self.manifold._is_on_manifold(x_lor0, "input to encoder")

        x_lor = self.encoder(x_lor0)                 # [B, N, D]
        if return_metrics:
            metrics["after_encoder"] = self._lorentz_util_metrics(x_lor, self.k)

        pooled = self.pool_tokens(x_lor)             # [B, D]
        if return_metrics:
            metrics["after_pool"] = self._lorentz_util_metrics(pooled, self.k)

        logits = self.classifier(pooled)             # [B, C(+1 depending on your head)]
        return (logits, metrics) if return_metrics else (logits, None)


        