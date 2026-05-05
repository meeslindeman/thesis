from dataclasses import dataclass
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    # override base defaults — hyperbolic manifold ops are sensitive to high LR
    lr: float = 1e-4
    dropout_rate: float = 0.0

    # shared with euclidean but not in base (attention dropout, patch settings)
    attn_dropout_rate: float = 0.1
    final_op: str = "pooling"
    patch_embed_type: str = "linear"

    # optimizer
    optimizer: str = "adamw"    # adamw | riemannian

    # residual connection scaling
    residual_weight_init: float = 0.0
    residual_delta_init: float = 2.0

    # optional LorentzBatchNorm in transformer blocks
    layernorm: bool = False

    # hyperbolic geometry
    curvature: float = 1.0
    head_mode: str = "split_concat"    # "split_concat" | "full_midpoint"
    concat_mode: str = "direct"        # "direct" | "logradius"
    compute_scores: str = "lorentz_inner"  # "lorentz_inner" | "signed_dist"

    # layer initialisation
    reset_params: str = "lorentz_kaiming"
    a_default: float = 0.0
    use_weight_norm: bool = False
