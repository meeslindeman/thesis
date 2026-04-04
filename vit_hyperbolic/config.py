from dataclasses import dataclass

@dataclass
class Config:
    # training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto" 
    seed: int = 0

    # debug / checks
    detect_anomaly: bool = False

    # model (LorentzViT)
    patch_size: int = 4
    hidden_size: int = 48
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    mlp_dim: int = 192
    dropout_rate: float = 0.0
    image_size: int = 32
    num_classes: int = 10
    num_channels: int = 3

    # ablations
    final_op: str = "pooling"
    patch_embed_type: str = "linear"
    embeddings: str = "lorentz"
    residual: str = "geodesic"
    layernorm: bool = False
    dropout: bool = False
    normalization: str = "euclidean_scale"

    # hyperbolic specifics
    curvature: float = 0.1
    head_mode: str = "split_concat"
    concat_mode: str = "logradius" # "direct", "logradius"
    compute_scores: str = "lorentz_inner" # "signed_dist", "lorentz_inner"
    reset_params: str = "lorentz_kaiming"
    a_default: float = 0.0
    pre_exp_scale: float = 0.4

    # wandb
    group: str = "default"
    name: str = "run"