from dataclasses import dataclass, field

DATASET_CONFIGS = {
    "cifar10":  {"image_size": 32,  "num_classes": 10,  "num_channels": 3},
    "cifar100": {"image_size": 32,  "num_classes": 100, "num_channels": 3},
    # "tiny_imagenet": {"image_size": 64, "num_classes": 200, "num_channels": 3},
}

@dataclass
class Config:
    # training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 0

    # dataset
    dataset: str = "cifar10"
    image_size: int = field(init=False)
    num_classes: int = field(init=False)
    num_channels: int = field(init=False)

    # debug / checks
    detect_anomaly: bool = False
    check_metrics: bool = False
    check_nan: bool = False

    # model (LorentzViT)
    patch_size: int = 4
    hidden_size: int = 48
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    mlp_dim: int = 192
    dropout_rate: float = 0.0

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
    concat_mode: str = "logradius"
    compute_scores: str = "lorentz_inner"

    # layer init
    reset_params: str = "lorentz_kaiming"
    a_default: float = 0.0
    use_weight_norm: bool = False

    # embedding init
    pre_exp_scale: float = 0.4

    # wandb
    group: str = "default"
    name: str = "run"

    def __post_init__(self):
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Available: {list(DATASET_CONFIGS)}")
        for k, v in DATASET_CONFIGS[self.dataset].items():
            setattr(self, k, v)