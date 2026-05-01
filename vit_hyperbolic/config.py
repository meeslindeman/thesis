from dataclasses import dataclass, field

DATASET_CONFIGS = {
    "cifar10":  {"image_size": 32,  "num_classes": 10,  "num_channels": 3},
    "cifar100": {"image_size": 32,  "num_classes": 100, "num_channels": 3},
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

    # early stopping (0 = disabled)
    patience: int = 0

    # model (LorentzViT)
    patch_size: int = 4
    hidden_size: int = 48
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    mlp_dim: int = 192
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.1

    # ablations
    final_op: str = "pooling"
    patch_embed_type: str = "linear"
    residual_weight_init: float = 0.0 
    residual_delta_init: float = 2.0
    layernorm: bool = False

    # hyperbolic specifics
    curvature: float = 1.0
    head_mode: str = "split_concat"
    concat_mode: str = "direct"
    compute_scores: str = "lorentz_inner"

    # layer init
    reset_params: str = "lorentz_kaiming"
    a_default: float = 0.0
    use_weight_norm: bool = False

    # loss
    label_smoothing: float = 0.1

    # scheduler ("none", "cosine", "warmup_cosine")
    scheduler: str = "none"
    warmup_epochs: int = 5

    def __post_init__(self):
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Available: {list(DATASET_CONFIGS)}")
        for k, v in DATASET_CONFIGS[self.dataset].items():
            setattr(self, k, v)