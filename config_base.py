from dataclasses import dataclass, field

DATASET_CONFIGS = {
    # CIFAR / TinyImageNet (main experiments)
    "cifar10": {"image_size": 32, "num_classes": 10, "num_channels": 3},
    "cifar100": {"image_size": 32, "num_classes": 100, "num_channels": 3},
    "tiny_imagenet": {"image_size": 64, "num_classes": 200, "num_channels": 3},
    # IDMM small datasets (224×224, torchvision download)
    "flowers102": {"image_size": 224, "num_classes": 102, "num_channels": 3},
    "pets": {"image_size": 224, "num_classes": 37, "num_channels": 3},
    "dtd": {"image_size": 224, "num_classes": 47, "num_channels": 3},
    "aircrafts": {"image_size": 224, "num_classes": 100, "num_channels": 3},
}


@dataclass
class Config:
    # Shared: training
    batch_size: int = 128
    epochs: int = 300
    lr: float = 1.0e-3
    weight_decay: float = 0.1
    device: str = "auto"
    seed: int = 0
    patience: int = 0  # early stopping (0 = disabled)
    grad_clip: float = 0.0
    label_smoothing: float = 0.1

    # Shared: dataset
    dataset: str = "cifar100"
    augmentation: str = "simple"  # simple or advanced
    image_size: int = field(init=False)
    num_classes: int = field(init=False)
    num_channels: int = field(init=False)

    # Shared: optimizer and scheduler
    optimizer: str = "adamw"  # adamw or riemannian
    scheduler: str = "warmup_cosine"  # none, cosine, warmup_cosine, step
    warmup_epochs: int = 10
    lr_step_size: int = 30  # [hypercore] StepLR period
    lr_gamma: float = 0.1  # [hypercore] StepLR decay factor

    # Shared: architecture
    patch_size: int = 4
    hidden_size: int = 192
    num_hidden_layers: int = 12
    num_attention_heads: int = 3
    mlp_dim: int = 768
    patch_embed_type: str = "linear"  # linear or conv
    final_op: str = "pooling"  # pooling or cls
    dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.1
    attn_dropout_rate: float = 0.1

    # Euclidean ViT
    qkv_bias: bool = True
    initializer_range: float = 0.02    

    # Hyperbolic
    curvature: float = 1.0

    # LorentzViT
    residual_weight_init: float = 0.0
    residual_delta_init: float = 2.0
    layernorm: bool = True
    concat_mode: str = "direct"  # direct or logradius
    compute_scores: str = "lorentz_inner"  # lorentz_inner or signed_dist
    reset_params: str = "xavier"
    a_default: float = 0.05
    use_weight_norm: bool = False

    # HyperCore
    hyp_lr: float = 3e-4
    hyp_weight_decay: float = 1e-4
    stabilize: int = 1

    def __post_init__(self):
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. Available: {list(DATASET_CONFIGS)}"
            )
        for k, v in DATASET_CONFIGS[self.dataset].items():
            setattr(self, k, v)


# Alias so existing code (idmm, adversarial) that imports BaseConfig keeps working
BaseConfig = Config
