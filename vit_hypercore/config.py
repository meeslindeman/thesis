from dataclasses import dataclass, field

DATASET_CONFIGS = {
    "cifar10":  {"image_size": 32, "num_classes": 10,  "num_channels": 3},
    "cifar100": {"image_size": 32, "num_classes": 100, "num_channels": 3},
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

    # model — same names as vit/ and vit_hyperbolic/
    patch_size: int = 4
    hidden_size: int = 48          # total spatial dim; must be divisible by num_attention_heads
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    mlp_dim: int = 192
    dropout_rate: float = 0.0

    # derived — set in __post_init__, not exposed as CLI args
    hidden_channel: int = field(init=False)   # hidden_size // num_attention_heads
    mlp_hidden_size: int = field(init=False)  # mlp_dim + 1 (time coord)

    # hyperbolic
    curvature: float = 1.0

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

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        self.hidden_channel = self.hidden_size // self.num_attention_heads
        self.mlp_hidden_size = self.mlp_dim + 1
