from dataclasses import dataclass, field

DATASET_CONFIGS = {
    # CIFAR / TinyImageNet (main experiments)
    "cifar10":       {"image_size": 32,  "num_classes": 10,  "num_channels": 3},
    "cifar100":      {"image_size": 32,  "num_classes": 100, "num_channels": 3},
    "tiny_imagenet": {"image_size": 64,  "num_classes": 200, "num_channels": 3},
    # IDMM small datasets (224×224, torchvision download)
    "flowers102":    {"image_size": 224, "num_classes": 102, "num_channels": 3},
    "pets":          {"image_size": 224, "num_classes": 37,  "num_channels": 3},
    "dtd":           {"image_size": 224, "num_classes": 47,  "num_channels": 3},
    "aircrafts":     {"image_size": 224, "num_classes": 100, "num_channels": 3},
}

@dataclass
class BaseConfig:
    # training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.1
    device: str = "auto"
    seed: int = 0

    # dataset
    dataset: str = "cifar100"
    image_size: int = field(init=False)
    num_classes: int = field(init=False)
    num_channels: int = field(init=False)

    # early stopping (0 = disabled)
    patience: int = 5

    # model (common to all three variants)
    patch_size: int = 4
    hidden_size: int = 48
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    mlp_dim: int = 192
    dropout_rate: float = 0.1

    # loss
    label_smoothing: float = 0.1

    # scheduler ("none", "cosine", "warmup_cosine")
    scheduler: str = "none"
    warmup_epochs: int = 10

    # gradient clipping (0.0 = disabled)
    grad_clip: float = 0.0

    def __post_init__(self):
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Available: {list(DATASET_CONFIGS)}")
        for k, v in DATASET_CONFIGS[self.dataset].items():
            setattr(self, k, v)
