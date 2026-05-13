from dataclasses import dataclass, field
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    # override base defaults (Euclidean optimizer params)
    lr: float = 3e-4          # 1e-4 (paper, bs=64) scaled to bs=128, snapped to grid
    weight_decay: float = 1e-2
    dropout_rate: float = 0.1

    # Riemannian optimizer params (ManifoldParameter only)
    # Paper grid-searched hyp LR in {1e-4,3e-4,1e-3,5e-3}, WD in {0,1e-4,1e-3}
    hyp_lr: float = 3e-4      # same batch-size scaling as euc_lr
    hyp_weight_decay: float = 1e-4
    stabilize: int = 1   # retract to manifold every N steps; their training script uses 1

    # scheduler override: paper uses StepLR
    scheduler: str = "step"
    lr_step_size: int = 30   # StepLR: epochs between LR drops (paper uses 30)
    lr_gamma: float = 0.1    # StepLR decay factor (paper uses 0.1)

    # hyperbolic geometry (curvature=-1 in paper; geoopt uses positive convention)
    curvature: float = 1.0

    # derived — set in __post_init__, not exposed as CLI args
    hidden_channel: int = field(init=False)   # hidden_size // num_attention_heads (full Lorentz dim per head, incl. time)
    mlp_hidden_size: int = field(init=False)  # mlp_dim // num_attention_heads (per-head; LViT multiplies by num_heads internally)

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.mlp_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"mlp_dim ({self.mlp_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        self.hidden_channel = self.hidden_size // self.num_attention_heads
        self.mlp_hidden_size = self.mlp_dim // self.num_attention_heads
