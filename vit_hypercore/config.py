from dataclasses import dataclass, field
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    # override base defaults (Euclidean optimizer params)
    lr: float = 1e-4
    weight_decay: float = 1e-2   
    dropout_rate: float = 0.1

    # Riemannian optimizer params (ManifoldParameter only)
    # Paper grid-searched hyp LR in {1e-4,3e-4,1e-3,5e-3}, WD in {0,1e-4,1e-3}
    hyp_lr: float = 1e-4
    hyp_weight_decay: float = 1e-3
    stabilize: int = 1   # retract to manifold every N steps; their training script uses 1

    # hyperbolic geometry (curvature=-1 in paper; geoopt uses positive convention)
    curvature: float = 1.0

    # derived — set in __post_init__, not exposed as CLI args
    hidden_channel: int = field(init=False)   # hidden_size // num_attention_heads
    mlp_hidden_size: int = field(init=False)  # mlp_dim + 1  (time coordinate)

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        self.hidden_channel = self.hidden_size // self.num_attention_heads
        self.mlp_hidden_size = self.mlp_dim + 1
