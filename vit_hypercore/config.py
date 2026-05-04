from dataclasses import dataclass, field
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    # override base default — hypercore layers run without input dropout
    dropout_rate: float = 0.0

    # override: hypercore has ManifoldParameters that require a Riemannian optimizer
    optimizer: str = "riemannian"

    # hyperbolic geometry
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
