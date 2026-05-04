from dataclasses import dataclass
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    # euclidean-specific
    attn_dropout_rate: float = 0.1
    qkv_bias: bool = True
    initializer_range: float = 0.02
    patch_embed_type: str = "linear"   # "linear" | "conv"
    final_op: str = "pooling"          # "pooling" | "cls"
