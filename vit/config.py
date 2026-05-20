from dataclasses import dataclass
from config_base import BaseConfig

@dataclass
class Config(BaseConfig):
    qkv_bias: bool = True
    initializer_range: float = 0.02

    patch_embed_type: str = "conv"    # "linear" | "conv"
    final_op: str = "cls"           # "pooling" | "cls"

    attn_dropout_rate: float = 0.1
    stochastic_depth_rate: float = 0.1  
    dropout_rate: float = 0.0
