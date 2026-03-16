from dataclasses import dataclass


@dataclass
class Config:
    # Training
    batch_size:    int   = 256
    epochs:        int   = 30
    lr:            float = 1e-3
    weight_decay:  float = 0.0
    seed:          int   = 0

    # Debug / checks
    detect_anomaly: bool = False
    check_metrics:  bool = False

    # Model (LorentzViT)
    patch_size:          int   = 4
    hidden_size:         int   = 48
    num_hidden_layers:   int   = 4
    num_attention_heads: int   = 4
    mlp_dim:             int   = 192
    dropout_rate:        float = 0.0
    initializer_range:   float = 0.02
    image_size:          int   = 32
    num_classes:         int   = 10
    num_channels:        int   = 3
    qkv_bias:            bool  = True

    # Ablations
    final_op:        str  = "pooling"
    patch_embed_type: str = "linear"
    embeddings:      str  = "lorentz"
    use_midpoint:    bool = False
    use_layer_norm:  bool = False
    init_mode:       str  = "normalized"

    # Hyperbolic specifics
    curvature:       float = 0.1
    head_mode:       str   = "split_concat"
    concat_mode:     str   = "logradius"
    compute_scores:  str   = "lorentz_inner"
    reset_params:    str   = "lorentz_kaiming"
    skip_connection: bool  = False
    a_default:       float = 0.0
