import torch
import torch.nn as nn

from .manifold.lorentz import CustomLorentz
from .nn.lmlr import LorentzMLR
from .vit import LorentzViT


class ViTClassifier(nn.Module):
    """Fully hyperbolic ViT classifier (Lorentz encoder + Lorentz decoder)."""

    def __init__(self, enc_kwargs: dict, dec_kwargs: dict):
        super().__init__()
        self.enc_manifold = CustomLorentz(k=enc_kwargs["k"], learnable=enc_kwargs["learn_k"])
        self.dec_manifold = CustomLorentz(k=dec_kwargs["k"], learnable=dec_kwargs["learn_k"])

        self.encoder = LorentzViT(
            manifold=self.enc_manifold,
            remove_linear=True,
            **{k: v for k, v in enc_kwargs.items() if k not in ("k", "learn_k")},
        )
        self.decoder = LorentzMLR(self.dec_manifold, dec_kwargs["embed_dim"], dec_kwargs["num_classes"])

    def _bridge(self, x):
        """Handle curvature mismatch between encoder and decoder manifolds."""
        if self.enc_manifold.k != self.dec_manifold.k:
            temp = self.dec_manifold.rescale_to_max(self.enc_manifold.logmap0(x))
            x = self.dec_manifold.expmap0(temp)
        return x

    def get_embeddings(self, x):
        return self._bridge(self.encoder(x))

    def forward(self, x):
        x = self.encoder(x)
        x = self._bridge(x)
        return self.decoder(x)


def build_hexformer(cfg) -> ViTClassifier:
    """Construct a fully hyperbolic HexFormer ViT from a Config object."""
    enc_kwargs = {
        "k": cfg.curvature,
        "learn_k": cfg.learn_k,
        "num_layers": cfg.num_hidden_layers,
        "img_dim": (cfg.num_channels, cfg.image_size, cfg.image_size),
        "num_classes": cfg.num_classes,
        "patch_size": cfg.patch_size,
        "heads": cfg.num_attention_heads,
        "hidden_dim": cfg.hidden_size,
        "mlp_dim": cfg.mlp_dim,
        "dropout": cfg.dropout_rate,
    }
    dec_kwargs = {
        "k": cfg.curvature,
        "learn_k": cfg.learn_k,
        "embed_dim": cfg.hidden_size,
        "num_classes": cfg.num_classes,
    }
    return ViTClassifier(enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs)
