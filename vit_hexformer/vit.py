import math
import torch
import torch.nn as nn

from .manifold.lorentz import CustomLorentz
from .nn.lfc import LorentzFullyConnected
from .nn.lbn import LorentzLayerNorm
from .nn.lmlr import LorentzMLR
from .nn.transformer import LorentzEmbedding, LorentzTransformerEncoder


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, LorentzFullyConnected):
        nn.init.xavier_normal_(m.weight.weight)
        if m.weight.bias is not None:
            nn.init.constant_(m.weight.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class LorentzViT(nn.Module):
    """Fully hyperbolic ViT (HexFormer architecture, Lorentz encoder)."""

    def __init__(
        self,
        manifold: CustomLorentz,
        num_layers=9,
        img_dim=(3, 32, 32),
        num_classes=100,
        patch_size=4,
        heads=12,
        hidden_dim=192,
        mlp_dim=384,
        dropout=0.0,
        remove_linear=False,
    ):
        super().__init__()
        self.manifold = manifold
        c, h, w = img_dim
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.patch_dim = patch_size ** 2 * c
        self.hidden_dim = hidden_dim
        num_tokens = self.num_patches + 1  # +1 for CLS

        self.embed = LorentzEmbedding(manifold, hidden_dim + 1, self.patch_dim + 1, num_tokens)
        self.encoder = nn.Sequential(
            *[
                LorentzTransformerEncoder(
                    manifold, hidden_dim + 1, mlp_dim + 1, self.num_patches, heads, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = LorentzLayerNorm(manifold, hidden_dim + 1)
        self.predictor = None if remove_linear else LorentzMLR(manifold, hidden_dim, num_classes)

        self.apply(_init_weights)

    def forward(self, x):
        x = self._patchify(x)
        x = self.embed(x)
        x = self.encoder(x)[:, 0]
        x = self.final_ln(x)
        if self.predictor is not None:
            x = self.predictor(x)
        return x

    def _patchify(self, x):
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        return out.reshape(x.size(0), self.num_patches, -1)
