import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter

from ..manifold.lorentz import CustomLorentz
from .lfc import LorentzFullyConnected
from .lbn import LorentzLayerNorm


# ---- DropPath (Stochastic Depth) -----------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---- Embedding -----------------------------------------------------------
class LorentzEmbedding(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden_dim, patch_dim, num_tokens):
        super().__init__()
        self.manifold = manifold
        self.patch_embed = LorentzFullyConnected(manifold, patch_dim, hidden_dim)
        self.cls_token = ManifoldParameter(
            self.manifold.random_normal(1, 1, hidden_dim), manifold=self.manifold
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim - 1))

    def forward(self, x):
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        x = x.narrow(-1, 1, x.shape[-1] - 1) + self.pos_embed
        return self.manifold.add_time(x)


# ---- Multi-Head Attention ------------------------------------------------
class LorentzMultiHeadAttention(nn.Module):
    def __init__(
        self,
        manifold: CustomLorentz,
        num_features,
        num_patches,
        heads,
        dropout=0.0,
        learn_scale=False,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = (num_features - 1) // heads
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(
            self.head_dim ** (-0.5) * torch.ones((1, heads, 1, 1)),
            requires_grad=learn_scale,
        )
        self.softmax = nn.Softmax(dim=-1)

        self.q = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.k = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.v = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.o = LorentzFullyConnected(manifold, num_features, num_features, dropout=dropout)

    def _expmap_aggregate(self, v, score):
        v_tangent = self.manifold.logmap0(v)
        weighted = torch.matmul(score, v_tangent)
        return self.manifold.expmap0(weighted / (score.sum(dim=-1, keepdim=True) + 1e-8))

    def forward(self, x):
        b, n, _ = x.size()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dists = -self.manifold.csqdist(q, k) * self.scale.expand((b, self.heads, 1, 1))
        score = self.softmax(dists / self.temperature)

        attn = self._expmap_aggregate(v, score).permute(0, 2, 1, 3)

        attn_space = attn.narrow(-1, 1, attn.shape[-1] - 1).reshape(b, n, -1)
        attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
        attn_time_rescaled = torch.sqrt(
            torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k)
        )
        attn = torch.concat((attn_time_rescaled, attn_space), dim=-1)
        return self.o(attn)


# ---- Transformer Encoder Block -------------------------------------------
class LorentzTransformerEncoder(nn.Module):
    def __init__(
        self,
        manifold: CustomLorentz,
        hidden,
        mlp_hidden,
        num_patches,
        heads,
        dropout,
        stochastic_depth=0.1,
    ):
        super().__init__()
        self.manifold = manifold
        self.ln1 = LorentzLayerNorm(manifold, hidden)
        self.mha = LorentzMultiHeadAttention(manifold, hidden, num_patches, heads, dropout)
        self.ln2 = LorentzLayerNorm(manifold, hidden)
        self.mlp = nn.Sequential(
            LorentzFullyConnected(manifold, hidden, mlp_hidden, activation=nn.GELU(), dropout=dropout),
            LorentzFullyConnected(manifold, mlp_hidden, hidden, dropout=dropout),
        )
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        out = self.mha(self.ln1(x))
        out = self.drop_path(out.narrow(-1, 1, x.shape[-1] - 1)) + x.narrow(-1, 1, x.shape[-1] - 1)
        out = (
            self.drop_path(self.mlp(self.ln2(self.manifold.add_time(out))).narrow(-1, 1, x.shape[-1] - 1))
            + out
        )
        return self.manifold.add_time(out)
