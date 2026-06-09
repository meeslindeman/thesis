import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from . import nn as hnn
from .manifold.lorentz import Lorentz


class HyperbolicMLP(nn.Module):
    def __init__(self, manifold, in_channel, hidden_channel, dropout=0):
        super().__init__()
        self.manifold = manifold
        self.dense_1 = hnn.LorentzLinear(self.manifold, in_channel, hidden_channel - 1)
        self.dense_2 = hnn.LorentzLinear(self.manifold, hidden_channel, in_channel - 1)
        self.activation = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, manifold, in_channel, hidden_channel, dropout=0, num_heads=1, output_attentions=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = hnn.LorentzMultiheadAttention(manifold, in_channel, in_channel, num_heads, attention_type='full', trans_heads_concat=True)
        self.layernorm_1 = hnn.LorentzLayerNorm(manifold, num_heads * in_channel - 1)
        self.layernorm_2 = hnn.LorentzLayerNorm(manifold, num_heads * in_channel - 1)
        self.mlp = HyperbolicMLP(manifold, num_heads * in_channel, hidden_channel, dropout=dropout)
        self.manifold = manifold
        self.residual_1 = hnn.LResNet(manifold, use_scale=True, scale=27.5)
        self.residual_2 = hnn.LResNet(manifold, use_scale=True, scale=27.5)
        self.in_channel = in_channel

    def forward(self, x, output_attentions=False):
        x = self.layernorm_1(x)
        attention_output = self.attention(x, x, output_attentions=output_attentions)
        x = self.residual_1(x, attention_output)
        mlp_output = self.mlp(self.layernorm_2(x))
        x = self.residual_2(x, mlp_output)
        return x


class LViTEncoder(nn.Module):
    def __init__(self, manifold_in, num_layers, in_channel, hidden_channel, num_heads=1, dropout=0, output_attentions=False, manifold_out=None):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.num_heads = num_heads
        for _ in range(num_layers):
            block = LorentzTransformerBlock(manifold_in, in_channel, hidden_channel, dropout, num_heads, output_attentions)
            self.blocks.append(block)
        self.fc = hnn.LorentzLinear(manifold_in, self.num_heads * in_channel, self.num_heads * in_channel - 1, manifold_out=manifold_out)
        self.manifold_out = manifold_out
        self.manifold = manifold_in
        self.in_channel = in_channel

    def forward(self, x, output_attentions=False):
        for block in self.blocks:
            x = block(x, output_attentions=output_attentions)
        return self.fc(x)


class LViT(nn.Module):
    def __init__(self,
                 manifold_in,
                 manifold_hidden,
                 manifold_out,
                 image_size=32,
                 patch_size=4,
                 num_layers=6,
                 in_channel=3,
                 hidden_channel=65,
                 out_channel=10,
                 mlp_hidden_size=65*4,
                 num_heads=4,
                 dropout=0.1,
                 output_attentions=False):
        super().__init__()
        self.in_channel = in_channel + 1
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2
        self.num_heads = num_heads
        self.width = self.num_heads * self.hidden_channel

        self.patch_embedding = hnn.LorentzPatchEmbedding(manifold_in, image_size, patch_size, self.in_channel, self.width - 1)
        self.positional_encoding = ManifoldParameter(self.manifold_in.random_normal((1, self.num_patches, self.width)), manifold=self.manifold_in, requires_grad=True)
        self.add_pos = hnn.LResNet(manifold_in, use_scale=True, scale=3.0)
        self.dropout = hnn.LorentzDropout(self.manifold_hidden, dropout=dropout)
        # projects from manifold_in to manifold_hidden (handles curvature change if different)
        self.fc = hnn.LorentzLinear(manifold_in=manifold_in, manifold_out=self.manifold_hidden, in_features=self.width, out_features=self.width - 1)

        self.encoder = LViTEncoder(self.manifold_hidden, self.num_layers, self.hidden_channel, mlp_hidden_size, num_heads, dropout, output_attentions, manifold_out=manifold_out)

        if self.out_channel > 0:
            self.classifier = hnn.LorentzMLR(self.manifold_out, self.width, self.out_channel)

    def forward(self, x, output_attentions=False):
        x = x.permute(0, 2, 3, 1)
        x_hyp = self.manifold_in.projx(F.pad(x, pad=(1, 0)))
        embedding_output = self.patch_embedding(x_hyp)
        embedding_output = self.dropout(self.add_pos(embedding_output, self.positional_encoding))
        embedding_output = self.fc(embedding_output)
        encoder_output = self.encoder(embedding_output, output_attentions=output_attentions)
        if self.out_channel > 0:
            return self.classifier(self.manifold_out.lorentzian_centroid(encoder_output))
        return self.manifold_out.lorentzian_centroid(encoder_output)
