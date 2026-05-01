import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config["image_size"]
        patch_size = config["patch_size"]
        num_channels = config["num_channels"]
        hidden_size = config["hidden_size"]
        self.patch_embed_type = config["patch_embed_type"]
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        if self.patch_embed_type == "conv":
            self.projection = nn.Conv2d(
                num_channels, hidden_size, kernel_size=patch_size, stride=patch_size,
            )
        elif self.patch_embed_type == "linear":
            patch_dim = patch_size * patch_size * num_channels
            self.projection = nn.Linear(patch_dim, hidden_size)
        else:
            raise ValueError(f"Unknown patch_embed_type: {self.patch_embed_type!r}")

        self.patch_size = patch_size
        self.num_channels = num_channels

    def forward(self, x):
        if self.patch_embed_type == "conv":
            return self.projection(x).flatten(2).transpose(1, 2)
        B, C, H, W = x.shape
        x = x.reshape(B, C, self.grid_size, self.patch_size, self.grid_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.num_patches, -1)
        return self.projection(x)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_op = config["final_op"]
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        hidden_size = config["hidden_size"]

        if self.final_op == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            n_tokens = num_patches + 1
        elif self.final_op == "pooling":
            self.cls_token = None
            n_tokens = num_patches
        else:
            raise ValueError(f"Unknown final_op: {self.final_op!r}")

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.final_op == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        return self.dropout(x + self.position_embeddings)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_size = self.hidden_size // self.num_heads
        self.attn_dropout = config["attn_dropout_rate"]

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config["qkv_bias"])
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, self.hidden_size)
        return self.proj_dropout(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_size"])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config["hidden_size"])
        self.norm2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config["num_hidden_layers"]
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n)])
        self.norm = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.final_op = config["final_op"]
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])
        self._init_weights()

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_op == "cls":
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(self.pool(x))

    def _init_weights(self):
        std = self.config["initializer_range"]
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        if self.embedding.cls_token is not None:
            nn.init.trunc_normal_(self.embedding.cls_token, std=std)
        nn.init.trunc_normal_(self.embedding.position_embeddings, std=std)
