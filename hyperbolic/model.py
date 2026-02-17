import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic.lorentz import Lorentz
from hyperbolic.attention import LorentzMHA, LorentzMHAEfficient
from layers.LLinear import LorentzFullyConnected


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # projection layer
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # create learnable cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  

        # prepend cls token to the input
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class LorentzMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1, 
            out_features=config["mlp_dim"] + 1,
            manifold=config["manifold"],
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0)
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1, 
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0)
        )

        # TODO: lorentz dropout, activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.manifold = config["manifold"]
        self.skip_connection = config["skip_connection"]
        
        d = config["hidden_size"]          # spatial dim
        D = d + 1                          # lorentz dim

        self.attention = LorentzMHA(
            input_dim=D,                                   # Lorentz dim
            curvature=config.get("curvature", 0.1),
            num_heads=config["num_attention_heads"],
            compute_scores=config.get("compute_scores", "lorentz_inner"),
            head_mode=config.get("head_mode", "full_midpoint"),
            concat_mode=config.get("concat_mode", "direct"),
            out_dim=d,                                     # spatial out_dim; output is 1+out_dim = D
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0),
            attn_mask=config.get("attn_mask", None),
        )

        self.mlp = LorentzMLP(config)

    def forward(self, x):
        # x is Lorentz: [B, N, D]
        att_output = self.attention(x)  # [B, N, D]

        if self.skip_connection:
            x = self.manifold.geodesic_residual(x, att_output, alpha=0.5)
            # x = self.manifold.lorentz_residual(x, att_output)
        else:
            x = att_output

        mlp_output = self.mlp(x)        # [B, N, D]

        if self.skip_connection:
            mlp_output = self.manifold.geodesic_residual(x, att_output, alpha=0.5)
            # mlp_output = self.manifold.lorentz_residual(x, mlp_output)
        else:
            mlp_output = mlp_output

        return mlp_output


class LorentzEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = (LorentzTransformerBlock(config))
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

class LorentzViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embeddings(config)
        self.encoder = LorentzEncoder(config)
        self.manifold = config["manifold"]
        self.classifier = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,      
            out_features=config["num_classes"] + 1,    
            manifold=config["manifold"],
            reset_params=config.get("reset_params", "lorentz_kaiming"),
            a_default=config.get("a_default", 0.0),
            do_mlr=True
        )

    def forward(self, x):
        x_euc = self.embedding(x)               # [B, N, d]
        x_lor = self.manifold.expmap0(x_euc)    # [B, N, D]
        x_lor = self.encoder(x_lor)             # [B, N, D]
        cls = x_lor[:, 0, :]                    # [B, D]
        logits = self.classifier(cls)           # [B, C]
        return (logits, None)


        