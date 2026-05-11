import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_hyperbolic.blocks.embeddings import LorentzEmbeddings
from vit_hyperbolic.blocks.encoder import LorentzEncoder
from layers.LLinear import LorentzFullyConnected


class LorentzViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LorentzEncoder(config)
        self.manifold = config["manifold"]
        self.final_op = config["final_op"]

        self.embedding = LorentzEmbeddings(config)

        self.classifier = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,
            out_features=config["num_classes"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            do_mlr=True
        )

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_op == "cls":
            return x[:, 0, :]
        elif self.final_op == "pooling":
            return self.manifold.lorentz_midpoint(x)
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled Lorentz representation before classification."""
        x = self.embedding(x)
        x = self.encoder(x)
        return self.pool_tokens(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.classifier(pooled)
