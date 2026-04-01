import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_hyperbolic.blocks.embeddings import Embeddings, LorentzEmbeddings
from vit_hyperbolic.blocks.encoder import LorentzEncoder
from layers.LLinear import LorentzFullyConnected


class LorentzViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LorentzEncoder(config)
        self.manifold = config["manifold"]
        self.final_op = config["final_op"] 
        self.embedding_space = config["embeddings"]

        if self.embedding_space == "euclidean":
            self.embedding = Embeddings(config)
        elif self.embedding_space == "lorentz":
            self.embedding = LorentzEmbeddings(config)

        self.classifier = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,      
            out_features=config["num_classes"] + 1,    
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
            do_mlr=True
        )

        self.k = config["curvature"]
        print(f"Initialized LorentzViT with curvature k={self.k}")

        self.r_logit = nn.Parameter(torch.tensor(1.0))

    def learned_radius(self):
        return F.softplus(self.r_logit) + 1e-6

    def pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_op == "cls":
            return x[:, 0, :]      
        elif self.final_op == "pooling":
            return self.manifold.lorentz_midpoint(x)  
        else:
            raise ValueError(f"Invalid final_op: {self.final_op}. Must be 'cls' or 'pooling'.")

    @staticmethod
    def clamp_euclidean_radius(x: torch.Tensor, r_max: float, eps: float = 1e-6):
        norm = x.norm(dim=-1, keepdim=True)
        scale = (r_max / norm.clamp_min(eps)).clamp_max(1.0)
        return x * scale

    def forward(self, x: torch.Tensor):
        if self.embedding_space == "euclidean":
            x_euc = self.embedding(x)                  
            r_max = self.learned_radius()
            x_euc = self.clamp_euclidean_radius(x_euc, r_max) 
            x_lor0 = self.manifold.expmap0(x_euc)       

        elif self.embedding_space == "lorentz":
            x_lor0 = self.embedding(x)                 

        x_lor = self.encoder(x_lor0)
        pooled = self.pool_tokens(x_lor)
        logits = self.classifier(pooled)

        return logits
