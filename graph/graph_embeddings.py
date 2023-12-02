import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class Embeddings(torch.nn.Module):
    def __init__(self, embedding_size):
        super(Embeddings, self).__init__()
        self.num_node_features = 1 

        self.emb = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        self.embedding_size = embedding_size

    def forward(self, data: Data) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Generate embeddings for each node
        d = self.emb(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        return d  # Returns a tensor of shape [num_nodes, embedding_size]