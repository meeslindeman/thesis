import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class Sender(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, temp):
        super(Sender, self).__init__()
        self.temp = temp

        self.num_node_features = 1
        self.emb = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        self.fc = nn.Linear(embedding_size*4, hidden_size)

    def forward(self, data: Data, target_node_idx) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        graph_embedding = self.emb(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        target_embedding = graph_embedding[target_node_idx]
        
        target_embedding = target_embedding.unsqueeze(0)

        target_embedding = target_embedding.repeat(graph_embedding.size(0), 1)

        combined_representation = torch.cat([graph_embedding, target_embedding], dim=1)
        output = self.fc(combined_representation)

        return output

class Receiver(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        
        self.num_node_features = 1
        self.emb = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        self.fc = nn.Linear(hidden_size, embedding_size*2)

    def forward(self, data: Data, message, _aux_input=None) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        graph_embedding = self.emb(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        message_embedding = self.fc(message)

        message_embedding_t = message_embedding.t()

        dot_products = torch.matmul(graph_embedding, message_embedding_t)
        probabilities = F.softmax(dot_products, dim=1)

        return probabilities