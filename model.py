import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from graph.build_dataset import CustomGraphDataset

dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
original_graph, masked_graph, target_node_idx = dataset[1]

data = original_graph

class GAT(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_features, embedding_size)
        self.conv2 = GATv2Conv(embedding_size, 32)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class Sender(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Sender, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_features, embedding_size)
        self.conv2 = GATv2Conv(embedding_size, 32)

    def forward(self, data, target_node_idx):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class Receiver(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Receiver, self).__init__()
        self.embedding = Embeddings(embedding_size)
        output = embedding_size * 2
        self.fc = nn.Linear(vocab_size, output)

    def forward(self, data, message, _aux_input=None):
        embeddings = self.embedding(data) # [num_nodes, 128]
        message_embedding = self.fc(message) # [128]
        dot_products = torch.matmul(embeddings, message_embedding.t())
        return dot_products