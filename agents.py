import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class Sender(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Sender, self).__init__()
        self.num_node_features = 2

        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=1, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=1, heads=1, concat=True)
        self.fc = nn.Linear(embedding_size, hidden_size)

    def forward(self, data: Data, target_node_idx) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        x = F.relu(x)

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        x = F.relu(x)
        
        target_embedding = x[target_node_idx]                               # embedding size                             
        target_embedding = target_embedding.unsqueeze(0)                    # 1 x embedding_size

        # target_embedding = target_embedding.repeat(x.size(0), 1)

        # combined_representation = torch.cat([x, target_embedding], dim=1)

        output = self.fc(target_embedding)                                  # 1 x hidden size

        return output

class Receiver(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        self.num_node_features = 2

        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=1, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=1, heads=1, concat=True)
        self.fc1 = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, data, _aux_input=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding_size
        x = F.relu(x)

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding_size
        x = F.relu(x)   

        message_embedding = self.fc1(message)                               # 1 x embeddin_size

        dot_products = torch.matmul(x, message_embedding.t())               # nodes x 1

        probabilities = F.softmax(dot_products, dim=0)                      # nodes x 1

        return probabilities