import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class Sender(nn.Module):
    """
    Sender module that takes input data and target node index, and outputs a hidden representation of the target node.
    """
    def __init__(self, embedding_size, hidden_size, temperature):
        super(Sender, self).__init__()
        self.num_node_features = 2
        self.heads = 2
        self.temp = temperature

        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=self.heads, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=1, heads=self.heads, concat=True)
        self.fc = nn.Linear((embedding_size * self.heads), hidden_size)

    def forward(self, data: Data, _aux_input=None) -> torch.Tensor:
        """
        Forward pass of the sender module.

        Args:
            data (Data): Input data containing node features, edge indices, and edge attributes.
            target_node_idx (int): Index of the target node.

        Returns:
            torch.Tensor: Hidden representation of the target node.
        """
        x, edge_index, edge_attr, labels = data.x, data.edge_index, data.edge_attr, data.labels

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        h = F.relu(h)

        h = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        h = F.relu(h)

        target_node_idx = torch.nonzero(labels, as_tuple=True)[0].item()
        
        target_embedding = h[target_node_idx]                               # embedding size                
        target_embedding = target_embedding.unsqueeze(0)                    # 1 x embedding_size

        output = self.fc(target_embedding)                                  # 1 x hidden size

        return output

class Receiver(nn.Module):
    """
    Receiver module that performs message passing and computes probabilities based on dot products.
    """

    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        self.num_node_features = 2
        self.heads = 2

        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=self.heads, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=1, heads=self.heads, concat=True)
        self.fc = nn.Linear(hidden_size, (embedding_size * self.heads))

    def forward(self, message, data, _aux_input=None):
        """
        Forward pass of the Receiver module.

        Args:
            message (torch.Tensor): The input message.
            data (torch_geometric.data.Data): The input data.
            _aux_input (None, optional): Auxiliary input (not used).

        Returns:
            torch.Tensor: The computed probabilities.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        h = F.relu(h)

        h = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)     # nodes x embedding size
        h = F.relu(h)   

        message = torch.cat([message, message], dim=0)

        message_embedding = self.fc(message)                                # 1 x embedding size

        dot_products = torch.matmul(h, message_embedding.t())               # nodes x 1

        probabilities = F.softmax(dot_products, dim=0)                      # nodes x 1

        return probabilities