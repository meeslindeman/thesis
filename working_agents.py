import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class Sender(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Sender, self).__init__()

        self.num_node_features = 1
        # First convolutional layer
        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        # Additional convolutional layers
        self.conv2 = GATv2Conv(embedding_size * 2, embedding_size, edge_dim=1, heads=2, concat=True)
        # Final layer to match the hidden size of the RNN cell
        self.fc = nn.Linear(embedding_size * 2, hidden_size)

    def forward(self, data: Data, target_node_idx) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First convolutional layer with ReLU activation
        x = self.conv1(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        # Second convolutional layer
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Target node embedding
        target_embedding = x[target_node_idx]
        hidden_state = self.fc(target_embedding)
        return hidden_state

class Receiver(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        self.num_node_features = 1
        # Convolutional layers
        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        self.conv2 = GATv2Conv(embedding_size * 2, embedding_size, edge_dim=1, heads=2, concat=True)
        # To process the message
        self.fc1 = nn.Linear(hidden_size, embedding_size * 2)

    def forward(self, data: Data, message) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Generate embeddings for the graph
        x = self.conv1(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Process the message
        message_embedding = self.fc1(message)

        # Comparing message with each node's embedding
        dot_products = torch.matmul(x, message_embedding.t())
        probabilities = F.softmax(dot_products, dim=0)  # Convert to probabilities
        return probabilities