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
        self.fc = nn.Linear(embedding_size * 4, hidden_size)

    def forward(self, data: Data, target_node_idx) -> torch.Tensor:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First convolutional layer with ReLU activation
        x = self.conv1(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        # Second convolutional layer
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        
        # Target node embedding
        target_embedding = x[target_node_idx]
        
        # Adjust the target embedding to match the graph embedding shape
        target_embedding = target_embedding.unsqueeze(0)
        target_embedding = target_embedding.repeat(x.size(0), 1)

        # Combine graph and target embeddings
        combined_representation = torch.cat([x, target_embedding], dim=1)

        # Output layer
        output = self.fc(combined_representation)

        return output

class Receiver(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()

        self.num_node_features = 1
        # Convolutional layers
        self.conv1 = GATv2Conv(self.num_node_features, embedding_size, edge_dim=1, heads=2, concat=True)
        self.conv2 = GATv2Conv(embedding_size * 2, embedding_size, edge_dim=1, heads=2, concat=True)
        # To process the message
        self.fc1 = nn.Linear(hidden_size, embedding_size * 2)

    def forward(self, message, data, _aux_input=None):
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First convolutional layer with ReLU activation
        x = self.conv1(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)

        # Second convolutional layer
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)          

        # Adjust the message processing to handle the new message format
        message_embedding = self.fc1(message)                                       

        # Comparing message with each node's embedding
        dot_products = torch.matmul(x, message_embedding.t())
        probabilities = F.softmax(dot_products, dim=1)  # Convert to probabilities

        return probabilities