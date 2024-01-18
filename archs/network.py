import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, embedding_size, edge_dim=0, heads=heads, concat=True)
        self.conv2 = GATv2Conv(-1, embedding_size, edge_dim=0, heads=heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     
        h = F.relu(h)

        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr) 
        h = F.relu(h)    

        return h

class Transform(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads):
        super().__init__()
        self.conv1 = TransformerConv(num_node_features, embedding_size, edge_dim=1, heads=heads, concat=True)
        self.conv2 = TransformerConv(-1, embedding_size, edge_dim=1, heads=heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_attr = edge_attr.view(-1, 1)

        h = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)     
        h = F.relu(h)

        h = F.dropout(h, training=self.training)

        h = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)     
        h = F.relu(h)

        return h