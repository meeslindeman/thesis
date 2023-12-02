import torch
from graph.build_relationship import RelationshipBuilder
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, relationship_builder):
        self.relationship_builder = relationship_builder
        self.node_to_idx = {node[0]: idx for idx, node in enumerate(self.relationship_builder.nodes)}

    def build_graph(self):
        # Prepare node features (gender in this case)
        node_features = [[1 if node[1] == 'M' else 0] for node in self.relationship_builder.nodes]
        relationship_encoding = self.relationship_builder.get_relationship_encoding()

        # Prepare edges and their attributes
        edge_list = []
        for edge in self.relationship_builder.edges:
            source, target, relationship = edge
            edge_list.append([self.node_to_idx[source], self.node_to_idx[target]])

        edge_attributes = [relationship_encoding[edge[2]] for edge in self.relationship_builder.edges]

        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float) # Modify this as per your encoding scheme

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr) 

# rb = RelationshipBuilder('/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families/greyjoy.txt')
# gb = GraphBuilder(rb)
# d = gb.build_graph()
# print("Data: ", d)
# print("Nodes: ", d.x)
# print("Edge index: ", d.edge_index)
# print("Edge attr: ", d.edge_attr)

