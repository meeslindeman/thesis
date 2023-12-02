import os
import torch
import random
from torch_geometric.data import Dataset
from graph.build_relationship import RelationshipBuilder
from graph.build_graph import GraphBuilder

class CustomGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['data.pt']

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def process(self):
        # Check if the dataset is already processed
        if not os.path.isfile(self.processed_paths[0]):
            self.data = []

            for file_path in os.listdir(self.root):
                if file_path.endswith('.txt'):
                    full_path = os.path.join(self.root, file_path)
                    relationship_builder = RelationshipBuilder(full_path)
                    graph_builder = GraphBuilder(relationship_builder)
                    graph_data = graph_builder.build_graph()
                    self.data.append(graph_data)

            # Save the processed data
            torch.save(self.data, self.processed_paths[0])
        else:
            # Load the processed data
            self.data = torch.load(self.processed_paths[0])
    
    def mask_node(self, data, target_node_idx):
        masked_data = data.clone()  # Create a copy of the data object
        # Mask the node features of the target node (you can choose how to mask)
        masked_data.x[target_node_idx] = torch.zeros_like(data.x[target_node_idx])
        return masked_data

    def __getitem__(self, idx):
        data = self.data[idx]
        target_node_idx = random.randint(0, data.num_nodes - 1)  # Select a random node as target

        # Mask the target node for the receiver's version of the graph
        masked_data = self.mask_node(data, target_node_idx)

        return data, masked_data, target_node_idx

# dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
# graph, masked_graph, target_node = dataset[0] 
# print(graph.x)
# print("======")
# print(masked_graph.x)
# print("======")
# print(target_node)

# print()
# print(f'Dataset: {dataset}:')
# print('======================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')

# data = dataset[1]  # Get the first graph object.

# print()
# print(data)
# print('===========================================================================================================')

# # Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')