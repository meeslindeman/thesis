import os
import torch
import random
from torch_geometric.data import Dataset
from graph.build import create_family_tree, create_data_object
from graph.sequence import process_graph_to_sequence

class FamilyGraphDataset(Dataset):
    """
    Dataset class for generating family graph data.

    Args:
        root (str): Root directory path.
        number_of_graphs (int): Number of graphs to generate.
        generations (int): Number of generations in each family tree.

    Returns:
        Data object containing the family graph data.
    """
    def __init__(self, root, number_of_graphs=3200, generations=1, padding_len=9, transform=None, pre_transform=None):
        self.number_of_graphs = number_of_graphs
        self.generations = generations
        self.padding_len = padding_len
        super(FamilyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data = None
        self.process()

    @property
    def processed_file_names(self):
        return ['family_graphs.pt']

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    
    def generate_labels(self, num_nodes):
        target_node_idx = random.randint(0, num_nodes - 1)
        return target_node_idx
    
    def generate_root(self, num_nodes, target_node_idx):
        node_indices = list(range(num_nodes))
        node_indices.remove(target_node_idx)
        root_idx = random.choice(node_indices)
        return root_idx
    
    def process(self):
        if not os.path.isfile(self.processed_paths[0]):
            self.data = []
            for _ in range(self.number_of_graphs):
                family_tree = create_family_tree(self.generations)
                graph_data = create_data_object(family_tree)

                # Generate random labels for each node
                target_node_idx = self.generate_labels(graph_data.num_nodes)
                graph_data.target_node_idx = target_node_idx

                root_idx = self.generate_root(graph_data.num_nodes, target_node_idx)
                graph_data.root_idx = root_idx

                # Process graph to sequence
                sequence = process_graph_to_sequence(graph_data, self.padding_len)
                graph_data.sequence = sequence

                self.data.append(graph_data)

            torch.save(self.data, self.processed_paths[0])
        else:
            self.data = torch.load(self.processed_paths[0])