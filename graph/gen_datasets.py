from graph.dataset import FamilyGraphDataset
import os

def generate_datasets(number_of_graphs, generations):
    # Define the root directory
    root = 'data'

    for g in generations:
        # Generate folder name based on parameters
        folder_name = f"gens={g[0]}"
        directory = os.path.join(root, folder_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Instantiate the dataset
        dataset = FamilyGraphDataset(root=directory, number_of_graphs=number_of_graphs, generations=g[0], padding_len=g[1])
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of nodes: {dataset[0].num_nodes}")
        print(f"Number of edges: {dataset[0].num_edges}")
        print(f"Number of features: {dataset[0].num_features}")