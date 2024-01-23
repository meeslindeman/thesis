from graph.dataset import FamilyGraphDataset
import os

def generate_datasets(number_of_graphs, generations):
    # Define the root directory
    root = 'data'

    for g in generations:

        # Generate folder name based on parameters
        folder_name = f"gens={g}"
        directory = os.path.join(root, folder_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Instantiate the dataset
        dataset = FamilyGraphDataset(root=directory, number_of_graphs=number_of_graphs, generations=g)
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of nodes: {dataset[0].num_nodes}")