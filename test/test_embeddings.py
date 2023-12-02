from graph.build_dataset import CustomGraphDataset
from graph.graph_embeddings import Embeddings
from options import Options

# Load the dataset
dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
original_graph, masked_graph, target_node_idx = dataset[0]  # Get the first data point from the dataset

options = Options()

# Initialize the Embeddings class
embedding_layer = Embeddings(embedding_size=options.embedding_size)

# Choose either the original graph or the masked graph
graph_to_embed = original_graph  # or masked_graph
print(graph_to_embed)

# Generate embeddings for the chosen graph
embeddings = embedding_layer(graph_to_embed)

# Print the embeddings
print("Generated Embeddings:", embeddings)
print("Shape of Embeddings:", embeddings.shape)