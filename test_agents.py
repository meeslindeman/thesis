import torch
from graph.build_dataset import CustomGraphDataset
from agents_basic import Sender, Receiver
from options import Options

# Load the dataset
dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
original_graph, masked_graph, target_node_idx = dataset[1]

# Initialize Sender and Receiver

options = Options()

sender = Sender(embedding_size=options.embedding_size, message=options.vocab_size)
receiver = Receiver(embedding_size=options.embedding_size, message=options.vocab_size)

# Sender produces a message
sender_output = sender(original_graph, target_node_idx)
print("Sender's message:", sender_output)
print("Sender's shape: ", sender_output.shape)

# Receiver tries to identify the target node
receiver_output = receiver(masked_graph, sender_output)
print("Receiver's output:", receiver_output)
print("Receiver's shape: ", receiver_output.shape)

# Checking if the receiver's highest probability node is the target node
predicted_node = torch.argmax(receiver_output, dim=0)
print("Predicted target node:", predicted_node.item(), "\nActual target node:", target_node_idx)

