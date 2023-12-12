import torch
import random
from graph_b.dataset import FamilyGraphDataset
from agents import Sender, Receiver
from options import Options

# Load dataset 
dataset = FamilyGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/data', number_of_graphs=2, generations=3)
data = dataset[0]

options = Options()

sender = Sender(embedding_size=options.embedding_size, hidden_size=options.hidden_size) 
receiver = Receiver(embedding_size=options.embedding_size, hidden_size=options.hidden_size) 

target_node_idx = random.randint(0, data.num_nodes - 1)

# Sender produces a message
sender_output = sender(data, target_node_idx)
print("Sender's message:", sender_output)
print("Sender's shape: ", sender_output.shape)

# Receiver tries to identify the target node
receiver_output = receiver(sender_output, data)
print("Receiver's output:", receiver_output)
print("Receiver's shape: ", receiver_output.shape)

# Checking if the receiver's highest probability node is the target node
predicted_node = torch.argmax(receiver_output, dim=0)
print("Predicted target node:", predicted_node.item(), "\nActual target node:", target_node_idx)

