import torch
from graph.build_dataset import CustomGraphDataset
from agents import Sender, Receiver
# from agents_basic import Sender, Receiver
from options import Options

# Load the dataset
dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
original_graph, masked_graph, target_node_idx = dataset[1]

# Initialize Sender and Receiver

options = Options()

sender = Sender(embedding_size=options.embedding_size, hidden_size=options.hidden_size) 
receiver = Receiver(embedding_size=options.embedding_size, hidden_size=options.hidden_size) 

# sender = Sender(embedding_size=options.embedding_size, message=options.hidden_size) 
# receiver = Receiver(embedding_size=options.embedding_size, message=options.hidden_size) 

# Sender produces a message
sender_output = sender(original_graph, target_node_idx)
print("Sender's message:", sender_output)
print("Sender's shape: ", sender_output.shape)

# Receiver tries to identify the target node
receiver_output = receiver(sender_output, masked_graph)
print("Receiver's output:", receiver_output)
print("Receiver's shape: ", receiver_output.shape)

# Checking if the receiver's highest probability node is the target node
predicted_node = torch.argmax(receiver_output, dim=0)
# print("Predicted target node:", predicted_node.item(), "\nActual target node:", target_node_idx)

# Iterate over each predicted node and compare with the actual target node
for i, predicted_node in enumerate(predicted_node):
    print(f"Sample {i}: Predicted target node: {predicted_node.item()}, Actual target node: {target_node_idx}")


