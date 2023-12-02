import matplotlib.pyplot as plt
import torch

from working_agents import Sender, Receiver
from graph.build_dataset import CustomGraphDataset

def run_experiment(embedding_size, vocab_size, num_runs):
    correct_guesses = 0

    for _ in range(num_runs):
        # Initialize Sender and Receiver
        sender = Sender(embedding_size=embedding_size, vocab_size=vocab_size)
        receiver = Receiver(embedding_size=embedding_size, vocab_size=vocab_size)

        # Load your dataset and run your model here
        dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
        original_graph, masked_graph, target_node_idx = dataset[1]

        sender_output = sender(original_graph, target_node_idx)
        receiver_output = receiver(masked_graph, sender_output)

        predicted_node = torch.argmax(receiver_output, dim=0)

        # Check if the prediction is correct
        if predicted_node.item() == target_node_idx:
            correct_guesses += 1

    # Calculate the percentage of correct guesses
    accuracy = (correct_guesses / num_runs) * 100
    return accuracy

# Define the configurations to test
embedding_sizes = list(range(32, 257, 8))
vocab_sizes = list(range(100, 251, 5))

num_runs = 30

# Run experiments and collect results
embedding_results = [run_experiment(size, 100, num_runs) for size in embedding_sizes]
vocab_results = [run_experiment(128, size, num_runs) for size in vocab_sizes]

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(embedding_sizes, embedding_results, marker='o')
plt.title(f'Accuracy vs Embedding Size (n={num_runs})')
plt.xlabel('Embedding Size')
plt.ylabel('Accuracy (%)')

plt.subplot(1, 2, 2)
plt.plot(vocab_sizes, vocab_results, marker='o', color='orange')
plt.title(f'Accuracy vs Vocabulary Size (n={num_runs})')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
