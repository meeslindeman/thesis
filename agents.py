import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.graph_embeddings import Embeddings

class Sender(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, temp):
        super(Sender, self).__init__()
        self.temp = temp

        self.embedding = Embeddings(embedding_size)
        self.fc = nn.Linear((embedding_size * 4), hidden_size)

    def forward(self, data, target_node_idx):
        embeddings = self.embedding(data)
        target_embedding = embeddings[target_node_idx]
        
        # Reshape target_embedding to match the dimensions of embeddings
        target_embedding = target_embedding.unsqueeze(0)  # Reshape to [1, embedding_size]

        # Ensure target_embedding is repeated for each element in the batch
        target_embedding = target_embedding.repeat(embeddings.size(0), 1)

        # Combine and transform the representations
        combined_representation = torch.cat([embeddings, target_embedding], dim=1)
        output = self.fc(combined_representation)
        return output

class Receiver(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Receiver, self).__init__()

        self.embedding = Embeddings(embedding_size)
        self.fc = nn.Linear(hidden_size, (embedding_size * 2))

    def forward(self, data, message, _aux_input=None):
        embeddings = self.embedding(data) 
        message_embedding = self.fc(message)

        message_embedding_t = message_embedding.t()

        # Compute dot products
        dot_products = torch.matmul(embeddings, message_embedding_t)

        # Convert dot products to probabilities or scores
        # For example, using softmax if the task is to choose one node out of many
        probabilities = F.softmax(dot_products, dim=1)

        return probabilities