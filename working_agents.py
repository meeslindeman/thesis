import torch
import torch.nn as nn
from graph.graph_embeddings import Embeddings
from options import Options

class Sender(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Sender, self).__init__()
        output = embedding_size * 2
        self.embedding = Embeddings(embedding_size)
        self.fc = nn.Linear(output, vocab_size)

    def forward(self, data, target_node_idx):
        #print("Sender received data:", data)
        embeddings = self.embedding(data)
        target_embedding = embeddings[target_node_idx]
        logits = self.fc(target_embedding)
        return logits

class Receiver(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Receiver, self).__init__()
        self.embedding = Embeddings(embedding_size)
        output = embedding_size * 2
        self.fc = nn.Linear(vocab_size, output)

    def forward(self, data, message, _aux_input=None):
        #print("Receiver received data:", data)
        embeddings = self.embedding(data) # [num_nodes, 128]
        message_embedding = self.fc(message) # [128]
        dot_products = torch.matmul(embeddings, message_embedding.t())
        return dot_products