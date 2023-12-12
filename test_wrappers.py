import torch
import egg.core as core
import torch.nn.functional as F
import random
from graph_b.dataset import FamilyGraphDataset
from agents import Sender, Receiver
from options import Options

def get_game(options: Options):
    sender = Sender(options.embedding_size, options.hidden_size)
    receiver = Receiver(options.embedding_size, options.hidden_size)

    if options.training_mode == 'gs':
        sender_wrapped = core.RnnSenderGS(sender, options.vocab_size, options.embedding_size, options.hidden_size, max_len=5, temperature=1.0, cell=options.sender_cell)
        receiver_wrapped = core.RnnReceiverGS(receiver, options.vocab_size, options.embedding_size, options.hidden_size, cell=options.sender_cell)
        game = core.SenderReceiverRnnGS(sender_wrapped, receiver_wrapped, loss)
    elif options.training_mode == 'rf':
        sender_wrapped = core.RnnSenderReinforce(sender, options.vocab_size, options.embedding_size, options.hidden_size, max_len=5)
        receiver_wrapped = core.RnnReceiverReinforce(receiver, options.vocab_size, options.embedding_size, options.hidden_size)
        game = core.SenderReceiverRnnReinforce(sender_wrapped, receiver_wrapped, loss)
    else:
        raise ValueError(f"Unknown training mode: {options.training_mode}")

    return game

def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()
        return nll, {"acc": acc}

# Test script
if __name__ == "__main__":
    options = Options()

    # Load the dataset and extract a graph
    dataset = FamilyGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/data', number_of_graphs=10, generations=3)
    data = dataset[0]

    # Initialize the game
    game = get_game(options)

    target_node_idx = random.randint(0, data.num_nodes - 1)

    # Sender produces a message
    sender_output = game.sender(data, target_node_idx)
    print("Sender's message:", sender_output)
    print("Sender's shape:", sender_output.shape)

    # Receiver tries to identify the target node
    receiver_output = game.receiver(sender_output, data)
    print("Receiver's output:", receiver_output)
    print("Receiver's shape:", receiver_output.shape)

    # Checking if the receiver's highest probability node is the target node
    reshaped_output = receiver_output.reshape(receiver_output.size(0), -1)
