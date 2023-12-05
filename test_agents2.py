import torch
import egg.core as core
import torch.nn.functional as F
from graph.build_dataset import CustomGraphDataset
from agents import Sender, Receiver
from options import Options

def get_game(options: Options):
    sender = Sender(options.embedding_size, options.hidden_size, options.vocab_size, temp=1.0)
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
    dataset = CustomGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families')
    original_graph, masked_graph, target_node_idx = dataset[1]

    # Initialize the game
    game = get_game(options)

    # Sender produces a message
    sender_output = game.sender(original_graph, target_node_idx)
    print("Sender's message:", sender_output)
    print("Sender's shape:", sender_output.shape)

    # Receiver tries to identify the target node
    receiver_output = game.receiver(masked_graph, sender_output)
    print("Receiver's output:", receiver_output)

    # Checking if the receiver's highest probability node is the target node
    predicted_node = torch.argmax(receiver_output, dim=1)
    print("Predicted target node:", predicted_node.item(), "\nActual target node:", target_node_idx)

