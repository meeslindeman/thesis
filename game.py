import egg.core as core
from agents import Sender, Receiver
from options import Options
import torch.nn.functional as F

def get_game(options: Options):
    sender = Sender(
        embedding_size=options.embedding_size, 
        vocab_size=options.vocab_size)
    
    receiver = Receiver(
        embedding_size=options.embedding_size, 
        vocab_size=options.vocab_size)

    # Define the loss function here
    def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()
        return nll, {"acc": acc}

    if options.training_mode == 'rf':
        sender_wrapped = core.RnnSenderReinforce(sender, options.vocab_size, options.embedding_size, options.hidden_size, options.max_len)
        receiver_wrapped = core.RnnReceiverReinforce(sender, options.vocab_size, options.embedding_size, options.hidden_size, options.max_len)
        game = core.SenderReceiverRnnReinforce(sender_wrapped, receiver_wrapped, loss)
    elif options.training_mode == 'gs':
        sender_wrapped = core.RnnSenderGS(sender, options.vocab_size, options.embedding_size, options.hidden_size, options.max_len, temperature=1.0, cell=options.sender_cell)
        receiver_wrapped = core.RnnReceiverGS(receiver, options.vocab_size, options.embedding_size, options.hidden_size, options.max_len, cell=options.sender_cell)
        game = core.SenderReceiverRnnGS(sender_wrapped, receiver_wrapped, loss)
    else:
        raise ValueError(f"Unknown training mode: {options.training_mode}")

    return game
