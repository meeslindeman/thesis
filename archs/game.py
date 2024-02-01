import egg.core as core
import torch.nn.functional as F
from archs.agents import SenderDual, ReceiverDual, SenderGAT, ReceiverGAT, SenderTransform, ReceiverTransform, SenderRel, ReceiverRel
from options import Options

def get_game(opts: Options, num_node_features: int):
    if opts.agents == "dual":
        sender = SenderDual(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, temperature=opts.gs_tau) 
        receiver = ReceiverDual(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size)
    elif opts.agents == "transform":
        sender = SenderTransform(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, temperature=opts.gs_tau) 
        receiver = ReceiverTransform(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size) 
    elif opts.agents == "gat":
        sender = SenderGAT(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, temperature=opts.gs_tau) 
        receiver = ReceiverGAT(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size) 
    else:
        print("Invalid agent type")

    sender_gs = core.RnnSenderGS(sender, 
                                 opts.vocab_size, 
                                 opts.embedding_size, 
                                 opts.hidden_size, 
                                 max_len=opts.max_len, 
                                 temperature=opts.gs_tau, 
                                 cell=opts.sender_cell)
    
    receiver_gs = core.RnnReceiverGS(receiver, 
                                     opts.vocab_size, 
                                     opts.embedding_size, 
                                     opts.hidden_size, 
                                     cell=opts.sender_cell)

    def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        """
        NLL loss - differentiable and can be used with both GS and Reinforce
        """
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()
        return nll, {"acc": acc}

    game = core.SenderReceiverRnnGS(sender_gs, receiver_gs, loss_nll)

    return game

