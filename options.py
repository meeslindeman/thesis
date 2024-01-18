from dataclasses import dataclass

@dataclass
class Options:
    # Agents
    embedding_size: int = 40 # default: 50
    heads: int = 2 # default: 1
    hidden_size: int = 20 # default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4 # default: 1
    gs_tau: float = 1.0 # default: 1.0

    # Training
    n_epochs: int = 2
    agents: str = 'rel' # 'dual', 'transform', 'gat', 'rel
    vocab_size: int = 100 # default: 100
    batch_size: int = 1 # always set to 1
    accuracy: float = 0.5 # set desired stopping accuracy

    # Dataset
    generations: int = 2

    # Configuration for callbacks -> see logger.py for additional options
    # Only topographic_similarity is implemented
    callbacks_config = {
        'early_stopper': False,
        'message_entropy': False,
        'print_validation': False,
        'topographic_similarity': False,
        'disent': False
    }

    # Set this according to parameters in main.py
    def __str__(self):
        return f"{self.vocab_size}_{self.max_len}"
