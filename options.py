from dataclasses import dataclass

@dataclass
class Options:
    # Agents
    embedding_size: int = 10 # default: 10
    heads: int = 4 # default: 4
    hidden_size: int = 20 # default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4 # default: 4
    gs_tau: float = 1.0 # default: 1.0

    # Training
    n_epochs: int = 40
    agents: str = 'dual' # 'dual', 'transform', 'gat', 'rel
    vocab_size: int = 100 # default: 100
    batch_size: int = 50
    accuracy: float = 1.0 # set desired stopping accuracy

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
        return f"{self.agents}_{self.generations}"
