from dataclasses import dataclass

@dataclass
class Options:
    # Agents
    embedding_size: int = 40 # Default: 50
    heads: int = 2 # Default: 1
    hidden_size: int = 20 # Default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4 # Default: 1
    gs_tau: float = 1.0 # Default: 1.0

    # Training
    n_epochs: int = 100
    agents: str = 'dual' # 'dual', 'transform', 'gat', 'rel
    vocab_size: int = 100 # Default: 100
    batch_size: int = 1 # Always set to 1
    accuracy: float = 1.0 # Set desired stopping accuracy

    # Dataset
    # Delete the processed folder when changing this
    number_of_graphs: int = 100
    generations: int = 2 

    # Configuration for callbacks
    callbacks_config = {
        'early_stopper': False,
        'message_entropy': False,
        'print_validation': False,
        'topographic_similarity': False,
        'disent': False
    }

    # Set this depending on the amount of options changed
    def __str__(self):
        return f"{self.agents}_{self.hidden_size}"