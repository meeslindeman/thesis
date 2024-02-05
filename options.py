from dataclasses import dataclass

@dataclass
class Options:
    """
    Class representing the options for the experiment.
    """
    # Agents
    embedding_size: int = 10 # default: 10
    heads: int = 4 # default: 4
    hidden_size: int = 20 # default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4 # default: 4
    gs_tau: float = 1.0 # default: 1.0

    # Training
    n_epochs: int = 2
    agents: str = 'dual' # 'dual', 'transform', 'gat'
    vocab_size: int = 100 # default: 100
    batch_size: int = 32
    accuracy: float = 1.0 # set desired stopping accuracy
    topsim: bool = True
    random_seed: int = 42

    # Generations to be used (dataset graph size)
    generations: int = 3

    # Set this according to parameters in main.py
    def __str__(self):
        return f"{self.agents}_{self.generations}"
