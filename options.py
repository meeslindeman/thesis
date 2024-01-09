from dataclasses import dataclass

@dataclass
class Options:
    # Agents
    embedding_size: int = 40 # Default: 50
    heads: int = 2 # Default: 1
    hidden_size: int = 20 # Default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4 # Default: 1
    gs_tau: int = 1.0 # Default: 1.0

    # Training
    n_epochs: int = 5
    agents: str = 'dual' # 'dual', 'transform', 'gat'
    vocab_size: int = 100 # Default: 100
    batch_size: int = 1 