import egg.core
from dataclasses import dataclass

@dataclass
class Options:
    # Agents
    embedding_size: int = 32
    hidden_size: int = 16
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    max_len: int = 4
    temp: int = 1.0

    # Training
    n_epochs: int = 10
    vocab_size: int = 20
    batch_size: int = 2
    training_mode: str = 'gs'  # 'rf' for Reinforce or 'gs' for Gumbel-Softmax


    def init_egg_params(self):
        egg_params = egg.core.init(params=[
            '--random_seed=42',
            '--lr=1e-3',
            '--optimizer=adam',
            f'--batch_size={self.batch_size}',
            f'--n_epochs={self.n_epochs}',
            f'--vocab_size={self.vocab_size}'
        ])