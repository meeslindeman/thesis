import egg.core
from dataclasses import dataclass

@dataclass
class Options:
    embedding_size: int = 64
    vocab_size: int = 50
    n_epochs: int = 10
    batch_size: int = 32
    training_mode: str = 'gs'  # 'rf' for Reinforce or 'gs' for Gumbel-Softmax
    hidden_size: int = 64
    max_len: int = 5
    output_size: int = 8
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'

    def init_egg_params(self):
        egg_params = egg.core.init(params=[
            '--random_seed=42',
            '--lr=1e-3',
            f'--batch_size={self.batch_size}',
            f'--n_epochs={self.n_epochs}',
            f'--vocab_size={self.vocab_size}'
        ])