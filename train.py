import egg.core as core
from egg.core import ConsoleLogger
from options import Options
from analysis.timer import timer
from analysis.logger import ResultsCollector

def perform_training(opts: Options, train_loader, val_loader, game):
    results = []

    opts = core.init(params=['--random_seed=42', 
                            '--lr=1e-3',  
                            f'--batch_size={opts.batch_size}',
                            f'--n_epochs={opts.n_epochs}',
                            f'--vocab_size={opts.vocab_size}',
                            '--optimizer=adam',
                            '--update_freq=10'])

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game, 
        optimizer=optimizer, 
        train_data=train_loader,
        validation_data=val_loader, 
        # callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)]
        callbacks=[ResultsCollector(results, print_to_console=True)]
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()
    
    return '\n'.join(results), trainer
