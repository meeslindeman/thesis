import egg.core as core
import torch 
from options import Options
from analysis.logger import ResultsCollector, get_callbacks

def perform_training(opts: Options, train_loader, val_loader, game):
    results = []

    core.init(params=['--random_seed=42',
                      '--lr=1e-3',
                      f'--batch_size={opts.batch_size}',
                      f'--n_epochs={opts.n_epochs}',
                      f'--vocab_size={opts.vocab_size}',
                      '--update_freq=10'])

    callbacks_list = get_callbacks(opts)
    callbacks = [ResultsCollector(results, print_to_console=True, callbacks_list=callbacks_list)] 

    trainer = core.Trainer(
        game=game, 
        optimizer=torch.optim.Adam(game.parameters()), 
        train_data=train_loader,
        validation_data=val_loader, 
        callbacks=callbacks
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()
    
    return '\n'.join(results), trainer
