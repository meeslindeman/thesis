import egg.core as core
import torch 
from options import Options
from analysis.logger import ResultsCollector, get_callbacks

def perform_training(opts: Options, train_loader, val_loader, game):
    results = []

    core.init(params=['--random_seed=7',
                      '--lr=1e-2',
                      '--optimizer=adam'])

    callbacks_list = get_callbacks(opts)
    callbacks = [ResultsCollector(results, print_to_console=True, callbacks_list=callbacks_list)] 

    optimizer = torch.optim.Adam(game.parameters())

    trainer = core.Trainer(
        game=game, 
        optimizer=optimizer, 
        train_data=train_loader,
        validation_data=val_loader, 
        callbacks=callbacks
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()
    
    return '\n'.join(results), trainer
