import egg.core as core
from egg.core import ConsoleLogger
from options import Options

def perform_training(options: Options, train_loader, valid_loader, game):
    # Define the training process here
    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=[ConsoleLogger(as_json=True, print_train_loss=True)]
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()

    return trainer
