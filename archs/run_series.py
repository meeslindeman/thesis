import os
import logging
from graph.dataset import FamilyGraphDataset
from analysis.save import results_to_dataframe
from archs.game import get_game
from archs.dataloader import get_loaders
from archs.train import perform_training
from options import Options

def run_experiment(opts: Options, target_folder: str, save: bool = True):
    logging.info(f"Running {str(opts)}")

    dataset = FamilyGraphDataset(root=f'data/gens={opts.generations}')

    train_loader, valid_loader = get_loaders(opts, dataset)
    game = get_game(opts, dataset.num_node_features)
    results, trainer = perform_training(opts, train_loader, valid_loader, game)

    return results_to_dataframe(results, dataset[0].num_nodes, opts, target_folder, save=save)

def run_series_experiments(opts: [Options], target_folder: str):
    results = []

    for opts in opts:
        result = run_experiment(opts, target_folder, False)

        filename = f"{str(opts)}.csv"
        result.to_csv(os.path.join(target_folder, filename), index=False)

        # logging.info(f"Experiment with options {str(opts)} completed.")
        
    return results, target_folder