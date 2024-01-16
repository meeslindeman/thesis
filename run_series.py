import os
import logging
from graph.dataset import FamilyGraphDataset
from save import results_to_dataframe
from game import get_game
from dataloader import get_loaders
from train import perform_training
from datetime import datetime
from options import Options

def run_experiment(opts: Options, target_folder: str, save: bool = True):
    dataset = FamilyGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/data', number_of_graphs=opts.number_of_graphs, generations=opts.generations)
    logging.info(f"Running {str(opts)}")

    train_loader, valid_loader = get_loaders(dataset)
    game = get_game(opts)
    results, trainer = perform_training(opts, train_loader, valid_loader, game)

    return results_to_dataframe(results, dataset[0].num_nodes, opts, target_folder, save=save)

def run_series_experiments(opts: [Options], target_folder: str):
    results = []

    for opts in opts:
        # Run the experiment with the current options
        result = run_experiment(opts, target_folder, False)

        # Generate a unique filename
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{str(opts)}.csv"

        # Save the DataFrame to the target folder with the unique filename
        result.to_csv(os.path.join(target_folder, filename), index=False)

        # Optionally, print or log that this experiment is done
        # logging.info(f"Experiment with options {str(opts)} completed.")
        
    return results, target_folder