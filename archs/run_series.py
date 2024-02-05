import os
import logging
from graph.dataset import FamilyGraphDataset
from analysis.save import results_to_dataframe
from archs.game import get_game
from archs.dataloader import get_loaders
from archs.train import perform_training
from options import Options

def run_experiment(opts: Options, target_folder: str, save: bool = True):
    """
    Run an experiment with the given options.

    Args:
        opts (Options): The options for the experiment.
        target_folder (str): The folder to save the experiment results.
        save (bool, optional): Whether to save the experiment results. Defaults to True.

    Returns:
        pandas.DataFrame: The experiment results as a DataFrame.
    """

    logging.info(f"Running {str(opts)}")

    dataset = FamilyGraphDataset(root=f'data/gens={opts.generations}')

    train_loader, valid_loader = get_loaders(opts, dataset)
    game = get_game(opts, dataset.num_node_features)
    results, trainer = perform_training(opts, train_loader, valid_loader, game)

    return results_to_dataframe(results, dataset[0].num_nodes, opts, target_folder, save=save)

def run_series_experiments(opts: [Options], target_folder: str):
    """
    Run a series of experiments with different options and save the results to a target folder.

    Args:
        opts (list): A list of Options objects representing different experiment configurations.
        target_folder (str): The path to the folder where the experiment results will be saved.

    Returns:
        The experiment results and the target folder path.
    """

    results = []

    for opts in opts:
        result = run_experiment(opts, target_folder, False)

        filename = f"{str(opts)}.csv"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        result.to_csv(os.path.join(target_folder, filename), index=False)
        
    return results, target_folder