import argparse
import logging
import coloredlogs
from options import Options
from archs.run_series import run_experiment, run_series_experiments
from analysis.plot import plot_experiment, plot_all_experiments

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

def run_experiments(options_input):
    if isinstance(options_input, Options):
        # Single experiment
        results = run_experiment(options_input, 'results')
        plot_experiment(options_input, results, mode='both', save=False)
    elif isinstance(options_input, list):
        # Multiple experiments
        results, target_folder = run_series_experiments(options_input, 'results')
        plot_all_experiments(target_folder, mode='both', save=False)
    else:
        raise ValueError("Invalid input for options_input")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments based on the provided options.')
    parser.add_argument('--single', action='store_true', help='Run a single experiment')
    parser.add_argument('--vocab_size', type=int, default=10, help='Vocabulary size for the experiment')
    parser.add_argument('--n_epochs', type=int, default=50, help='Amount of epochs for the experiment')

    args = parser.parse_args()

    if args.single:
        # Run a single experiment
        single_options = Options(vocab_size=args.vocab_size, n_epochs=args.n_epochs)
        run_experiments(single_options)
    else:
        # Run multiple experiments
        multiple_options = [
            Options(vocab_size=10, max_len=1),
            Options(vocab_size=20, max_len=3)
        ]
        run_experiments(multiple_options)
