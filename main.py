import argparse
import logging
import coloredlogs
from options import Options
from archs.run_series import run_experiment, run_series_experiments
from analysis.plot import plot_experiment, plot_all_experiments
from analysis.timer import timer

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

@timer
def run_experiments(options_input):
    if isinstance(options_input, Options):
        results = run_experiment(options_input, 'results')
        plot_experiment(options_input, results, mode='both', save=False)
    elif isinstance(options_input, list):
        results, target_folder = run_series_experiments(options_input, 'results')
        plot_all_experiments(target_folder, mode='both', save=False)
    else:
        raise ValueError("Invalid input for options_input")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments based on the provided options.')
    parser.add_argument('--single', action='store_true', help='Run a single experiment')

    args = parser.parse_args()

    if args.single:
        # Run a single experiment: set options in command line
        single_options = Options()
        run_experiments(single_options)
    else:
        # Run multiple experiments: set __str__ in Options and labels in plot.py accordingly
        multiple_options = [
            Options(agents='dual', generations=2),
            Options(agents='dual', generations=3),
            Options(agents='dual', generations=4)
        ]
        run_experiments(multiple_options)
