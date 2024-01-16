import logging
import coloredlogs
import pandas as pd
from options import Options
from run_series import run_experiment, run_series_experiments
from analysis.plot import plot_experiment, plot_all_experiments

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

# For one experiment
# results = run_experiment(Options, 'results')
# plot_experiment(Options, results, mode='both', save=False)

# For multiple experiments: adjust __str__ method in Options class !!!
options_list = [Options(agents='dual', hidden_size=20),
                Options(agents='transform', hidden_size=20),
                Options(agents='dual', hidden_size=40),
                Options(agents='transform', hidden_size=40),
                Options(agents='dual', hidden_size=60),
                Options(agents='transform', hidden_size=60),
                Options(agents='gat', hidden_size=20),
                Options(agents='gat', hidden_size=40),
                Options(agents='gat', hidden_size=60)]

results, target_folder = run_series_experiments(options_list, 'results')
target_folder = 'results'
plot_all_experiments(target_folder, mode='both', save=True)