from graph.gen_datasets import generate_datasets
from analysis.plot import plot_experiment, plot_all_experiments
from analysis.final_metrics import get_final_accuracies

generate_datasets(5000, [2, 3, 4])
# plot_all_experiments('results', mode='both', save=False)
# get_final_accuracies('results', save=True)