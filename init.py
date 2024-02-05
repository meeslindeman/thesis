from graph.gen_datasets import generate_datasets

# Set children in graph.build_graph.py:
generate_datasets(number_of_graphs=3200, generations=[(2,9), (3,27), (4,62)])