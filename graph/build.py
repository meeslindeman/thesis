import torch
from torch_geometric.data import Data
import random
import numpy as np
import names

def create_random_name(gender):
    if gender == 0:  # Male
        return names.get_first_name(gender='male')
    else:  # Female
        return names.get_first_name(gender='female')

def create_graph(n):
    x = torch.tensor([[random.randint(0, 1), random.randint(18, 100)] for _ in range(n)], dtype=torch.float)
    edge_index = []
    edge_type = []
    married = np.zeros(n, dtype=bool)
    parent_of = {i: [] for i in range(n)}  # Keep track of who is parent of whom
    child_of = {i: [] for i in range(n)}  # Keep track of parents for each individual

    for i in range(n):
        for j in range(i + 1, n):
            age_i, age_j = x[i, 1].item(), x[j, 1].item()
            age_diff = abs(age_i - age_j)

            # Rule out marriage if they share a parent or if the age difference is less than 20
            if not set(parent_of[i]).isdisjoint(parent_of[j]) or age_diff < 20:
                continue

            # Assign married status if both are not already married, and the age difference is reasonable
            if age_diff >= 20 and not married[i] and not married[j]:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_type.extend([1, 1])
                married[i] = married[j] = True

            # Check if they are part of the same immediate family
            if set(parent_of[i]).intersection(parent_of[j]) or i in parent_of[j] or j in parent_of[i]:
                continue  # Skip if they are in the same immediate family

            # Skip if they share a parent (to avoid incest)
            if not set(parent_of[i]).isdisjoint(parent_of[j]):
                continue

            # Determine parent-child relationships based on age difference
            if age_diff >= 20 and len(child_of[i if age_i < age_j else j]) < 2:
                parent, child = (i, j) if age_i > age_j else (j, i)

                # Ensure no one is a child of their child and they have less than two parents
                if parent not in parent_of[child]:
                    edge_index.append([parent, child])
                    edge_type.append(0)
                    parent_of[parent].append(child)
                    child_of[child].append(parent)

            # Assign married status if both are not already married, and not in the same immediate family
            elif not married[i] and not married[j]:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_type.extend([1, 1])
                married[i] = married[j] = True

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_type, dtype=torch.long)
    
    # Generate names with gender consideration
    names_list = [names.get_first_name(gender='male' if gender == 0 else 'female') for gender, _ in x]

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, names_list

# Set the number of nodes you want in the graph
# n = 10  # Example: 10 nodes
# graph_data, node_names = create_graph(n)
# print(graph_data)
# print(node_names)