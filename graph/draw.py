import matplotlib.pyplot as plt
import networkx as nx
from dataset import FamilyGraphDataset

def draw_graph(data):
    G = nx.DiGraph()

    # Add nodes with age attributes
    for i, attr in enumerate(data.x):
        G.add_node(i, age=int(attr[1]))

    # Initialize dictionaries for different relationships
    married_labels, child_of_labels, gave_birth_to_labels = {}, {}, {}

    # Add edges and categorize edge labels by relationship type
    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        if attr.tolist() == [1, 0, 0]:
            married_labels[(start.item(), end.item())] = 'married'
        elif attr.tolist() == [0, 1, 0]:
            child_of_labels[(start.item(), end.item())] = 'child-of'
        elif attr.tolist() == [0, 0, 1]:
            gave_birth_to_labels[(start.item(), end.item())] = 'gave-birth-to'

    # Define node colors based on gender
    node_colors = ['tab:blue' if gender == 0 else 'tab:red' for gender, _ in data.x.tolist()]

    # Position nodes using Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5, connectionstyle='arc3,rad=0.1')

    # Draw edge labels for 'married' with specific alignment
    nx.draw_networkx_edge_labels(G, pos, edge_labels=married_labels, font_size=7, horizontalalignment='center')

    # Draw edge labels for 'child-of' with specific alignment
    nx.draw_networkx_edge_labels(G, pos, edge_labels=child_of_labels, font_size=7, horizontalalignment='left', clip_on=False, verticalalignment='bottom')

    # Draw edge labels for 'gave-birth-to' with specific alignment
    nx.draw_networkx_edge_labels(G, pos, edge_labels=gave_birth_to_labels, font_size=7, horizontalalignment='right', clip_on=False, verticalalignment='top')

    # Draw node labels with index and age
    labels = {i: f"{i}\nAge: {int(data.x[i, 1])}" for i in range(len(data.x))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.show()


datasetx = FamilyGraphDataset(root=f'data/gens={2}')
datasety = FamilyGraphDataset(root=f'data/gens={3}')

graph_2 = datasetx[3]
graph_3 = datasety[0]
draw_graph(graph_2)