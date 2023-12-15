import matplotlib.pyplot as plt
import networkx as nx
from build import create_family_tree, create_data_object

def draw_graph(data, names):
    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes with the name and age attributes
    for i, (name, attr) in enumerate(zip(names, data.x)):
        G.add_node(i, name=name, age=int(attr[1]))

    # Add edges and edge labels based on the edge_index and edge_attr from PyG data
    edge_labels = {}
    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        # Use 'married' or 'childOf' based on the edge_attr
        edge_labels[(start.item(), end.item())] = 'married' if attr.item() == 0 else 'childOf'

    # Define node colors: 'blue' for male, 'red' for female
    node_colors = ['tab:blue' if gender == 0 else 'tab:red' for gender, _ in data.x.tolist()]

    # Position nodes using spring layout
    pos = nx.spring_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)

    # Draw the directed edges (arrows)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5)

    # Draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Draw the node labels with name and age
    labels = {i: f"{names[i]}\n{int(data.x[i, 1])}" for i in range(len(names))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)  # Adjust font size as needed

    # Show the plot
    plt.show()

# Example usage:
family_tree, names_list = create_family_tree(3)  # Generate 3 generations
graph_data = create_data_object(family_tree)

draw_graph(graph_data, names_list)