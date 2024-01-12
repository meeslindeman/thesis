import matplotlib.pyplot as plt
import networkx as nx
from graph.build import create_family_tree, create_data_object

def draw_graph(data):
    """
    Draw a directed graph based on the given data.

    Parameters:
    data (object): The data object containing the necessary information for drawing the graph.
    """

    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes with only age attributes (since names are not used)
    for i, attr in enumerate(data.x):
        G.add_node(i, age=int(attr[1]), hair=int(attr[2]), height=int(attr[3]))

    # Add edges and edge labels based on the edge_index and edge_attr from PyG data
    edge_labels = {}
    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        # Use 'married' or 'childOf' based on the edge_attr
        edge_labels[(start.item(), end.item())] = 'married' if attr.item() == 0 else 'childOf'

    # Define node colors: 'blue' for male, 'red' for female
    node_colors = ['tab:blue' if gender == 0 else 'tab:red' for gender, _, _, _ in data.x.tolist()]

    # Position nodes using spring layout
    pos = nx.spring_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)

    # Draw the directed edges (arrows)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5)

    # Draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Draw the node labels with index and age
    labels = {i: f"Index: {i}\nAge: {int(data.x[i, 1])}\nHair Color: {'brown' if data.x[i, 2] == 0 else 'blonde' if data.x[i, 2] == 1 else 'black'}\nHeight: {int(data.x[i, 3])} cm" for i in range(len(data.x))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)  # Adjust font size as needed

    # Show the plot
    plt.show()

# Example usage:
family_tree = create_family_tree(3)  # Generate 3 generations
graph_data = create_data_object(family_tree)
# draw_graph(graph_data)