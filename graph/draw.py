import matplotlib.pyplot as plt
import networkx as nx
from build import create_graph

def draw_graph(data, names):
    # Create a new graph
    G = nx.Graph()

    # Add nodes with the name and age attributes
    for i, (name, attr) in enumerate(zip(names, data.x)):
        G.add_node(i, name=name, age=int(attr[1]))

    # Add edges and edge labels based on the edge_index and edge_attr from PyG data
    edge_labels = {}
    for start, end, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
        G.add_edge(start.item(), end.item())
        # Use 'married' or 'childOf' based on the edge_attr
        edge_labels[(start.item(), end.item())] = 'married' if attr.item() == 1 else 'childOf'

    # Define node colors: 'blue' for male, 'red' for female
    node_colors = ['blue' if gender == 0 else 'red' for gender, _ in data.x.tolist()]

    # Position nodes using spring layout
    pos = nx.spring_layout(G)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Draw the node labels with name and age
    labels = {i: f"{names[i]}\n{int(data.x[i, 1])}" for i in range(len(names))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)  # Adjust font size as needed

    # Show the plot
    plt.show()

n = 4  # Example: 10 nodes
graph_data, node_names = create_graph(n)
print(graph_data)
print(node_names)

draw_graph(graph_data, node_names)