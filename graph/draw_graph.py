import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from build_graph import GraphBuilder
from build_relationship import RelationshipBuilder

rb = RelationshipBuilder('/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families/greyjoy.txt')
gb = GraphBuilder(rb)
d = gb.build_graph()

def visualize_graph(G):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True)
    plt.show()

G = to_networkx(d, to_undirected=False)
visualize_graph(G)



