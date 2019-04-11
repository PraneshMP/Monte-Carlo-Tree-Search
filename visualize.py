import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def visualize(G):
    pos = graphviz_layout(G)
    values=[G.node[i]['attr_dict']['state'] for i in G]
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),node_values=values,node_color='r', node_size = 500)
    nx.draw_networkx_labels(G, pos,node_size=1500)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
    plt.show()