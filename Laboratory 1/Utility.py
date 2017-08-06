import networkx as nx
import matplotlib.pyplot as plt


def findstrongconnection(graph):
    comps = nx.strongly_connected_components(graph)
    return comps


def generate_graph(G):
    A = nx.to_scipy_sparse_matrix(G)
    fig = plt.figure()
    plt.spy(A)
    fig.show()


def draw_degree(G, inset):
    degree_sequence=sorted(nx.degree(G).values(), reverse=True) # degree sequence
    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree  plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
    if inset:
        plt.axes([0.45,0.45,0.45,0.45])
        Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
        pos = nx.spring_layout(Gcc)
        plt.axis('off')
        nx.draw_networkx_nodes(Gcc, pos, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
    plt.show()


def check_directed(graph):
    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges())
    num_strongly_connected_components = nx.number_strongly_connected_components(graph)
    print("Strongly Connection - ", num_strongly_connected_components)
    draw_degree(graph, False)