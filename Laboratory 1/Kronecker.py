import numpy as np
import networkx as nx
import Utility as utility


def generatekroneker():
    matrix = np.matrix('1 0 0 1 0 1; 1 1 1 1 1 1; 1 0 1 0 1 0; 0 0 0 1 0 1; 1 1 1 1 1 0; 1 1 1 1 1 1')
    othermatrix =np.matrix('1 1 1 1 1 1; 1 0 0 0 0 0; 0 1 0 1 0 1; 1 1 1 1 1 0; 1 1 0 1 1 1; 1 1 0 1 1 1')
    kron_result=othermatrix

    for item in range(6):
         output = np.kron(kron_result, othermatrix)
    G = nx.from_numpy_matrix(kron_result)
    return G


def exploregraph(G):
    print("Nodes: ", G.number_of_nodes())
    print("Edges: ", G.number_of_edges())
    utility.check_directed(G, False)
    print('')


G = generatekroneker()
exploregraph(G)