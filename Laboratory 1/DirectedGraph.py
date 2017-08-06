import random

import Utility as utility
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.cluster.vq import vq, kmeans
from scipy.spatial import Delaunay

facebook = "facebook_combined.txt"
amazon = "amazon0302.txt"


def adjusteigen(A, num_nodes):
    print("Re-plotting using eigenvectors")

    eigen_pos = dict()
    deg = A.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg, diags, A.shape[0], A.shape[1])
    Dinv = sp.sparse.spdiags(1 / deg, diags, A.shape[0], A.shape[1])
    L = Dinv * (D - A)
    E, V = sp.sparse.linalg.eigs(L, 3, None, 100.0, 'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i, 1].real, V[i, 2].real

    return eigen_pos, V


def get_default_graph():
    num_nodes = 100
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y) 

    G = nx.Graph()
    for i in range(num_nodes):
        node_name = str(i)
        G.add_node(node_name) 

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices

    edges = np.zeros((2, 6 * len(tri)), dtype=int)
    data = np.ones(6 * len(points))
    j = 0
    for i in range(len(tri)):
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][1]
        j += 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][0]
        j += 1
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][2]
        j +=  1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][0]
        j += 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][2]
        j += 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][1]
        j += 1

    data = np.ones(6 * len(tri))
    A = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

    for i in range(A.nnz):
        A.data[i] = 1.0

    G = nx.to_networkx_graph(A)

    return G


def load_graph(name, as_directed=False):
    graph = None
    if name == 'facebook':
        print('Loading facebook graph')
        if as_directed:
            graph = nx.read_edgelist(facebook,create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(facebook)

    if name == 'amazon':
        print('Loading amazon graph')
        if as_directed:
            graph = nx.read_edgelist(amazon, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(amazon)

    return graph


def draw(G, pos, fignum, draw_edges=True):
    fig = plt.figure(fignum, figsize=(8, 8))
    fig.clf()
    nx.draw_networkx_nodes(G,
                            pos,
                            node_size=20,
                            hold=False,
                        )

    if(draw_edges):
        nx.draw_networkx_edges(G,pos, hold=True)
    fig.show()




def executedirectedgraph():
    graph = load_graph('amazon', True)

    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges())

    checkconnectednodes(graph)
    utility.draw_degree(graph, False)
    input(" Enter ...")


def nodesintake(G, feat, pos, eigen_pos):
    book,distortion = kmeans(feat,3)
    codes,distortion = vq(feat, book)

    nodes = np.array(range(G.number_of_nodes()))
    W0 = nodes[codes == 0].tolist()
    W1 = nodes[codes == 1].tolist()
    W2 = nodes[codes == 2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)

    plt.figure(3)
    nx.draw_networkx_nodes(G, eigen_pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(G, eigen_pos, node_size=40, hold=True, nodelist=W1, node_color='b')
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos, node_size=40, hold=True, nodelist=W0, node_color='m')
    nx.draw_networkx_nodes(G, pos, node_size=40, hold=True, nodelist=W1, node_color='b')



def checkconnectednodes(G):
    try:
        num_strongly_connected_components = nx.number_strongly_connected_components(G)
        print("Strongly Connected - ", num_strongly_connected_components)
        comps = utility.findstrongconnection(G)
        component_sizes = []
        for nodes in comps:
            component_sizes.extend([len(nodes)])
        component_sizes.sort(reverse=True)
        print("Biggest connected : ", component_sizes[0]) 
        indegrees = G.in_degree().values()
        incount = 0
        for degree in indegrees:
            if degree == 0:
                incount += 1
        print("In degree - ", incount)
 
        outdegrees = G.out_degree().values()
        outcount = 0
        for degree in outdegrees:
            if degree == 0:
                outcount += 1
        print("Outdegree = ", outcount)

    except Exception as e:
        print("Exception: - ", e)


def executeundirectedgraph():
    graph = load_graph('facebook', True)

    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges())   

    checkconnectednodes(graph) 
    utility.draw_degree(graph, False) 
    input(" Enter ...")



def clusternodes(G):
    print ("Clustering -  networkx ")


def clustervalue(G):
    print("Clustering - local program ")


def clusteringmetric(G):
    print("Quality of Cluster")

def main():

    graph = load_graph("facebook")
    utility.draw_degree(graph, False)
    num_nodes = graph.number_of_nodes()

    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(num_nodes):
        pos[i] = x[i], y[i]

    print("Graph nodes: ", graph.number_of_nodes())
    print("Graph edges: ", graph.number_of_edges())  
    print("Position : ", len(pos)) 
    A = nx.adjacency_matrix(graph)
    graph = nx.Graph(A)
    print("nodes : ", graph.number_of_nodes())
    print("edges : ", graph.number_of_edges())   
    draw(graph, pos, 1, False) 
    draw(graph, pos, 2)
 
    eigenv_pos, V = adjusteigen(A, num_nodes)
    draw(graph, eigenv_pos, 3)

    print("Plotting spring layout")
    pos=nx.spring_layout(graph)
    draw(graph, pos, 4)

    # Look at the clustering
    features = np.column_stack((V[:, 1], V[:, 2]))
    nodesintake(graph, features, pos, eigenv_pos)


main()

#executedirectedgraph()
#executeundirectedgraph()