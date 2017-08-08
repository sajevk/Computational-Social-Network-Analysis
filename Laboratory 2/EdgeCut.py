import networkx as nx
import numpy as np
import scipy as sp
import random
import platform
import os
from scipy.spatial import Delaunay

# import the plotting functionality from matplotlib
import matplotlib.pyplot as plt

# import kmeans
from scipy.cluster.vq import vq, kmeans

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

if platform.system() != "Darwin":
    facebook = "Amazon0302.txt"
    twitter = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\twitter_combined.txt"
    google = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\web-Google1.txt"
    roads_CA = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\roadNet-CA.txt"
    amazon = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\com-amazon-ungraph.txt"
    college = "C:\\Users\\paulb\\PycharmProjects\\COMP47270\\Datasets\\collegemsg.txt"
else:
    facebook = "Amazon0302.txt"
    twitter = "/Users/paulba/PycharmProjects/COMP47270/Datasets/twitter_combined.txt"
    google = "/Users/paulba/PycharmProjects/COMP47270/Datasets/web-Google1.txt"
    roads_CA = "/Users/paulba/PycharmProjects/COMP47270/Datasets/roadNet-CA.txt"
    amazon = "/Users/paulba/PycharmProjects/COMP47270/Datasets/com-amazon-ungraph.txt"
    college = "/Users/paulba/PycharmProjects/COMP47270/Datasets/collegemsg.txt"


def get_default_graph():
    num_nodes = 100
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Make a graph with num_nodes nodes and zero edges
    # Plot the nodes using x,y as the node positions

    G = nx.Graph()
    for i in range(num_nodes):
        node_name = str(i)
        G.add_node(node_name)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

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


def load_graph(name, as_directed=True):
    graph = None
    if name == 'facebook':
        print('Loading facebook graph')
        if as_directed:
            graph = nx.read_edgelist(facebook,create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(facebook)

    if name == 'twitter':
        print('Loading twitter graph')
        if as_directed:
            graph = nx.read_edgelist(twitter, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(twitter)

    if name == 'google':
        print('Loading google graph')
        if as_directed:
            graph = nx.read_edgelist(google, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(google)

    if name == 'amazon':
        print('Loading amazon graph')
        if as_directed:
            graph = nx.read_edgelist(amazon, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(amazon)

    if name == 'roads_CA':
        print('Loading roads_CA graph')
        if as_directed:
            graph = nx.read_edgelist(roads_CA, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(roads_CA)

    if name == 'college':
        print('Loading collegemsg')
        if as_directed:
            graph = nx.read_edgelist(college, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(college)

    if graph is None:
        graph = get_default_graph()

    return graph


def plot_graph(g, pos, fig_num):
    label = dict()
    label_pos=dict()
    for i in range(g.number_of_nodes()):
        label[i] = i
        label_pos[i] = pos[i][0]+0.02, pos[i][1]+0.02

    fig = plt.figure(fig_num, figsize=(8, 8))
    fig.clf()
    nx.draw_networkx_nodes(g,
                           pos,
                           node_size=40,
                           hold=False,
                           )

    nx.draw_networkx_edges(g,pos, hold=True)

    nx.draw_networkx_labels(g,
                            label_pos,
                            label,
                            font_size=10,
                            hold=True,
                            )
    fig.show()


def get_random_positions(graph_size):
    x = [random.random() for i in range(graph_size)]
    y = [random.random() for i in range(graph_size)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(graph_size):
        pos[i] = x[i], y[i]

    return pos

def get_eigen_pos(A, num_nodes):
    #   eigen_pos holds the positions
    eigen_pos = dict()
    deg = A.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg,diags,A.shape[0],A.shape[1])
    Dinv = sp.sparse.spdiags(1/deg,diags,A.shape[0],A.shape[1])
    # Normalised laplacian
    L = Dinv*(D - A)
    E, V= sp.sparse.linalg.eigs(L,3,None,100.0,'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i,1].real,V[i,2].real

    return eigen_pos, V


def cluster_nodes(G, feat, pos, eigen_pos):
    book, distortion = kmeans(feat, 3)
    codes, distortion = vq(feat, book)

    nodes = np.array(range(G.number_of_nodes()))
    W0 = nodes[codes == 0].tolist()
    W1 = nodes[codes == 1].tolist()
    W2 = nodes[codes == 2].tolist()
    print("W0 ", W0)
    print("W1 ", W1)
    print("W2 ", W2)
    #plt.figure(3)
    #nx.draw_networkx_nodes(G,
    #                       eigen_pos,
    #                       node_size=40,
    #                       hold=True,
    #                       nodelist=W0,
    #                       node_color='m'
    #                       )
    #nx.draw_networkx_nodes(G,
    #                       eigen_pos,
    #                       node_size=40,
    #                       hold=True,
    #                       nodelist=W1,
    #                       node_color='b'
    #                       )
    plt.figure(2)
    nx.draw_networkx_nodes(G,
                           eigen_pos,
                           node_size=40,
                           hold=True,
                           nodelist=W0,
                           node_color='m'
                           )
    nx.draw_networkx_nodes(G,
                           eigen_pos,
                           node_size=40,
                           hold=True,
                           nodelist=W1,
                           node_color='b'
                           )


def main():
    graph = load_graph("facebook")
    pos=get_random_positions(graph.number_of_nodes())
    # Need to rebuild the graph...
    A = nx.adjacency_matrix(graph)   # will use the adjacency matrix later
    graph = nx.Graph(A)
    # plot the graph...
    plot_graph(graph, pos, 1)

    #Use eigen_pos for positions
    eigen_pos, V = get_eigen_pos(A, graph.number_of_nodes())
    # plot the graph...
    plot_graph(graph, eigen_pos, 2)

    # Cluster the nodes using kmeans
    features = np.column_stack((V[:, 1], V[:, 2]))
    cluster_nodes(graph, features, pos, eigen_pos)

    # Finally, use the columns of A directly for clustering
    #cluster_nodes(graph, A.todense(), pos, eigen_pos)

    print("processing completed")


main()