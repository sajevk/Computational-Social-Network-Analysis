import networkx as nx
from itertools import product

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.cluster.vq import vq, kmeans
import numpy as np
import scipy as sp
import random
import platform
import community
import operator

facebook = "facebook_combined.txt"
amazon = "Amazon0301.txt"

# Reference code for networkx from https://networkx.readthedocs.io/en/latest/_modules/networkx/algorithms/community/quality.html
def modularity(G, communities, weight='weight'):
    r"""Returns the modularity of the given partition of the graph.
    Modularity is defined in [1]_ as
    .. math::
        Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_ik_j}{2m}\right)
            \delta(c_i,c_j)
    where *m* is the number of edges, *A* is the adjacency matrix of
    `G`, :math:`k_i` is the degree of *i* and :math:`\delta(c_i, c_j)`
    is 1 if *i* and *j* are in the same community and 0 otherwise.
    Parameters
    ----------
    G : NetworkX Graph
    communities : list
        List of sets of nodes of `G` representing a partition of the
        nodes.
    Returns
    -------
    Q : float
        The modularity of the paritition.
    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.
    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.algorithms.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285704
    References
    ----------
    .. [1] M. E. J. Newman *Networks: An Introduction*, page 224.
       Oxford University Press, 2011.
    """
    # if not is_partition(G, communities):
    #    raise NotAPartition(G, communities)

    multigraph = G.is_multigraph()
    directed = G.is_directed()
    m = G.size(weight=weight)
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in G[u][v].items())
            else:
                w = G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm


# Performance and associated helper functions taken from networkx source code...
def intra_community_edges(G, partition):
    """Returns the number of intra-community edges according to the given
    partition of the nodes of `G`.
    `G` must be a NetworkX graph.
    `partition` must be a partition of the nodes of `G`.
    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.
    """
    return sum(G.subgraph(block).size() for block in partition)


def inter_community_edges(G, partition):
    """Returns the number of inter-community edges according to the given
    partition of the nodes of `G`.
    `G` must be a NetworkX graph.
    `partition` must be a partition of the nodes of `G`.
    The *inter-community edges* are those edges joining a pair of nodes
    in different blocks of the partition.
    Implementation note: this function creates an intermediate graph
    that may require the same amount of memory as required to store
    `G`.
    """
    # Alternate implementation that does not require constructing a new
    # graph object (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in G.edges() if aff[u] != aff[v])
    #
    return nx.quotient_graph(G, partition, create_using=nx.MultiGraph()).size()


def inter_community_non_edges(G, partition):
    """Returns the number of inter-community non-edges according to the
    given partition of the nodes of `G`.
    `G` must be a NetworkX graph.
    `partition` must be a partition of the nodes of `G`.
    A *non-edge* is a pair of nodes (undirected if `G` is undirected)
    that are not adjacent in `G`. The *inter-community non-edges* are
    those non-edges on a pair of nodes in different blocks of the
    partition.
    Implementation note: this function creates two intermediate graphs,
    which may require up to twice the amount of memory as required to
    store `G`.
    """
    # Alternate implementation that does not require constructing two
    # new graph objects (but does require constructing an affiliation
    # dictionary):
    #
    #     aff = dict(chain.from_iterable(((v, block) for v in block)
    #                                    for block in partition))
    #     return sum(1 for u, v in nx.non_edges(G) if aff[u] != aff[v])
    #
    return inter_community_edges(nx.complement(G), partition)


def performance(G, partition):
    """Returns the performance of a partition.
    The *performance* of a partition is the ratio of the number of
    intra-community edges plus inter-community non-edges with the total
    number of potential edges.
    Parameters
    ----------
    G : NetworkX graph
        A simple graph (directed or undirected).
    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes. Each block of the partition represents a
        community.
    Returns
    -------
    float
        The performance of the partition, as defined above.
    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.
    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <http://arxiv.org/abs/0906.0612>
    """
    # Compute the number of intra-community edges and inter-community
    # edges.
    intra_edges = intra_community_edges(G, partition)
    inter_edges = inter_community_non_edges(G, partition)
    # Compute the number of edges in the complete graph (directed or
    # undirected, as it depends on `G`) on `n` nodes.
    #
    # (If `G` is an undirected graph, we divide by two since we have
    # double-counted each potential edge. We use integer division since
    # `total_pairs` is guaranteed to be even.)
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    return (intra_edges + inter_edges) / total_pairs


def readgraph(readedges):
    num_nodes = 100
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Make a graph with num_nodes nodes and zero edges
    # Plot the nodes using x,y as the node positions

    graph = nx.Graph()
    for i in range(num_nodes):
        node_name = str(i)
        graph.add_node(node_name)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices

    if readedges:
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
            j += 1
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
        adjacency_matrix = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

        for i in range(adjacency_matrix.nnz):
            adjacency_matrix.data[i] = 1.0

        graph = nx.to_networkx_graph(adjacency_matrix)

    return graph


def readgraph(id, state=False):

    if id == 'facebook':
        print('Analysing Facebook community')
        if state:
            graph = nx.read_edgelist(facebook,create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(facebook)

    if id == 'amazon':
        print('Analysing Amazon community')
        if state:
            graph = nx.read_edgelist(amazon, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(amazon)

    return graph


def readpositions(graph_size):
    x = [random.random() for i in range(graph_size)]
    y = [random.random() for i in range(graph_size)]

    x = np.array(x)
    y = np.array(y)

    pos = dict()
    for i in range(graph_size):
        pos[i] = x[i], y[i]

    return pos


def read_default(graph):
    num_nodes = graph.number_of_nodes()
    A = nx.adjacency_matrix(graph)
    x = [random.random() for i in range(num_nodes)]
    y = [random.random() for i in range(num_nodes)]

    x = np.array(x)
    y = np.array(y)

    # Now add some edges - use Delaunay tesselation
    # to produce a planar graph. Delaunay tesselation covers the
    # convex hull of a set of points with triangular simplices (in 2D)

    points = np.column_stack((x, y))
    dl = Delaunay(points)
    tri = dl.simplices
    
    edges = np.zeros((2, 6 * len(tri)), dtype=int)
    # data = np.ones(6 * len(points))
    j = 0
    for i in range(len(tri)):
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][1]
        j = j + 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][0]
        j = j + 1
        edges[0][j] = tri[i][0]
        edges[1][j] = tri[i][2]
        j = j + 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][0]
        j = j + 1
        edges[0][j] = tri[i][1]
        edges[1][j] = tri[i][2]
        j = j + 1
        edges[0][j] = tri[i][2]
        edges[1][j] = tri[i][1]
        j = j + 1

    data = np.ones(6 * len(tri))
    adjacency_matrix = sp.sparse.csc_matrix((data, (edges[0, :], edges[1, :])))

    for i in range(adjacency_matrix.nnz):
        adjacency_matrix.data[i] = 1.0

    graph = nx.to_networkx_graph(adjacency_matrix)
    return graph



def count_edge_cuts(graph, w0, w1, w2, method):
    edge_cut_count = 0
    edge_uncut_count = 0
    for edge in graph.edges_iter():
        # This may be inefficient but I'll just check if both nodes are in 0, 1, or two
        if edge[0] in w0 and edge[1] in w0:
            edge_uncut_count += 1
        elif edge[0] in w1 and edge[1] in w1:
            edge_uncut_count += 1
        elif edge[0] in w2 and edge[1] in w2:
            edge_uncut_count += 1
        else:
            edge_cut_count += 1
    print('Community detection method is: ', method)
    print('Edge cuts: ', edge_cut_count)
    print('Contained edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count



def newman(G):
    if len(G.nodes()) == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        eb_il = eb.items()
        # eb_il.sort(key=lambda x: x[1], reverse=True)
        eb_il_sorted = sorted(eb_il, key=lambda x: x[1], reverse=True)
        return eb_il_sorted[0][0]

    components = list(nx.connected_component_subgraphs(G))

    while len(components) == 1:
        G.remove_edge(*find_best_edge(G))
        components = list(nx.connected_component_subgraphs(G))

    result = [c.nodes() for c in components]

    looper = 0
    for c in components:
        looper += 1
        result.extend(newman(c))

    return result

def count_edge_cuts_from_list(graph, list_of_partitions, method):
    edge_cut_count = 0
    edge_uncut_count = 0
    for edge in graph.edges_iter():
        found = False
        for lst in list_of_partitions:
            # This may be inefficient but I'll just check if both nodes are in 0, 1, or two
            if edge[0] in lst and edge[1] in lst and not found:
                edge_uncut_count += 1
                found = True
        if not found:
            edge_cut_count += 1
    print('Community detection method is: ', method)
    print('Edge cuts: ', edge_cut_count)
    print('Contained edges: ', edge_uncut_count)
    return edge_cut_count, edge_uncut_count


def modularity_eval(graph, list_of_partitions):
    print("Calculating modularity")
    mod = modularity(graph, list_of_partitions)
    return mod

def analysepartition(graph):
    partitions = community.best_partition(graph)
    communities = [partitions.get(node) for node in graph.nodes()]
    community_count = set(communities) 
    print("List of Partitions Detected: ", len(community_count))
    for i in community_count:
        print("Count community {} is {}.".format(i, communities.count(i)))
    return communities

def cluster(graph, feat, pos, eigen_pos, cluster_type):
    book, distortion = kmeans(feat, 3)
    codes, distortion = vq(feat, book)

    nodes = np.array(range(graph.number_of_nodes()))
    w0 = nodes[codes == 0].tolist()
    w1 = nodes[codes == 1].tolist()
    w2 = nodes[codes == 2].tolist()
    print("W0 ", w0)
    print("W1 ", w1)
    print("W2 ", w2)
    count_edge_cuts(graph, w0, w1, w2, cluster_type)
    communities = list()
    communities.append(w0)
    communities.append(w1)
    communities.append(w2)
    mod = modularity_eval(graph, communities)
    print("Modularity: ", mod)

    plt.figure(3)
    nx.draw_networkx_nodes(graph, eigen_pos, node_size=40, hold=True, nodelist=w0, node_color='m')
    nx.draw_networkx_nodes(graph, eigen_pos, node_size=40, hold=True, nodelist=w1, node_color='b')

    plt.figure(2)
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=True, nodelist=w0, node_color='m')
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=True, nodelist=w1, node_color='b')


def GraphCheck(graph):
    print(" Dimensions of the Graph:") 
    print(nx.info(graph))
    max_degree = 0
    min_degree = 999999
    ave_degree = 0
    counter = 0
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree > max_degree:
            max_degree = degree
        if min_degree > degree:
            min_degree = degree
        ave_degree += degree
        counter += 1
    ave_degree = ave_degree / counter
 
    print("Maximum Degree Node ", max_degree)
    print("Minimum Degree Node ", min_degree)
    print("Average Degree Node ", ave_degree)

def newman_eval(G):
    comp = newman(G)
    print("Newman's list ", len(comp))
    return comp


def plot_graph(graph, pos, fig_num):

    label = dict()
    label_pos = dict()
    for i in range(graph.number_of_nodes()):
        label[i] = i
        label_pos[i] = pos[i][0]+0.02, pos[i][1]+0.02

    fig = plt.figure(fig_num, figsize=(8, 8))
    fig.clf()
    nx.draw_networkx_nodes(graph, pos, node_size=40, hold=False)
    nx.draw_networkx_edges(graph, pos, hold=True)
    nx.draw_networkx_labels(graph, label_pos, label, font_size=10, hold=True)
    fig.show()

def editsidenodes(graph, node, neighbours):
    with suppress(Exception):   # Needed if the edge was already removed.
        first = True
        for neighbour in neighbours:
            if not first:
                graph.remove_edge(node, neighbour)
            first = False
    return graph


def readbasic(graph):
    bt = nx.betweenness_centrality(graph)
    sorted_bt = sorted(bt.items(), key=operator.itemgetter(1))
    sorted_bt.reverse()
    sorted_list = list(sorted_bt)
    node_index = 0
    while nx.number_connected_components(graph) < 4:
        top_node = sorted_list[node_index][0]
        top_neighbours = nx.neighbors(graph, top_node)
        graph = editsidenodes(graph, top_node, top_neighbours)
        node_index += 1 
    components = sorted(nx.connected_components(graph), key = len, reverse=True)
    return_components = list()
    for i in range(nx.number_connected_components(graph)):
        print(components[i])
        return_components.append(components[i])
    return return_components


def readcommunity(community_list, index):
    return_list = list()
    node = 0
    for i in community_list:
        if community_list[node] == index:
            return_list.append(node)
        node += 1
    return return_list



def execute():
    gr = readgraph("facebook")
    pos = readpositions(gr.number_of_nodes())
    am = nx.adjacency_matrix(gr)
    gr = nx.Graph(am)
    plot_graph(gr, pos, 1)
    num_nodes = gr.number_of_nodes()
    GraphCheck(gr)
    plot_graph(gr, pos, 2)

    # Networkx algorithm
    partitions = analysepartition(gr)
    partitions_count = set(partitions)
    list_of_partitions = list()
    length = len(partitions_count)
    for i in range(length):
        comm = readcommunity(partitions, i)
        print(comm)
        list_of_partitions.append(comm)
    count_edge_cuts_from_list(gr, list_of_partitions, "Extended Community")
    mod = modularity_eval(gr, list_of_partitions)
    print("Modularity: ==============================> ", mod)
    # Modified
    gr = nx.Graph(am)
    communities = readbasic(gr)
    gr = nx.Graph(am)
    count_edge_cuts_from_list(gr, communities, "Modified")
    mod = modularity_eval(gr, communities)
    print("Modularity:  ==============================>  ", mod)
    eigen_pos = dict()
    deg = am.sum(0)
    diags = np.array([0])
    D = sp.sparse.spdiags(deg, diags, am.shape[0], am.shape[1])
    Dinv = sp.sparse.spdiags(1 / deg, diags, am.shape[0], am.shape[1])
    # Normalised laplacian
    L = Dinv * (D - am)
    E, V = sp.sparse.linalg.eigs(L, 3, None, 100.0, 'SM')
    V = V.real

    for i in range(num_nodes):
        eigen_pos[i] = V[i, 1].real, V[i, 2].real

    plot_graph(gr, eigen_pos, 3)

    # Now let's see if the eigenvectors are good for clustering
    # Use kmeans to cluster the points in the vector V
    features = np.column_stack((V[:, 1], V[:, 2]))
    cluster(gr, features, pos, eigen_pos, "Eigen Values")
    cluster(gr, am.todense(), pos, eigen_pos, "Adjacency")
    gr = nx.Graph(am)
    gncomps = newman_eval(gr)
    count_edge_cuts_from_list(gr, gncomps, "Newman")
    mod = modularity_eval(gr, gncomps)
    print("Modularity  ==============================>  ", mod)



execute()