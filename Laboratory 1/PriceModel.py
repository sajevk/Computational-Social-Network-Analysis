import networkx as nx
import random
import Utility as utilites
import cProfile

citationcount=25
nums = 500
start = 1

def generaterandomalign(nums, degrees):
    limit = 0
    init = 0
    random_chance = random.randint(1, 10)
    if random_chance <= start:
        return random.randint(0, len(degrees)-1)

    while nums > limit:
        limit += degrees[init]
        init += 1
    return init


def citationupdate(graph, i, degrees):
    if i <= citationcount:

        for j in range(i):
            graph.add_edge(i, j)
            degrees[j] += 1

    else:
        for j in range(citationcount):

            total=graph.number_of_edges()
            random_no = random.randint(0,total)
            node_to_cite=generaterandomalign(random_no, degrees)
            graph.add_edge(i,node_to_cite)
            degrees[node_to_cite] += 1

def printdiag(G):
    for node in G.edges_iter():
        print(node)

def executemodel():
    degrees = {}
    graph = nx.DiGraph()
    graph.add_node(0)
    for i in range(nums):
        print(i)
        degrees[i] = 0
        graph.add_node(i)
        citationupdate(graph, i, degrees)
    return graph

def execute():
    full_graph = executemodel()

    printdiag(full_graph)
    utilites.draw_degree(full_graph, False)
    input("Press enter to finish.")

execute()