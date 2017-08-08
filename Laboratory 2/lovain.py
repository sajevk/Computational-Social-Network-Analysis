import networkx as nx
import matplotlib.pyplot as plt
import community as cm

data_file_name = 'facebook_combined.txt'
G_fb = nx.read_edgelist(data_file_name, comments='#', create_using=nx.Graph(), nodetype=int)

print(nx.info(G_fb))

spring_pos = nx.spring_layout(G_fb)


def detectCommunities(G_fb, layout):
    parts = cm.best_partition(G_fb)
    values = [parts.get(node) for node in G_fb.nodes()]

    plt.axis("off")
    nx.draw_networkx(G_fb, pos=layout, cmap=plt.get_cmap("jet"), node_color=values, node_size=35, with_labels=False)
    # plt.savefig("results/communities.png", form
    # at = "PNG")

    print "Louvain Modularity: ", cm.modularity(parts, G_fb)
    print "Louvain Partition: ", parts
plt.show()


detectCommunities(G_fb, spring_pos)