import networkx as nx
import numpy as np

# Standard stochastic Block model
def SBM(P: np.array, pi:list, n:int):

    assert np.sum(pi) > 0.9999


    community_list = [i for i in range(len(pi))]

    G = nx.Graph()
    # create node list and corresponding community
    node_list = [(node, {'community':np.random.choice(community_list, p=pi)}) for node in range(n)]
    G.add_nodes_from(node_list)

    # add edges
    nodes_community=nx.get_node_attributes(G,'community')
    for i in range(len(G.nodes)):
        for j in range(i, len(G.nodes)):

            if np.random.uniform() <= P[nodes_community[i], nodes_community[j]]:
                G.add_edge(i,j)

    
    return G



 

