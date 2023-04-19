
import numpy as np
import networkx as nx
from numpy.testing import assert_array_less

import sys, os
sys.path.append('../')
from  GTST.MMD import BoostrapMethods, MMD_b, MMD_l, MMD_u, MONK_EST

default_eigvalue_precision = float("-1e-5")

def generate_H0_false() -> tuple:
    """
    Generate two sampels of binomial graphs when H0 is true
    """


    n1 = n2 = 70  # sample sizes
    # generate two samples
    g1 = [nx.fast_gnp_random_graph(30,0.3) for _ in range(n1)]
    g2 = [nx.fast_gnp_random_graph(30,0.5) for _ in range(n2)]

    # Set node labels as the degree
    for j in range(len(g1)):
        nx.set_node_attributes(g1[j],  {key: str(value) for key, value in dict(g1[j].degree).items()} , 'label')
    for j in range(len(g2)):
        nx.set_node_attributes(g2[j], {key: str(value) for key, value in dict(g2[j].degree).items()}, 'label')

    # For loop to label each edge either 'a' or 'b' the labelling probabilities are different for the two samples
    for j in range(len(g1)):
        nx.set_edge_attributes(g1[j], {(i,k):np.random.choice(['a','b'], p = [0.3,0.7]) for i,k in g1[j].edges }, 'edge_label')
    for j in range(len(g2)):
        nx.set_edge_attributes(g2[j], {(i,k):np.random.choice(['a','b'], p = [0.7,0.3]) for i,k in g2[j].edges }, 'edge_label')


    for j in range(len(g1)):
        nx.set_node_attributes(g1[j], dict( ( (i, np.random.normal(loc = 0, scale = 0.01, size = (1,))) for i in range(len(g1[j])) ) ), 'attr')
    for j in range(len(g2)):
        nx.set_node_attributes(g2[j], dict( ( (i, np.random.normal(loc = 0.1, scale = 0.01, size = (1,))) for i in range(len(g2[j])) ) ), 'attr')

    return g1, g2


def generate_H0_false_but_same_topology() -> tuple:
    """
    Generate two sampels of binomial graphs when H0 is true
    """


    n1 = n2 = 70  # sample sizes
    # generate two samples
    g1 = [nx.fast_gnp_random_graph(30,0.3) for _ in range(n1)]
    g2 = [nx.fast_gnp_random_graph(30,0.3) for _ in range(n2)]

    # Set node labels as the degree
    for j in range(len(g1)):
        nx.set_node_attributes(g1[j],  {key: np.random.choice(['a','b', 'c'], p = [0.3,0.5, 0.2]) for key, _ in dict(g1[j].degree).items()} , 'label')
    for j in range(len(g2)):
        nx.set_node_attributes(g2[j], {key: np.random.choice(['a','b', 'c'], p = [0.4,0.1, 0.5]) for key, _ in dict(g2[j].degree).items()}, 'label')

    # For loop to label each edge either 'a' or 'b' the labelling probabilities are different for the two samples
    for j in range(len(g1)):
        nx.set_edge_attributes(g1[j], {(i,k):np.random.choice(['a','b'], p = [0.3,0.7]) for i,k in g1[j].edges }, 'edge_label')
    for j in range(len(g2)):
        nx.set_edge_attributes(g2[j], {(i,k):np.random.choice(['a','b'], p = [0.7,0.3]) for i,k in g2[j].edges }, 'edge_label')


    for j in range(len(g1)):
        nx.set_node_attributes(g1[j], dict( ( (i, np.random.normal(loc = 0, scale = 0.01, size = (1,))) for i in range(len(g1[j])) ) ), 'attr')
    for j in range(len(g2)):
        nx.set_node_attributes(g2[j], dict( ( (i, np.random.normal(loc = 0.1, scale = 0.01, size = (1,))) for i in range(len(g2[j])) ) ), 'attr')



    def edge_dist(loc, scale ):
        return np.random.normal(loc = loc, scale = scale)
    def add_weight(G, loc, scale ):
        edge_w = dict()
        for e in G.edges():
            edge_w[e] = edge_dist(loc, scale)
        return edge_w


    for G in g1:
        nx.set_edge_attributes(G, add_weight(G, loc = 0.5, scale = 0.05), "weight")
    for G in g2:
        nx.set_edge_attributes(G, add_weight(G, loc = 0.5, scale = 3), "weight")

    return g1, g2



def generate_H0_false_directed():

    n1 = n2 = 70
    g1_di = [nx.fast_gnp_random_graph(30,0.2) for _ in range(n1)]  # sample 1
    g2_di = [nx.fast_gnp_random_graph(30,0.2) for _ in range(n2)]  # sample 2

    # for loop for both samples to convert the networkx graph to a networkx directed graph object
    for j in range(len(g1_di)):
        g1_di[j] = nx.DiGraph(g1_di[j])
    for j in range(len(g2_di)):
        g2_di[j] = nx.DiGraph(g2_di[j])

    # for loop for both samples that removes edges with different removal probabilties between the two samples
    for j in range(len(g1_di)):
        edges= list(g1_di[j].edges())
        for e,u in edges:
            if np.random.uniform() <0.3:
                g1_di[j].remove_edge(e,u)
    for j in range(len(g2_di)):
        edges= list(g2_di[j].edges())
        for e,u in edges:
            if np.random.uniform() <0.7:
                g2_di[j].remove_edge(e,u)

    return g1_di, g2_di


def assert_positive_eig(K):
    """Assert true if the calculated kernel matrix is valid."""
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    assert_array_less(default_eigvalue_precision, min_eig)

def assert_low_p_val(p_values, info = ""):
    assert np.all(np.fromiter(p_values.values(), dtype=float) < 0.01), f"Some p value too high {p_values} for {info}"



def assert_low_p_val_from_K(K,g1,g2):
    # Test MMD, should give very low p-val
    pval = BoostrapMethods(list_of_functions=[MMD_u, MMD_b, MMD_l], function_arguments={'MMD_u':{'n1':len(g1), 'n2':len(g2)}, 
                                                                                        'MMD_b':{'n1':len(g1), 'n2':len(g2)}, 
                                                                                        'MMD_l':{'n1':len(g1), 'n2':len(g2)}})
    pval.Bootstrap(K, 1000)
    assert np.all(np.fromiter(pval.p_values.values(), dtype=float) < 0.01), f"Some p value too high {pval.p_values}"


def generate_Xs_H0_false():
    # Geenrate X1, X2
    G = nx.fast_gnp_random_graph(11, 0.25, seed=42)  # generate a random graph
    assert nx.is_connected(G)

    #  Add random weights to the graphs, either positive or negative
    for e in G.edges():
        if np.random.uniform() <0.1:
            w = np.random.uniform(low = 0.1, high = 0.3)
            G.edges[e[0], e[1]]['weight'] = -w
        else:
            w = np.random.uniform(low = 0.1, high = 0.3)
            G.edges[e[0], e[1]]['weight'] = w

    # Extract adjacency matrix and fill the diagonal so that the resulting matrix will be positive definite.
    A = np.array(nx.adjacency_matrix(G).todense())
    np.fill_diagonal(A, np.sum(np.abs(A), axis = 1)+0.1)

    # Copy the adjacency matrix, and remove some edges for that graph,  note the seed is assume to be 45 when G was constructed
    A_s = A.copy()
    A_s[7,4] = 0
    A_s[4,7] = 0
    A_s[5,2] = 0
    A_s[2,5] = 0
    A_s[0,3] = 0
    A_s[3,0] = 0
    A_s[10,1] = 0
    A_s[1,10] = 0
    A_s[8,6] = 0
    A_s[6,8] = 0
    A_s[8,5] = 0
    A_s[5,8] = 0


    # Simulate random variables one has A as its precision and one has A_s (the sparse copy of A) as its precision matrix.
    # Note the precision matrix is the inverse covariance.
    X1 = np.random.multivariate_normal(np.zeros(11),np.linalg.inv(A), size = 20000)
    X2 = np.random.multivariate_normal(np.ones(11)*0.5,np.linalg.inv(A_s), size = 20000)

    return X1,X2


