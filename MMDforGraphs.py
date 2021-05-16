
import networkx as nx
import numpy as np
import grakel as gk
import matplotlib
import scipy
import string
import pickle 
import pandas as pd
import math
from tqdm import tqdm
#import time
from datetime import datetime
import os

from numba import njit

import SBM


# Biased empirical maximum mean discrepancy
@njit
def MMD_b(K: np.array, n: int, m: int):

    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n ** 2) * Kx.sum() + 1.0 / (n * m) * Ky.sum() - 2.0 / (m ** 2) * Kxy.sum()

# Unbiased empirical maximum mean discrepancy
@njit
def MMD_u(K: np.array, n: int, m: int):
    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n* (n - 1.0)) * (Kx.sum() - np.diag(Kx).sum()) + 1.0 / (m * (m - 1.0)) * (Ky.sum() - np.diag(Ky).sum()) - 2.0 / (n * m) * Kxy.sum()

@njit
def BootstrapPval(B:int, K:np.array, n:int, m:int, seed:int):
    """
    :B number of bootstraps
    :K kernel array
    :n number of samples from sample 1
    :m number of samples from sample 2
    :seed for reproducibility
    """
    # mmd biased value of sample
    mmd_b_sample = MMD_b(K, n, m)
    mmd_u_sample = MMD_u(K, n, m)

    # mmd unbiased value of sample

    # Calculate bootstrapped mmd
    seeded_prng = np.random.seed(seed)
    # list to store null distribution
    mmd_b_null = np.zeros(B)
    mmd_u_null = np.zeros(B)
    K_i = np.empty(K.shape)

    # Bootstrapp with replacement or without that is the question
    p_b_value = 0
    p_u_value = 0
    for b in range(B):  
        #index = rng.randint(low = 0, high = len(K)-1, size = n+m)
        index = np.random.permutation(n+m)
        #K_i = K[index, index[:, None]] #None sets new axix
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]
        #    K_i[i,] = K[:, :]
        # calculate mmd under the null
        mmd_b_null[b] = MMD_b(K_i, n, m)
        mmd_u_null[b] = MMD_u(K_i, n, m)


    p_b_value =  (mmd_b_null > mmd_b_sample).sum()/float(B)
    p_u_value =  (mmd_u_null > mmd_u_sample).sum()/float(B)

    return p_b_value, p_u_value, mmd_b_null, mmd_u_null, mmd_b_sample, mmd_u_sample


@njit
def Boot_median(statistic: np.array, B:int, n:int, m:int, seed:int):

    """
    Permutate a list B times and calculate the median of index :n minus median of n:(n+m)
    """

    result = np.empty(B)

    for boot in range(B):

        index = np.random.permutation(n+m)

        tmp1 = np.zeros(n)
        tmp2 = np.zeros(m)

        for i in range(n):
            tmp1[i] = statistic[index[i]]
        for i in range(m):
            tmp2[i] = statistic[index[n+i]]

        result[boot] = np.median(tmp1) - np.median(tmp2)
        #result[boot] = np.median(statistic[index[:n]]) - np.median(statistic[index[n:(n+m)]])

    return result


# Kernel Matrix for graphs. Based on the grakel package
def KernelMatrix(graph_list: list, kernel: dict, normalize:bool):
    init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
    K = init_kernel.fit_transform(graph_list)
    return K

def GenerateBinomialGraph(n:int,nr_nodes:int,p:float, label:list = None, attributes:list = None):
    """
    :n Number of samples
    :nr_nodes number of nodes
    :label list for node labelling
    :param attributes: list for attributes
    :return: list of networkx graphs
    """
    Gs = []
    for i in range(n):
        G = nx.fast_gnp_random_graph(nr_nodes, p)
        if not label is None:
            nx.set_node_attributes(G, label, 'label')
        if not label is None:
            nx.set_node_attributes(G, attributes, 'attributes')
        Gs.append(G)

    return Gs

def generateSBM(n:int, pi:list, P:list, label:list, nr_nodes:int):
    """
    :n Number of samples
    :pi probability of belonging to block, must sum to 1
    :P Block probability matrix
    :label list for node labelling
    :return: list of networkx graphs
    """

    Gs = []
    for i in range(n):
        G = SBM.SBM(P, pi, nr_nodes)
        nx.set_node_attributes(G, label, 'label')
        Gs.append(G)

    return Gs

def generateSBM2(n:int, sizes:list, P:list, label:list):
    """
    :n Number of samples
    :sizes number of node in each block
    :P Block probability matrix
    :label list for node labelling
    :return: list of networkx graphs
    """

    Gs = []
    for i in range(n):
        G = nx.stochastic_block_model(sizes, P, )
        nx.set_node_attributes(G, label, 'label')
        Gs.append(G)

    return Gs



def CalculateGraphStatistics(Gs, n,m):
    """


    :param Gs: list of networkx graphs
    :param n: number of graphs in first sample
    :param m: number of graphs in second sample
    """



    # average degree
    avg_degree_list = np.array(list(map(lambda x: np.average(x.degree, axis = 0)[1], Gs)))
    avg_degree_sample = np.median(avg_degree_list[:n]) - np.median(avg_degree_list[n:(n+m)])
    # median degree
    median_degree_list = np.array(list(map(lambda x: np.median(x.degree, axis = 0)[1], Gs)))
    median_degree_sample = np.median(avg_degree_list[:n]) - np.median(avg_degree_list[n:(n+m)])
    # median of maximal degree
    max_degree_list = np.array(list(map(lambda x: np.max(x.degree, axis = 0)[1], Gs)))
    max_degree_sample = np.median(max_degree_list[:n]) - np.median(max_degree_list[n:(n+m)])
    # average neighbour degree
    avg_neigh_degree_list = np.array(list(map(lambda x: np.average(list(nx.average_neighbor_degree(x).values())), Gs)))
    avg_neigh_degree_sample = np.median(avg_neigh_degree_list[:n]) - np.median(avg_neigh_degree_list[n:(n+m)])
    # median of average clustering
    avg_clustering_list = np.array(list(map(lambda x: nx.average_clustering(x), Gs)))
    avg_clustering_sample = np.median(avg_clustering_list[:n]) - np.median(avg_clustering_list[n:(n+m)])
    # median of transitivity
    transitivity_list = np.array(list(map(lambda x: nx.transitivity(x), Gs)))
    transitivity_sample = np.median(transitivity_list[:n]) - np.median(transitivity_list[n:(n+m)])

    test_statistic_sample = dict()
    test_statistic_sample['avg_degree'] = avg_degree_sample
    test_statistic_sample['median_degree'] = median_degree_sample
    test_statistic_sample['max_degree'] = max_degree_sample
    test_statistic_sample['avg_neigh_degree'] = avg_neigh_degree_sample
    test_statistic_sample['avg_clustering'] = avg_clustering_sample
    test_statistic_sample['transitivity'] = transitivity_sample

    test_statistic_list = dict()
    test_statistic_list['avg_degree'] = avg_degree_list
    test_statistic_list['median_degree'] = median_degree_list
    test_statistic_list['max_degree'] = max_degree_list
    test_statistic_list['avg_neigh_degree'] = avg_neigh_degree_list
    test_statistic_list['avg_clustering'] = avg_clustering_list
    test_statistic_list['transitivity'] = transitivity_list



    return test_statistic_list, test_statistic_sample



def GenerateScaleFreeGraph(power:float, nr_nodes:int):
    """
    create a graph with degrees following a power law distribution
    https://stackoverflow.com/questions/28920824/generate-a-scale-free-network-with-a-power-law-degree-distributions
    """
#create a graph with degrees following a power law distribution

    # the outer loop makes sure that we have an even number of edges so that we can use the configuration model
    while True: 
        # s keeps the degree sequence
        s=[]
        while len(s)<nr_nodes:
            nextval = int(nx.utils.powerlaw_sequence(1, power)[0]) #100 nodes, power-law exponent 2.5
            if nextval!=0:
                s.append(nextval)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G= nx.Graph(G) # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def GenerateSamplesOfScaleFreeGraphs(n:int,nr_nodes:int,power:float, label:list = None, attributes:list = None):
    """
    :n Number of samples
    :nr_nodes: number of nodes
    :param power: power law parameter
    :param label: list for node labelling
    :param attributes: list for attributes
    :return: list of networkx graphs
    """
    Gs = []
    for i in range(n):
        G = GenerateScaleFreeGraph(power, nr_nodes)
        if not label is None:
            nx.set_node_attributes(G, label, 'label')
        if not label is None:
            nx.set_node_attributes(G, attributes, 'attributes')
        Gs.append(G)

    return Gs
