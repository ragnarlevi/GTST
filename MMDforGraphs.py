
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


# def ConfusionCase(threshold:float, p_val:float, H0_true: bool):
#     """
#     Return the True negative = TN, True positive = TP, False negative = FN or False Positive = FP
#     """
#     if H0_true:
#         if p_val < threshold:
#             case = "FN"
#         else:
#             case = "TP"
#     if H1_true:
#         if p_val < threshold:
#             case = "TN"
#         else:
#             case = "FP"

#     return case



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
    Permutate a list B times and calculate the median of index :n minus median of n:(n+n)
    """
    """
    Permutate a list B times and calculate the median of index :n minus median of n:(n+n)
    """
    seeded_prng = np.random.seed(seed)
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

def GenerateBinomialGraph(n:int,nr_nodes:int,p:float, label: list):
    """
    :n Number of samples
    :nr_nodes number of nodes
    :label list for node labelling
    :return: list of networkx graphs
    """
    Gs = []
    for i in range(n):
        G = nx.fast_gnp_random_graph(nr_nodes, p)
        nx.set_node_attributes(G, label, 'label')
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

def GraphTwoSample(n:int, m:int, type1: str, type2: str, **kwargs):
    """
    Generate a list of two samples of Random Graphs. First n1 graphs belong to sample 1 and the next n2 graphs belong to sample 2

    :n number of samples from population 1
    :m number of samples from population 2
    :type1 Population type 1
    :Type2 Population type 2
    :**kwargs extra arguments for population distributions
    :return: list of networkx graphs
    """

    # TODO make sperate function for if statements 
    # TODO Find better naming method for kwargs

    if str.lower(type1) == "binomial":
        G1 = GenerateBinomialGraph(n = n, nr_nodes = kwargs["nr_nodes_1"], p = kwargs["p_edge_1"], label = kwargs["label_1"])
    if str.lower(type2) == "binomial":
        G2 = GenerateBinomialGraph(n = m, nr_nodes = kwargs["nr_nodes_2"], p = kwargs["p_edge_2"], label = kwargs["label_2"])
    
    if str.lower(type1) == "sbm":
        G1 = generateSBM(n,kwargs["sizes_1"], kwargs["P_1"], kwargs["label_1"])
    if str.lower(type2) == "sbm":
        G2 = generateSBM(m,kwargs["sizes_2"], kwargs["P_2"], kwargs["label_2"])
   
    G1.extend(G2)

    return G1






if __name__ == "__main__":
    print(os.getcwd())

    # Erdos Renyi graphs

    base_node = 80

    nr_nodes = [10, 25, 50, 75, 100, 200, 500]# 1000, 2000] # [1] [1,3,5,7, 10, 15,20, 50] #[10, 25, 50, 75, 100, 200, 500, 1000, 2000]
    nr_samples = [10] # [10, 25, 50]#, 75, 100, 150]
    center_prob = 0.05
    # sizes = [30, 30, 40]
    #P = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]

    # distance from center is the how far the two probabities are from the center_prob, 
    # if distance_from_center = 0 then the two populations are the same  
    distance_from_center = [0.0,0.005,0.01,0.015,0.02, 0.045]

    # alpha p-value rejection criterion
    alpha = 0.05

    # store time of run
    now = datetime.now()
    time = pd.Timestamp(now)

    # Store outcome in a data
    df = pd.DataFrame() #pd.read_pickle("runs.pkl")

    for n in nr_samples:

        for nr_node in tqdm(nr_nodes):

            for p_distance in distance_from_center:

                total_time = datetime.now()
                # print("Number of samples: " + str(n))
                
                # Set number of nodes for this instance
                nr_nodes_1 = nr_node # base_node
                nr_nodes_2 = nr_node #base_node + nr_node

                # Set number of samples for this instance, assume same sample size
                # n = n
                m = n

                # Set probability
                p_edge_1 = center_prob + p_distance
                p_edge_2 = center_prob - p_distance

                # Set block probabilities
                # P_2 = P
                # P_2[2][2] = P_2[2][2] + p_distance
                
                # Set label (all nodes have same label, just required for some kernels)
                label_1 = dict( ( (i, 'a') for i in range(np.sum(nr_nodes)) ) )
                label_2 = dict( ( (i, 'a') for i in range(np.sum(nr_nodes)) ) )

                # Generate a sample
                # print("Creating sample ....")
                start_time = datetime.now()
                
                # Number of Bootstraps
                B = 500

                # Sample and bootstrap N times so that we can estimate the power.
                N = 500
                p_b_values = [1.0] * N
                p_u_values = [1.0] * N
                
                # We also test the accept region based on uniform convergence bounds
                accept_b = [1.0] * N
                accept_u = [1.0] * N

                mmd_b_samples = [1.0] * N
                mmd_u_samples = [1.0] * N
                for sample in range(N):


                    Gs = GraphTwoSample(n = n, m = m, type1 ="binomial", type2 = "binomial", 
                                        nr_nodes_1 = nr_nodes_1, 
                                        nr_nodes_2 = nr_nodes_2, 
                                        p_edge_1 = p_edge_1 , 
                                        p_edge_2 = p_edge_2, 
                                        label_1 =label_1, 
                                        label_2 = label_2)

                
                    # Gs =  GraphTwoSample(n = n, m = m, type1 ="sbm", type2 = "sbm",
                    #                     sizes_1 = sizes,
                    #                     P_1 = P,
                    #                     sizes_2 = sizes,
                    #                     P_2 = P_2,
                    #                     label_1 =label_1, 
                    #                     label_2 = label_2)
                

                    # print("Creating graph list ....")
                    graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')
                            
                    # Fit a kernel
                    kernel = [{"name": "WL-OA", "n_iter": 3}] 
                    K = KernelMatrix(graph_list, kernel, True)

                    # Calculate Bootstrap
                    # B number of bootstraps
                    p_b_value, p_u_value, mmd_b_null, mmd_u_null, mmd_b_sample, mmd_u_sample = BootstrapPval(B = B, K = K, n = n, m = m, seed = 123)

                    p_b_values[sample] = p_b_value
                    p_u_values[sample] = p_u_value
                    
                    accept_b[sample] = mmd_b_sample < (math.sqrt(2*K.max()/n))*(1 + math.sqrt(2*math.log(1/alpha)))
                    accept_u[sample] = mmd_u_sample < (4*K.max()/math.sqrt(n))*math.sqrt(math.log(1/alpha))

                    mmd_b_samples[sample] = mmd_b_sample
                    mmd_u_samples[sample] = mmd_u_sample
                
                print("--- %s  Power estimation loop ---" % (datetime.now() - start_time))
                
                # Calculate fraction of rejections
                rejections_b = (np.array(p_b_values) < alpha).sum()/float(N)                  
                rejections_u = (np.array(p_u_values) < alpha).sum()/float(N)   

                rejection_statistic_b = (N-np.sum(accept_b))/float(N)
                rejection_statistic_u = (N-np.sum(accept_u))/float(N)

                # Store the run information in a dataframe,
                df = df.append({'kernel': 'shortest_path', 
                                'rejections_b': rejections_b,
                                'rejections_u': rejections_u,
                                'rejections_statistic_b': rejection_statistic_b,
                                'rejections_statistic_u': rejection_statistic_u,
                                'nr_nodes_1': nr_nodes_1,
                                'nr_nodes_2': nr_nodes_2,
                                'p_edge_1': p_edge_1,
                                'p_edge_2': p_edge_2,
                                'n':n,
                                'm':m,
                                'timestap':time,
                                'run_info':"B:" + str(B)+ ",N:" + str(N) + ",wl_itr:3",
                                'run_time':str((datetime.now() - total_time))}, 
                                ignore_index=True)

    	        # Save the dataframe such that if out-of-memory or time-out happen we at least have some of the information.
                start_time = datetime.now()
                with open('data/run_wl_oa_10samples_p05.pkl', 'wb') as f:
                    pickle.dump(df, f)
                #df.to_pickle("data/run_wl_oa_10samples_different_n.pkl")
                print("--- %s  Pickle ---" % (datetime.now() - start_time))







