

# This script experiments with MMD for two binomial graphs



import networkx as nx
import numpy as np
import grakel as gk # graph kernels module
import matplotlib
import scipy
import string
import pickle # save data frame (results) in a .pkl file
import pandas as pd
import math
# from tqdm import * # Estimation of loop time
from tqdm import tqdm as tqdm
from datetime import datetime
import os, sys

# add perant dir to PYTHONPATH
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print(os.getcwd())

# Load module which
import MMDforGraphs as mg


def foo_():
    time.sleep(0.3)

if __name__ == "__main__":
    # Erdos Renyi graphs


    nr_nodes_1_list = [20, 40, 60, 80, 100, 150, 200]
    #nr_nodes_2 = 50
    n_list = [5, 10, 15, 20, 50, 100, 150]
    #m = 10
    center_prob = 0.05

    # Number of Bootstraps
    B = 500

    # Sample and bootstrap N times so that we can estimate the power.
    N = 500
    
    # Kernel specification
    kernel = [{"name": "WL-OA", "n_iter": 3}]

    alphas = np.linspace(0.025, 0.975, 39)

    # distance from center is the how far the two probabities are from the center_prob, 
    # if distance_from_center = 0 then the two populations are the same  
    distance_from_center = [0.0,0.005,0.01,0.015,0.02, 0.045, 0.2]
    # store time of run
    now = datetime.now()
    time = pd.Timestamp(now)

    # Where should the dataframe be saved
    path = 'data/binomial/wl_oa_10samples_p05.pkl'
    # Store outcome in a data
    df = pd.DataFrame() #pd.read_pickle("runs.pkl")
    
    for nr_nodes_1 in nr_nodes_1_list:
        nr_nodes_2 = nr_nodes_1
        print(nr_nodes_1)

        for n in n_list:
            m = n

            H0_true = True
            for p_distance in tqdm(distance_from_center):

                if p_distance == 0.0:
                    H0_true = True
                else:
                    H0_true =  False    

                total_time = datetime.now()
                
                # Set probability
                p_edge_1 = center_prob
                p_edge_2 = center_prob + p_distance

                # Set block probabilities
                        
                # Set label (all nodes have same label, just required for some kernels)
                label_1 = dict( ( (i, 'a') for i in range(nr_nodes_1) ) )
                label_2 = dict( ( (i, 'a') for i in range(nr_nodes_2) ) )

                # Generate a sample
                # print("Creating sample ....")
                start_time = datetime.now()
                        
                p_b_values = np.array([-1.0] * N)
                p_u_values = np.array([-1.0] * N)
                        
                # We also test the accept region based on uniform convergence bounds
                accept_b = np.array([-1.0] * N)
                accept_u = np.array([-1.0] * N)

                mmd_b_samples = np.array([-1.0] * N)
                mmd_u_samples = np.array([-1.0] * N)

                # p value for each test statistic
                test_statistic_p_val = {'avg_degree':[0] * N,
                                        'max_degree':[0] * N,
                                        'avg_neigh_degree':[0] * N,
                                        'avg_clustering':[0] * N,
                                        'transitivity':[0] * N}

                #with tqdm(total=N, file=sys.stdout) as pbar:
                for sample in range(N):
                
                    # sample binomial graphs
                    Gs = mg.GenerateBinomialGraph(n = n, nr_nodes = nr_nodes_1, p = p_edge_1, label = label_1)
                    G2 = mg.GenerateBinomialGraph(n = m, nr_nodes = nr_nodes_2, p = p_edge_2, label = label_2)
                    Gs.extend(G2)


                    # Calculate basic  graph statistics
                    # average degree
                    avg_degree_list = np.array(list(map(lambda x: np.average(x.degree, axis = 0)[1], Gs)))
                    avg_degree_sample = np.median(avg_degree_list[:n]) - np.median(avg_degree_list[n:(n+m)])
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

                    test_statistic = dict()

                    # Bootstrap basic graph statistics
                    test_statistic['avg_degree'] = mg.Boot_median(avg_degree_list, B, n,m, 123)
                    test_statistic['max_degree'] = mg.Boot_median(max_degree_list, B, n,m, 123)
                    test_statistic['avg_neigh_degree'] = mg.Boot_median(avg_neigh_degree_list, B, n,m, 123)
                    test_statistic['avg_clustering'] = mg.Boot_median(avg_clustering_list, B, n,m, 123)
                    test_statistic['transitivity'] = mg.Boot_median(transitivity_list, B, n,m, 123)
                    
                    # calculate the p value for the test statistic
                    test_statistic_p_val['avg_degree'][sample] = (test_statistic['avg_degree'] < avg_degree_sample).sum()/float(B)
                    test_statistic_p_val['max_degree'][sample] = (test_statistic['max_degree'] < max_degree_sample).sum()/float(B)
                    test_statistic_p_val['avg_neigh_degree'][sample] = (test_statistic['avg_neigh_degree'] < avg_neigh_degree_sample).sum()/float(B)
                    test_statistic_p_val['avg_clustering'][sample] = (test_statistic['avg_clustering'] < avg_clustering_sample).sum()/float(B)
                    test_statistic_p_val['transitivity'][sample] = (test_statistic['transitivity'] < transitivity_sample).sum()/float(B)
                    # print("Creating graph list ....")
                    graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')
                                    
                    # Fit a kernel
                    K = mg.KernelMatrix(graph_list, kernel, True)

                    # Calculate Bootstrap
                    # B number of bootstraps
                    p_b_value, p_u_value, mmd_b_null, mmd_u_null, mmd_b_sample, mmd_u_sample = mg.BootstrapPval(B = B, K = K, n = n, m = m, seed = 123)

                    p_b_values[sample] = p_b_value
                    p_u_values[sample] = p_u_value
                        
                    mmd_b_samples[sample] = mmd_b_sample
                    mmd_u_samples[sample] = mmd_u_sample


                            
                print("--- %s  Power estimation loop ---" % (datetime.now() - start_time))
                        
                # Calculate fraction of rejections

                for alpha in alphas:

                    rejections_b = (np.array(p_b_values) < alpha).sum()/float(N)                  
                    rejections_u = (np.array(p_u_values) < alpha).sum()/float(N)

                    power_avg_degree = (np.array(test_statistic_p_val['avg_degree']) < alpha).sum()/float(N)
                    power_max_degree = (np.array(test_statistic_p_val['max_degree']) < alpha).sum()/float(N)
                    power_avg_neigh_degree = (np.array(test_statistic_p_val['avg_neigh_degree']) < alpha).sum()/float(N)
                    power_avg_clustering = (np.array(test_statistic_p_val['avg_clustering']) < alpha).sum()/float(N)
                    power_transitivity = (np.array(test_statistic_p_val['transitivity']) < alpha).sum()/float(N)

                    

                    tmp_b = np.sqrt(mmd_b_samples) < (math.sqrt(2*K.max()/n))*(1 + math.sqrt(2*math.log(1/alpha)))
                    tmp_u = mmd_u_samples < (4*K.max()/math.sqrt(n))*math.sqrt(math.log(1/alpha))   

                    power_distfree_b = (float(N)-float(np.sum(tmp_b)))/float(N)
                    power_distfree_u = (float(N)-float(np.sum(tmp_u)))/float(N)

                    # Store the run information in a dataframe,
                    df = df.append({'kernel': str(kernel), 
                                    'power_permutation_b': rejections_b,
                                    'power_permutation_u': rejections_u,
                                    'power_distfree_b': power_distfree_b,
                                    'power_distfree_u': power_distfree_u,
                                    'power_avg_degree':power_avg_degree,
                                    'power_max_degree':power_max_degree,
                                    'power_avg_neigh_degree':power_avg_neigh_degree,
                                    'power_avg_clustering':power_avg_clustering,
                                    'power_transitivity':power_transitivity,
                                    'alpha':alpha,
                                    'nr_nodes_1': nr_nodes_1,
                                    'nr_nodes_2': nr_nodes_2,
                                    'p_edge_1': p_edge_1,
                                    'p_edge_2': p_edge_2,
                                    'ratio': np.round(p_edge_2/p_edge_1,3),
                                    'n':n,
                                    'm':m,
                                    'timestap':time,
                                    'H0_true':H0_true,
                                    'B':B,
                                    'N':N,
                                    'run_time':str((datetime.now() - total_time))}, ignore_index=True)

            # Save the dataframe such that if out-of-memory or time-out happen we at least have some of the information.
            start_time = datetime.now()
            with open(path, 'wb') as f:
                    pickle.dump(df, f)
            #df.to_pickle("data/run_wl_oa_10samples_different_n.pkl")
            print("--- %s  Pickle ---" % (datetime.now() - start_time))
















