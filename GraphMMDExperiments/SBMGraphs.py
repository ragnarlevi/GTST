

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
import SBM

def foo_():
    time.sleep(0.3)

if __name__ == "__main__":
    # Erdos Renyi graphs


    nr_nodes_1 = 60
    nr_nodes_2 = 60
    sizes = [20, 20 ,20]
    n = 10
    m = 10
    Block_Matrix_1 = np.array([[0.15, 0.05, 0.05],
                              [0.05, 0.15, 0.05],
                              [0.05, 0.05, 0.15]])

    Block_Matrix_2 = np.array([[0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1]])

    pi = [1/3] * 3


    # Number of Bootstraps
    B = 500

    # Sample and bootstrap N times so that we can estimate the power.
    N = 500
    
    # Kernel specification
    kernel = [{"name": "WL-OA", "n_iter": 3}]

    alphas = np.linspace(0.025, 0.975, 39)

    # We create a 
    lambas = np.linspace(0.0, 0.2, 9)
    lambas = np.concatenate((lambas, [1]))
    # store time of run
    now = datetime.now()
    time = pd.Timestamp(now)

    # Where should the dataframe be saved
    path = 'data/SBM/wl_oa_10samples.pkl'
    # Store outcome in a data
    df = pd.DataFrame() #pd.read_pickle("runs.pkl")
    
    H0_true = True
    for l in tqdm(lambas):

        if l == 0.0:
            H0_true = True
        else:
            H0_true =  False    

        total_time = datetime.now()
        
        # Set block probabilities
        p1 = np.array(Block_Matrix_1)
        p2 = (1-l)*np.array(Block_Matrix_1) + l*np.array(Block_Matrix_2)

                
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
            Gs = mg.generateSBM(n, pi, p1, label_1, nr_nodes_1)
            G2 = mg.generateSBM(m, pi, p2, label_2, nr_nodes_2)
            # Gs = mg.generateSBM2(n, sizes, p1, label_1)
            # G2 = mg.generateSBM2(m, sizes, p2, label_2)
            Gs.extend(G2)

            # Calculate basic  graph statistics
            # median  average degree
            avg_degree = np.median([ np.average(G.degree, axis=0)[1] for G in Gs[:n]]) - np.median([ np.average(G.degree, axis=0)[1] for G in Gs[n:(n+m)]])
            # median of maximal degree
            max_degree = np.median([ np.max(G.degree, axis=0)[1] for G in Gs[:n]]) - np.median([ np.max(G.degree, axis=0)[1] for G in Gs[n:(n+m)]])
            # median average neighbour degree
            avg_neigh_degree = np.median([np.average(list(nx.average_neighbor_degree(G).values())) for G in Gs[:n]]) - np.median([np.average(list(nx.average_neighbor_degree(G).values())) for G in Gs[n:(n+m)]])
            # median of average clustering
            avg_clustering = np.median([nx.average_clustering(G) for G in Gs[:n]]) - np.median([nx.average_clustering(G) for G in Gs[n:(n+m)]])
            # median of transitivity
            transitivity = np.median([nx.transitivity(G) for G in Gs[:n]]) - np.median([nx.transitivity(G) for G in Gs[n:(n+m)]])

            test_statistic = {'avg_degree':[0] * B,
                                    'max_degree':[0] * B,
                                    'avg_neigh_degree':[0] * B,
                                    'avg_clustering':[0] * B,
                                    'transitivity':[0] * B}

            # Bootstrap basic graph statistics
            rng = np.random.RandomState(123)
            # for boot in range(B):
            #     index = rng.permutation(n+m)
            #     test_statistic['avg_degree'] = np.median([ np.average(Gs[i].degree, axis=0)[1] for i in index[:n]]) - np.median([ np.average(Gs[i].degree, axis=0)[1] for i in index[n:(n+m)]])
            #     test_statistic['max_degree'] = np.median([ np.max(Gs[i].degree, axis=0)[1] for i in index[:n]]) - np.median([ np.max(Gs[i].degree, axis=0)[1] for i in index[n:(n+m)]])
                # test_statistic['avg_neigh_degree'] = np.median([np.average(list(nx.average_neighbor_degree(Gs[i]).values())) for i in index[:n]]) -\
                #      np.median([np.average(list(nx.average_neighbor_degree(Gs[i]).values())) for i in index[n:(n+m)]])
                #test_statistic['avg_clustering'] = np.median([nx.average_clustering(Gs[i]) for i in index[:n]]) - np.median([nx.average_clustering(Gs[i]) for i in index[n:(n+m)]])
                #test_statistic['transitivity'] = np.median([nx.transitivity(Gs[i]) for i in index[:n]]) - np.median([nx.transitivity(Gs[i]) for i in index[n:(n+m)]])
            
            # calculate the p value for the test statistic
            test_statistic_p_val['avg_degree'] = (test_statistic['avg_degree'] > avg_degree).sum()/float(B)
            test_statistic_p_val['max_degree'] = (test_statistic['max_degree'] > max_degree).sum()/float(B)
            test_statistic_p_val['avg_neigh_degree'] = (test_statistic['avg_neigh_degree'] > avg_neigh_degree).sum()/float(B)
            test_statistic_p_val['avg_clustering'] = (test_statistic['avg_clustering'] > avg_clustering).sum()/float(B)
            test_statistic_p_val['transitivity'] = (test_statistic['transitivity'] > transitivity).sum()/float(B)
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

            

            tmp_b = mmd_b_samples < (math.sqrt(2*K.max()/n))*(1 + math.sqrt(2*math.log(1/alpha)))
            tmp_u = mmd_b_samples < (4*K.max()/math.sqrt(n))*math.sqrt(math.log(1/alpha))   

            power_distfree_b = (N-np.sum(tmp_b))/float(N)
            power_distfree_u = (N-np.sum(tmp_u))/float(N)

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
                            'lambda': l,
                            'p1': str(p1),
                            'p2': str(p2),
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
















