

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
import argparse
import warnings

# add perant dir to PYTHONPATH
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print(os.getcwd())

# Load module which
import MMDforGraphs as mg


parser = argparse.ArgumentParser()
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')
parser.add_argument('-p', '--path', type=str,metavar='', help='Give path (including filename) to where the data should be saved')
parser.add_argument('-s', '--Gstats', type=int,metavar='', help='Should graph statistics be used to test')
group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_false', help = 'print verbose')


args = parser.parse_args()


if __name__ == "__main__":
    # Erdos Renyi graphs
    # np.seterr(divide='ignore', invalid='ignore')

    if args.verbose:
        print("The number of bootstraps is " + str(args.NrBootstraps) +  "\n" + \
        "The number of sample iterations is " + str(args.NrSampleIterations) + "\n" +\
        "the path is " + str(args.path) + "\n" +\
        "Gstats: " + str(bool(args.Gstats)))


    # Number of Bootstraps
    B = args.NrBootstraps
    # Sample and bootstrap N times so that we can estimate the power.
    N = args.NrSampleIterations
    # Where should the dataframe be saved
    path = args.path
    # Should Graph statistics be calculated?
    graphStatistics = bool(args.Gstats)


    # Variables not parsed yet
    nr_nodes_same = True
    nr_samples_same = True
    nr_nodes_1_list = [100, 150, 200]#20, 30, 50, 
    #nr_nodes_2 = 50
    n_list = [5, 10, 15, 20, 50, 100]
    #m = 10
    #center_prob = 0.05
    power = 2.5
    # are nodes labelled?
    labelling = False
    # are nodes attributed?
    attributed = False
    assert not (labelling is True and attributed is True), "Graph with both labels and attributes not currently supported"
    # set seed
    seed = 123
    
    # Kernel specification
    # kernel = [{"name": "WL", "n_iter": 4}]
    # kernel = [{"name": "WL-OA", "n_iter": 5}]
    # kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "vertex_histogram"}]
    # kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "SP"}]
    # kernel = [{"name": "SP", "with_labels": True}]
    # kernel = [{"name": "svm_theta"}]
    # kernel = [{"name": "pyramid_match", "with_labels":False}]

    # kernel = [{"name": "ML", "n_samples":20}]
    kernel = [{"name": "GR", "k":4}]

    alphas = np.linspace(0.025, 0.975, 39)

    # Average degree that we add to the sample 2 on top of avg degree of sample 1
    power_add = [0.01, 0.05, 0.1, 0.15, 0.2]

    # store time of run
    now = datetime.now()
    time = pd.Timestamp(now)

    # Store outcome in a data
    df = pd.DataFrame() #pd.read_pickle("runs.pkl")
    
    for nr_nodes_1 in nr_nodes_1_list:
        if nr_nodes_same:
            nr_nodes_1_list = [nr_nodes_1]
        
        for nr_nodes_2 in nr_nodes_1_list:
            print(nr_nodes_1)
            print(nr_nodes_2)

            for n in n_list:
                if nr_samples_same:
                    m_list = [n]

                for m in m_list:
                    print(n)
                    print(m)

                    for sample_2_margin in tqdm(power_add,position=0, leave=True):
                        power_1 = power
                        power_2 = power + sample_2_margin
                        
                                
                        # Set label (all nodes have same label, just required for some kernels)
                        if labelling:
                            label_1 = dict( ( (i, 'a') for i in range(nr_nodes_1) ) )
                            label_2 = dict( ( (i, 'a') for i in range(nr_nodes_2) ) )
                        
                        if attributed:
                            attr_1 = dict( ( (i, [1]) for i in range(nr_nodes_1) ) )
                            attr_2 = dict( ( (i, [1]) for i in range(nr_nodes_2) ) )

                        # keep p values for the biased and unbiased cases       
                        p_b_values = np.array([-99.0] * N)
                        p_u_values = np.array([-99.0] * N)          

                        # keep mmd value for each sample, both biased and unbiased
                        mmd_b_samples = np.array([-99.0] * N)
                        mmd_u_samples = np.array([-99.0] * N)

                        # p value for each test statistic
                        test_statistic_p_val = {'avg_degree':[0] * N,
                                                'median_degree':[0] * N,
                                                'max_degree':[0] * N,
                                                'avg_neigh_degree':[0] * N,
                                                'avg_clustering':[0] * N,
                                                'transitivity':[0] * N}

                        for sample in range(N):
                            total_time = datetime.now()
                        
                            # sample binomial graphs
                            if attributed:
                                Gs = mg.GenerateSamplesOfScaleFreeGraphs(n = n, nr_nodes = nr_nodes_1, power = power_1, attributes=attr_1)
                                G2 = mg.GenerateSamplesOfScaleFreeGraphs(n = m, nr_nodes = nr_nodes_2, power = power_2, attributes=attr_2)
                                Gs.extend(G2)
                                graph_list = gk.graph_from_networkx(Gs, node_labels_tag='attributes')
                            elif labelling:
                                Gs = mg.GenerateSamplesOfScaleFreeGraphs(n = n, nr_nodes = nr_nodes_1, power = power_1, label = label_1)
                                G2 = mg.GenerateSamplesOfScaleFreeGraphs(n = m, nr_nodes = nr_nodes_2, power = power_2, label = label_2)
                                Gs.extend(G2)
                                graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')
                            else:
                                Gs = mg.GenerateSamplesOfScaleFreeGraphs(n = n, nr_nodes = nr_nodes_1, power = power_1)
                                G2 = mg.GenerateSamplesOfScaleFreeGraphs(n = m, nr_nodes = nr_nodes_2, power = power_2)
                                Gs.extend(G2)
                                graph_list = gk.graph_from_networkx(Gs)

                            # Calculate basic  graph statistics
                            if graphStatistics:
                                test_statistic_list, test_statistic_sample = mg.CalculateGraphStatistics(Gs, n, m)

                                # bootstrap the list of graph statistics
                                test_statistic_boot = dict()
                                test_statistic_boot['avg_degree'] = mg.Boot_median(test_statistic_list['avg_degree'], B, n, m, seed = seed)
                                test_statistic_boot['median_degree'] = mg.Boot_median(test_statistic_list['median_degree'], B, n, m, seed = seed)
                                test_statistic_boot['max_degree'] = mg.Boot_median(test_statistic_list['max_degree'], B, n, m, seed = seed)
                                test_statistic_boot['avg_neigh_degree'] = mg.Boot_median(test_statistic_list['avg_neigh_degree'], B, n, m, seed = seed)
                                test_statistic_boot['avg_clustering'] = mg.Boot_median(test_statistic_list['avg_clustering'], B, n, m, seed = seed)
                                test_statistic_boot['transitivity'] = mg.Boot_median(test_statistic_list['transitivity'], B, n, m, seed = seed)
                            
                                # calculate the p value for the test statistic
                                test_statistic_p_val['avg_degree'][sample] = (test_statistic_boot['avg_degree'] > test_statistic_sample['avg_degree']).sum()/float(B)
                                test_statistic_p_val['median_degree'][sample] = (test_statistic_boot['median_degree'] > test_statistic_sample['median_degree']).sum()/float(B)
                                test_statistic_p_val['max_degree'][sample] = (test_statistic_boot['max_degree'] > test_statistic_sample['max_degree']).sum()/float(B)
                                test_statistic_p_val['avg_neigh_degree'][sample] = (test_statistic_boot['avg_neigh_degree'] > test_statistic_sample['avg_neigh_degree']).sum()/float(B)
                                test_statistic_p_val['avg_clustering'][sample] = (test_statistic_boot['avg_clustering'] > test_statistic_sample['avg_clustering']).sum()/float(B)
                                test_statistic_p_val['transitivity'][sample] = (test_statistic_boot['transitivity'] > test_statistic_sample['transitivity']).sum()/float(B)
                            # print("Creating graph list ....")
                            
                                            
                            # Fit a kernel
                            K = mg.KernelMatrix(graph_list, kernel, False)
                            if np.all((K == 0)):
                                warnings.warn("all element in K zero")


                            # Calculate Bootstrap
                            # B number of bootstraps
                            p_b_value, p_u_value, mmd_b_null, mmd_u_null, mmd_b_sample, mmd_u_sample = mg.BootstrapPval(B = B, K = K, n = n, m = m, seed = seed)

                            p_b_values[sample] = p_b_value
                            p_u_values[sample] = p_u_value
                                
                            mmd_b_samples[sample] = mmd_b_sample
                            mmd_u_samples[sample] = mmd_u_sample
   
                        # check if any element is still -1
                        assert not np.any(p_b_values == -99.0)
                        assert not np.any(p_u_values == -99.0)
                        assert not np.any(mmd_b_samples == -99.0)
                        assert not np.any(mmd_u_samples == -99.0)

                        # Calculate ROC curve

                        for alpha in alphas:
                            
                            # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

                            # power of MMD tests
                            rejections_b = (np.array(p_b_values) < alpha).sum()/float(N)                  
                            rejections_u = (np.array(p_u_values) < alpha).sum()/float(N)

                            # power of "sufficient" statistics
                            if graphStatistics:
                                power_avg_degree = (np.array(test_statistic_p_val['avg_degree']) < alpha).sum()/float(N)
                                power_median_degree = (np.array(test_statistic_p_val['median_degree']) < alpha).sum()/float(N)
                                power_max_degree = (np.array(test_statistic_p_val['max_degree']) < alpha).sum()/float(N)
                                power_avg_neigh_degree = (np.array(test_statistic_p_val['avg_neigh_degree']) < alpha).sum()/float(N)
                                power_avg_clustering = (np.array(test_statistic_p_val['avg_clustering']) < alpha).sum()/float(N)
                                power_transitivity = (np.array(test_statistic_p_val['transitivity']) < alpha).sum()/float(N)
                            else:
                                power_avg_degree = None
                                power_median_degree = None
                                power_max_degree = None
                                power_avg_neigh_degree = None
                                power_avg_clustering = None
                                power_transitivity = None
                            
                            # Distribution free tests
                            # acceptace regions, so type II error
                            tmp_b = np.sqrt(mmd_b_samples) < np.sqrt(2.0*float(K.max())/float(n))*(1.0 + np.sqrt(2.0*np.log(1/alpha)))
                            tmp_u = mmd_u_samples < (4*K.max()/np.sqrt(float(n)))*np.sqrt(np.log(1.0/alpha))   
                            # power = 1-type II error
                            power_distfree_b = 1.0 - float(np.sum(tmp_b))/float(N)
                            power_distfree_u = 1.0 - float(np.sum(tmp_u))/float(N)

                            # Store the run information in a dataframe,
                            df = df.append({'kernel': str(kernel), 
                                            'power_permutation_b': rejections_b,
                                            'power_permutation_u': rejections_u,
                                            'power_distfree_b': power_distfree_b,
                                            'power_distfree_u': power_distfree_u,
                                            'power_avg_degree':power_avg_degree,
                                            'power_median_degree':power_median_degree,
                                            'power_max_degree':power_max_degree,
                                            'power_avg_neigh_degree':power_avg_neigh_degree,
                                            'power_avg_clustering':power_avg_clustering,
                                            'power_transitivity':power_transitivity,
                                            'alpha':alpha,
                                            'nr_nodes_1': nr_nodes_1,
                                            'nr_nodes_2': nr_nodes_2,
                                            'power_1': power_1,
                                            'power_2': power_2,
                                            'ratio_power': np.round(power_2/power_1,3),
                                            'n':n,
                                            'm':m,
                                            'timestap':time,
                                            'B':B,
                                            'N':N,
                                            'run_time':str((datetime.now() - total_time))}, ignore_index=True)

                        # Save the dataframe such that if out-of-memory or time-out happen we at least have some of the information.
                        with open(path, 'wb') as f:
                                pickle.dump(df, f)

















