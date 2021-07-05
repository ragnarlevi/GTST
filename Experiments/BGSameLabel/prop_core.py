

# This script experiments with MMD for two binomial graphs



from logging import warn, warning
import networkx as nx
import numpy as np
import grakel as gk # graph kernels module
import pickle # save data frame (results) in a .pkl file
import pandas as pd
# from tqdm import * # Estimation of loop time
#from tqdm import tqdm as tqdm
from datetime import datetime
import os, sys
import argparse
import warnings
import concurrent.futures

# add perant dir 
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
# sys.path.append(os.path.abspath(".."))
print(os.getcwd())

# Load module which
import MMDforGraphs as mg


parser = argparse.ArgumentParser()
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')
parser.add_argument('-p', '--path', type=str,metavar='', help='Give path (including filename) to where the data should be saved')
parser.add_argument('-s', '--Gstats', type=int,metavar='', help='Should graph statistics be used to test')
parser.add_argument('-norm', '--normalize', type=int,metavar='', help='Should kernel be normalized')
parser.add_argument('-t_max', '--WalkIter', type=int,metavar='', help='Maximum number of iterations.')
parser.add_argument('-W', '--W', type=float,metavar='', help='BinWidth.')
parser.add_argument('-M', '--M', type=str,metavar='', help='The preserved distance metric (on local sensitive hashing).')
parser.add_argument('-mc', '--MinCore', type=int,metavar='', help='Core numbers bigger than min_core will only be considered.')
parser.add_argument('-n1', '--NrSamples1', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-n2', '--NrSamples2', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-nnode1', '--NrNodes1', type=int,metavar='', help='Number of nodes in each graph in sample 1')
parser.add_argument('-nnode2', '--NrNodes2', type=int,metavar='', help='Number of nodes in each graph in sample 2')
parser.add_argument('-k1', '--AverageDegree1', type=float,metavar='', help='Average degree of each graph in sample 1')
parser.add_argument('-k2', '--AverageDegree2', type=float,metavar='', help='Average degree of each graph in sample 2')
parser.add_argument('-d', '--division', type=int,metavar='', help='How many processes')


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
    # Should the kernels be normalized?
    normalize = args.normalize

    n1 = args.NrSamples1
    n2 = args.NrSamples2
    nnode1 = args.NrNodes1
    nnode2 = args.NrNodes2
    k1 = args.AverageDegree1
    k2 = args.AverageDegree2
    d = args.division
    t_max = args.WalkIter
    W = args.W
    M = args.M
    min_core = args.MinCore



    # which graph statistics functions should we test?
    Graph_Statistics_functions = [mg.average_degree, mg.median_degree, mg.avg_neigh_degree, mg.avg_clustering, mg.transitivity]

    # functions used for kernel testing
    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)

    # Initialize Graph generator class
    bg1 = mg.BinomialGraphs(n1, nnode1, k1, l = 'samelabels')
    bg2 = mg.BinomialGraphs(n2, nnode2, k2, l = 'samelabels')

    # Probability of type 1 error
    alphas = np.linspace(0.01, 0.99, 99)

    # set seed
    seed = 42
    
    now = datetime.now()
    time = pd.Timestamp(now)
    
    # Kernel specification
    #kernel = [{"name": "RW", "kernel_type": kt, 'p':nr_steps, 'with_labels':False, 'lambda':discount}]
    kernel = [{"name": "core_framework", "min_core": min_core}, {"name":"propagation", 't_max':t_max, 'w':W, 'M':M}]
    # kernel = [{"name": "weisfeiler_lehman", "n_iter": n_itr}, {"name": "vertex_histogram"}]
    #kernel = [{"name": "weisfeiler_lehman", "n_iter": n_itr}, {"name": "SP", "with_labels": True}]
    # kernel = [{"name": "SP", "with_labels": True}]
    # kernel = [{"name": "svm_theta"}]
    # kernel = [{"name": "pyramid_match", "with_labels":False}]
    # kernel = [{"name": "ML", "n_samples":20}]
    
    # Store outcome in a data
    df = pd.DataFrame()


    # caclulate process partition
    part = int(np.floor(N/d))
    if N % d != 0:
        N = part*d
        warnings.warn(f"Number of samples not an integer multiply of number of processes. N set as the floor {N}")

    p_values = dict()
    mmd_samples = dict()
    for i in range(len(MMD_functions)):
        key = MMD_functions[i].__name__
        p_values[key] = np.array([-1.0] * N)
        mmd_samples[key] = np.array([-1.0] * N)

    # Store the p-values for each test statistic for each test iteration
    test_statistic_p_val = dict()
    for i in range(len(Graph_Statistics_functions)):
        key = Graph_Statistics_functions[i].__name__
        test_statistic_p_val[key] = [0] * N

    # Store K max for acceptance region
    Kmax = np.array([0] * N)


    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(mg.iteration, n , kernel, normalize, graphStatistics, MMD_functions, Graph_Statistics_functions, bg1,bg2, B, kernel_hypothesis) for n in [part] * d]

        # For loop that takes the output of each process and concatenates them together
        cnt = 0
        for f in concurrent.futures.as_completed(results):
            
            for k,v in f.result().items():
                if k == "Kmax":
                    Kmax[cnt:(cnt+part)] = v
                elif k == 'p_values':
                    for i in range(len(MMD_functions)):
                        key = MMD_functions[i].__name__
                        p_values[key][cnt:(cnt+part)] = v[key]
                elif k == 'mmd_samples':
                    for i in range(len(MMD_functions)):
                        key = MMD_functions[i].__name__
                        mmd_samples[key][cnt:(cnt+part)] = v[key]
                elif k == 'test_statistic_p_val':
                    for i in range(len(Graph_Statistics_functions)):
                        key = Graph_Statistics_functions[i].__name__
                        test_statistic_p_val[key][cnt:(cnt+part)] = v[key]

            cnt += part

    for i in range(len(MMD_functions)):
                        key = MMD_functions[i].__name__
                        if np.any(p_values[key] < 0):
                            warnings.warn(f"Some p value is negative for {key}") 

    


    # Calculate ROC curve

    for alpha in alphas:
        
        # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

        # power of MMD tests (including distribution free test)
        power_mmd = dict()

        for i in range(len(MMD_functions)):
            key = MMD_functions[i].__name__
            power_mmd[key] = (np.array(p_values[key]) < alpha).sum()/float(N)
            if key == 'MMD_u':
                tmp = mmd_samples[key] < (4*Kmax/np.sqrt(float(n1)))*np.sqrt(np.log(1.0/alpha))
                power_mmd[key + "_distfree"] = 1.0 - float(np.sum(tmp))/float(N)
            if key == 'MMD_b':
                tmp = np.sqrt(mmd_samples[key]) < np.sqrt(2.0*Kmax/float(n1))*(1.0 + np.sqrt(2.0*np.log(1/alpha)))
                power_mmd[key + "_distfree"] = 1.0 - float(np.sum(tmp))/float(N)

        # power of "sufficient" statistics
        power_graph_statistics = dict()
        if graphStatistics:
            for i in range(len(Graph_Statistics_functions)):
                key = Graph_Statistics_functions[i].__name__
                power_graph_statistics[key] = (np.array(test_statistic_p_val[key]) < alpha).sum()/float(N)

        # Store the run information in a dataframe,
        tmp = pd.DataFrame({'kernel': str(kernel), 
                        'alpha':alpha,
                        'nr_nodes_1': nnode1,
                        'nr_nodes_2': nnode2,
                        'p_edge_1': k1/float(nnode1-1),
                        'p_edge_2': k2/float(nnode2-1),
                        'degree_1': k1 ,
                        'degree_2': k2,
                        'ratio_p': np.round(k1/float(nnode1-1)/k2/float(nnode2-1),3),
                        'ratio_degree': np.round(k2/k1,3),
                        'n':n1,
                        'm':n2,
                        'timestap':time,
                        'B':B,
                        'N':N,
                        'run_time':str((datetime.now() - now))}, index = [0])
        # Add power
        if len(power_graph_statistics) != 0:
            for k,v in power_graph_statistics.items():
                tmp[k] = v
        if len(power_mmd) != 0:
            for k,v in power_mmd.items():
                tmp[k] = v

        # add to the main data frame
        df = df.append(tmp, ignore_index=True)

    # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
    with open(path, 'wb') as f:
            pickle.dump(df, f)

    #print(datetime.now() - now )















