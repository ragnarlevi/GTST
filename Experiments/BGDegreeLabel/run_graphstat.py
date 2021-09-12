

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
# Where to save results
parser.add_argument('-p', '--path', type=str,metavar='', help='Give path (including filename) to where the data should be saved')

# Number of Iterations specifics
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')

# Graph generation specifics
parser.add_argument('-n1', '--NrSamples1', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-n2', '--NrSamples2', type=int,metavar='', help='Number of graphs in sample 2')
parser.add_argument('-nnode1', '--NrNodes1', type=int,metavar='', help='Number of nodes in each graph in sample 1')
parser.add_argument('-nnode2', '--NrNodes2', type=int,metavar='', help='Number of nodes in each graph in sample 2')
parser.add_argument('-k1', '--AverageDegree1', type=float,metavar='', help='Average degree of each graph in sample 1')
parser.add_argument('-k2', '--AverageDegree2', type=float,metavar='', help='Average degree of each graph in sample 2')

# parallelization specifics
parser.add_argument('-d', '--division', type=int,metavar='', help='How many processes')

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_false', help = 'print verbose')

args = parser.parse_args()



if __name__ == "__main__":
    # Erdos Renyi graphs
    # np.seterr(divide='ignore', invalid='ignore')


    # NOTE IF args.variable has not been given then args.variable returns None.  

    # Number of Bootstraps
    B = args.NrBootstraps
    # Sample and bootstrap N times so that we can estimate the power.
    N = args.NrSampleIterations
    # Where should the dataframe be saved
    path = args.path

    n1 = args.NrSamples1
    n2 = args.NrSamples2
    nnode1 = args.NrNodes1
    nnode2 = args.NrNodes2
    k1 = args.AverageDegree1
    k2 = args.AverageDegree2

    # number of cores
    d = args.division



    # Initialize Graph generator class
    bg1 = mg.BinomialGraphs(n1, nnode1, k1, l = 'degreelabels', fullyConnected = True)
    bg2 = mg.BinomialGraphs(n2, nnode2, k2, l = 'degreelabels', fullyConnected = True)

    # Probability of type 1 error
    alphas = np.linspace(0.01, 0.99, 99)


    now = datetime.now()
    time = pd.Timestamp(now)
    print(time)
    
    
    # Store outcome in a data
    df = pd.DataFrame()

    # caclulate process partition
    part = int(np.floor(N/d))
    if N % d != 0:
        N = part*d
        warnings.warn(f'Number of samples not an integer multiply of number of processes. N set as the floor {N}')

    print(part)
    print(N)
    print(d)
    
    Graph_Statistics_functions = [mg.average_degree, mg.median_degree, mg.avg_neigh_degree, mg.avg_clustering, mg.transitivity]

    p_values = dict()
    for i in range(len(Graph_Statistics_functions)):
        key = Graph_Statistics_functions[i].__name__
        p_values[key] = np.array([-1.0] * N)


    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(mg.iterationGraphStat, n , Graph_Statistics_functions, bg1, bg2, B) for n in [part] * d]

        # For loop that takes the output of each process and concatenates them together
        cnt = 0
        for f in concurrent.futures.as_completed(results):
            
            for k,v in f.result().items():
                p_values[k][cnt:(cnt+part)] = v


            cnt += part


    for i in range(len(Graph_Statistics_functions)):
                        key = Graph_Statistics_functions[i].__name__
                        if np.any(p_values[key] < 0):
                            warnings.warn(f"Some p value is negative for {key}") 

    


    # Calculate ROC curve

    for alpha in alphas:
        
        # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

        # power of MMD tests (including distribution free test)
        power_graph = dict()

        for i in range(len(Graph_Statistics_functions)):
            key = Graph_Statistics_functions[i].__name__
            #print(f'{key} pvaalue {p_values[key]}')
            power_graph[key] = (np.array(p_values[key]) < alpha).sum()/float(N)



        # Store the run information in a dataframe,
        tmp = pd.DataFrame({'kernel': 'GraphStats', 
                        'alpha':alpha,
                        'normalize':0,
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
        if len(power_graph) != 0:
            for k,v in power_graph.items():
                tmp[k] = v

        # # add specific kernel value
        # for k,v in kernel_specific_params.items():
        #     if not v is None:
        #         tmp[k] = v



        # add to the main data frame
        df = df.append(tmp, ignore_index=True)

    # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
    with open(path, 'wb') as f:
            pickle.dump(df, f)

    print(datetime.now() - now )

