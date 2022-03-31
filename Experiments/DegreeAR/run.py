

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
parser.add_argument('-k', '--AverageDegree', type=float,metavar='', help='Intercept degree')
parser.add_argument('-ar1', '--ar1', type=float,metavar='', help='Autocorrelation 1')
parser.add_argument('-ar2', '--ar2', type=float,metavar='', help='Autocorrelation 2')


# parallelization specifics
parser.add_argument('-d', '--division', type=int,metavar='', help='How many processes')

# Kernel specifics
parser.add_argument('-kernel', '--kernel', type=str,metavar='', help='Kernel')
parser.add_argument('-norm', '--normalize', type=int,metavar='', help='Should kernel be normalized')

# Shared parameters
parser.add_argument('-nitr', '--NumberIterations', type=int,metavar='', help='WL nr iterations, wl, wloa, wwl, dk')
parser.add_argument('-wlab', '--wlab', type=int,metavar='', help='With labels?, sp, rw, pyramid')
parser.add_argument('-type', '--type', type=str,metavar='', help='Type of... rw (geometric or exponential) , deepkernel (sp or wl)')
parser.add_argument('-l', '--discount', type=float,metavar='', help='RW, wwl lambda/discount')
parser.add_argument('-tmax', '--tmax', type=int,metavar='', help='Maximum number of walks, used in propagation and RW.')

# pyramid only
parser.add_argument('-L', '--histogramlevel', type=int,metavar='', help='Pyramid histogram level.')
parser.add_argument('-dim', '--dim', type=int,metavar='', help='The dimension of the hypercube.')

# Propagation only
parser.add_argument('-w', '--binwidth', type=float,metavar='', help='Bin width.')
parser.add_argument('-M', '--Distance', type=str,metavar='', help='The preserved distance metric (on local sensitive hashing):')

# ODD only
parser.add_argument('-dagh', '--DAGHeight', type=int,metavar='', help='Maximum (single) dag height. If None there is no restriction.')

# WWL only
parser.add_argument('-sk', '--sinkhorn', type=int,metavar='', help='sinkhorn?')

# RW only
parser.add_argument('-rwApprox', '--rwApprox', type=int, metavar='', help='Number of eigenvalues, rw approximation?')
parser.add_argument('-row_norm', '--row_norm', type=int, metavar='', help='row normalize?')
parser.add_argument('-adj_norm', '--adj_norm', type=int, metavar='', help='Normalize adjacency matrix?')

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
    # Should the kernels be normalized?
    normalize = args.normalize

    n1 = args.NrSamples1
    n2 = args.NrSamples2
    nnode1 = args.NrNodes1
    nnode2 = args.NrNodes2
    k_degree = args.AverageDegree
    ar1 = args.ar1
    ar2 = args.ar2


    # number of cores
    d = args.division
    print(d)
    kernel_name = args.kernel
    # add parameters parsed, may be none
    kernel_specific_params = dict()

    # WL iterations
    kernel_specific_params['nitr'] = args.NumberIterations   
    
    # with labels?
    if args.wlab is None:
        kernel_specific_params['with_labels'] = bool(1)
    else:
        kernel_specific_params['with_labels'] = bool(args.wlab)

    kernel_specific_params['L'] = args.histogramlevel
    kernel_specific_params['dim'] = args.dim

    kernel_specific_params['w'] = args.binwidth
    kernel_specific_params['tmax'] = args.tmax
    kernel_specific_params['M'] = args.Distance

    kernel_specific_params['type'] = args.type  
    kernel_specific_params['discount'] = args.discount   

    kernel_specific_params['dagh'] = args.DAGHeight
    kernel_specific_params['sinkhorn'] = bool(args.sinkhorn)

    # rw approximation
    kernel_specific_params['r'] = args.rwApprox
    kernel_specific_params['normalize_adj'] = args.adj_norm
    kernel_specific_params['row_normalize_adj'] = args.row_norm


    


    # Initialize Graph generator class
    bg1 = mg.TemporalBinomialGraphs(n = n1, nnode = nnode1, k = k_degree, ar = ar1, l = 'degreelabels', fullyConnected = True)
    bg2 = mg.TemporalBinomialGraphs(n = n2, nnode = nnode2, k = k_degree, ar = ar2, l = 'degreelabels', fullyConnected = True)


    # Probability of type 1 error
    alphas = np.linspace(0.001, 0.99, 999)


    now = datetime.now()
    time = pd.Timestamp(now)
    print(time)
    
    # Kernel specification
    # kernel = [{"name": "WL", "n_iter": 4}]

    if kernel_name == 'wl':
        kernel = [{"name": "weisfeiler_lehman", "n_iter": kernel_specific_params['nitr']}, {"name": "vertex_histogram"}]
    elif kernel_name == 'sp':
        kernel = [{"name": "SP", "with_labels": kernel_specific_params.get('with_labels', True)}]
    elif kernel_name == 'pyramid':
        kernel = [{"name": "pyramid_match", "with_labels":kernel_specific_params.get('with_labels', True), "L":kernel_specific_params['L'], "d":kernel_specific_params['dim']}]
    elif kernel_name == 'prop':
        kernel = [{"name": "propagation", "t_max": kernel_specific_params['tmax'], "w":kernel_specific_params['w'], "M":kernel_specific_params['M']}]
    elif kernel_name == 'wloa':
        kernel = [{"name": "WL-OA", "n_iter": kernel_specific_params['nitr']}]
    elif kernel_name == 'vh':
        # vertex histogram
        kernel = [{"name": "vertex_histogram"}]
    elif kernel_name == 'rw':
        # if we are performing k-step random walk, we need the discount factor
        if kernel_specific_params.get('tmax', None) is not None:
            nr_rw_steps = kernel_specific_params.get('tmax')
            mu_vec = np.power(kernel_specific_params['discount'], range(nr_rw_steps+1)) / np.array([np.math.factorial(i) for i in np.arange(nr_rw_steps+1)])
        else:
            mu_vec = None


        kernel = {"calc_type": kernel_specific_params['type'], 
                 "c":kernel_specific_params['discount'] , 
                    "r":kernel_specific_params['r'],
                    'k':kernel_specific_params.get('tmax', None),
                    "mu_vec":mu_vec,
                    "with_labels":kernel_specific_params.get('with_labels', False),
                    "normalize_adj": kernel_specific_params['normalize_adj'],
                    "row_normalize_adj": kernel_specific_params['row_normalize_adj'],
                    }
    elif kernel_name == 'odd':
        kernel = [{"name":'odd_sth', 'h':kernel_specific_params['dagh']}]
    elif kernel_name == 'dk':
        kernel = {"type":kernel_specific_params['type'], 'wl_it':kernel_specific_params.get('nitr', 4),'normalize':normalize}
    elif kernel_name == 'wwl':
        kernel = {'discount':kernel_specific_params['discount'],'h':kernel_specific_params['nitr'], 'sinkhorn':kernel_specific_params['sinkhorn'],'normalize':normalize }
    else:
        raise ValueError(f'No kernel names {kernel_name}')
    
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

    # Store K and K max for acceptance region
    Kmax = np.array([0] * N)
    Ks = np.zeros((N, bg1.n + bg2.n, bg1.n + bg2.n))

    if kernel_name == 'dk':
        kernel_library = "deepkernel"
    elif kernel_name == 'wwl':
        kernel_library = 'wwl'
    elif kernel_name == 'rw':
        kernel_library = 'randomwalk'
    else:
        kernel_library = "Grakel"

    # Kernel Calculation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(mg.iteration_K, n , kernel, normalize, bg1,bg2, kernel_library) for n in [part] * d]

        # For loop that takes the output of each process and concatenates them together
        cnt = 0
        for f in concurrent.futures.as_completed(results):
            
            for k, v in f.result().items():
                if k == "Kmax":
                    Kmax[cnt:(cnt+part)] = v
                elif k == 'K':
                    Ks[cnt:(cnt+part)] = v



            cnt += part

    # Store p values
    block_lengths = [2,3,4,5, 7, 10, 12, 15, 20]
    p_val_block_mmdu = {str(i):np.array([-1.0] * N) for i in block_lengths }
    p_val_block_mmdb = {str(i):np.array([-1.0] * N) for i in block_lengths }
    p_val_perm_mmdu = np.array([-1.0] * N)
    p_val_perm_mmdb = np.array([-1.0] * N)

    mmd_samples_mmdu = np.array([-1.0] * N)
    mmd_samples_mmdb = np.array([-1.0] * N)
    

    
    print("Bootstrap--------")
    # # Using the Kernel we bootstrap
    # def testfunction(parts, kernel_hypothesis, n,m, block_lengths, Ks, B):
    #     print(parts)


    
    #     p_val_block_mmdu = {str(i):np.array([-1.0] * len(parts)) for i in block_lengths }
    #     p_val_block_mmdb = {str(i):np.array([-1.0] * len(parts)) for i in block_lengths }
    #     p_val_perm_mmdu = np.array([-1.0] * len(parts))
    #     p_val_perm_mmdb = np.array([-1.0] * len(parts))

    #     mmd_samples_mmdu = np.array([-1.0] * len(parts))
    #     mmd_samples_mmdb = np.array([-1.0] * len(parts))


    #     function_arguments=[dict(n = n, m = m ), dict(n = n, m = n )]
    #     for sample in parts:
    #         print(f'{sample} {Ks[sample].shape}')
    #         # block bootstrap
    #         for block_length in block_lengths:
    #             print(block_length)
    #             kernel_hypothesis.Bootstrap(Ks[sample], function_arguments, B = B, method = 'NBB', boot_arg = {'n':n, 'm':m, 'l':block_length} )
    #             p_val_block_mmdu[str(block_length)][sample] = kernel_hypothesis.p_values['MMD_u']
    #             p_val_block_mmdb[str(block_length)][sample] = kernel_hypothesis.p_values['MMD_b']
    #         # Permutation
    #         kernel_hypothesis.Bootstrap(Ks[sample], function_arguments, B = B )
    #         p_val_perm_mmdu[sample] = kernel_hypothesis.p_values['MMD_u']
    #         p_val_perm_mmdb[sample] = kernel_hypothesis.p_values['MMD_b']

    #         mmd_samples_mmdu[sample] = kernel_hypothesis.sample_test_statistic['MMD_u']
    #         mmd_samples_mmdb[sample] = kernel_hypothesis.sample_test_statistic['MMD_b']

        
    #     return dict(p_val_block_mmdu = p_val_block_mmdu, p_val_block_mmdb = p_val_block_mmdb,p_val_perm_mmdu=p_val_perm_mmdu,p_val_perm_mmdb=p_val_perm_mmdb,mmd_samples_mmdu=mmd_samples_mmdu,mmd_samples_mmdb = mmd_samples_mmdb)



    # functions used for kernel testing
    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)

    print(np.cumsum([part] * d))
    # P value Calculation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #print("here")
        results = [executor.submit(mg.testfunction, np.arange(n-part, n), kernel_hypothesis, bg1.n, bg2.n, block_lengths, Ks, B) for n in np.cumsum([part] * d)]
        #print(results)

        # For loop that takes the output of each process and concatenates them together
        cnt = 0
        for f in concurrent.futures.as_completed(results):
            # print(f.result())
            
            for k, v in f.result().items():
                #print("results")
                #print(k)
                #print(v)
                if k == "p_val_block_mmdu":
                    for block_length in block_lengths:
                        p_val_block_mmdu[str(block_length)][cnt:(cnt+part)] = v[str(block_length)]
                elif k == "p_val_block_mmdb":
                    for block_length in block_lengths:
                        p_val_block_mmdu[str(block_length)][cnt:(cnt+part)] = v[str(block_length)]
                elif k == "p_val_perm_mmdu":
                    p_val_perm_mmdb[cnt:(cnt+part)] = v
                elif k == "p_val_perm_mmdb":
                    p_val_perm_mmdb[cnt:(cnt+part)] = v
                elif k == "mmd_samples_mmdu":
                    mmd_samples_mmdu[cnt:(cnt+part)] = v
                elif k == "mmd_samples_mmdb":
                    mmd_samples_mmdb[cnt:(cnt+part)] = v

            cnt += part
     
    


    # Calculate ROC curve
    print("ROC --------")

    for alpha in alphas:
        
        # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

        # power of MMD tests (including distribution free test)
        power_block_mmdu = {str(i):0 for i in block_lengths }
        power_block_mmdb = {str(i):0 for i in block_lengths }
        power_perm_mmdu = np.array([-1.0] * len(alphas))
        power_perm_mmdb = np.array([-1.0] * len(alphas))

        for block_length in [2,3,4,5, 7, 10, 12, 15, 20]:
            power_block_mmdu[str(block_length)]= (np.array(p_val_block_mmdu[str(block_length)]) < alpha).sum()/float(N)
            power_block_mmdb[str(block_length)]= (np.array(p_val_block_mmdb[str(block_length)]) < alpha).sum()/float(N)



        power_perm_mmdu = (np.array(p_val_perm_mmdu) < alpha).sum()/float(N)
        power_perm_mmdb = (np.array(p_val_perm_mmdb) < alpha).sum()/float(N)

        tmp = mmd_samples_mmdu < (4*Kmax/np.sqrt(float(n1)))*np.sqrt(np.log(1.0/alpha))
        power_mmdu_distfree = 1.0 - float(np.sum(tmp))/float(N)

        tmp = np.sqrt(mmd_samples_mmdb) < np.sqrt(2.0*Kmax/float(n1))*(1.0 + np.sqrt(2.0*np.log(1/alpha)))
        power_mmdb_distfree =1.0 - float(np.sum(tmp))/float(N)



        # Store the run information in a dataframe,
        tmp = pd.DataFrame({'kernel': str(kernel), 
                        'alpha':alpha,
                        'normalize':normalize,
                        'nr_nodes_1': nnode1,
                        'nr_nodes_2': nnode2,
                        'k':k,
                        'ar1':ar1,
                        'ar2':ar2,
                        'var1':0.25**2,
                        'var2':0.25**2,
                        'n':n1,
                        'm':n2,
                        'timestap':time,
                        'B':B,
                        'N':N,
                        'run_time':str((datetime.now() - now))}, index = [0])
        # Add power
        for block_length in block_lengths:
            tmp["MMDu_nbb_" + str(block_length)] = power_block_mmdu[str(block_length)]
            
        tmp['MMDu'] = power_perm_mmdu
        tmp['MMDb'] = power_perm_mmdb
        tmp['MMDu_distfree'] = power_mmdu_distfree
        tmp['MMDb_distfree'] = power_mmdb_distfree

        # add specific kernel value
        for k,v in kernel_specific_params.items():
            if not v is None:
                tmp[k] = v

        # add to the main data frame
        df = pd.concat((df, tmp), ignore_index=True)

    # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
    with open(path, 'wb') as f:
            pickle.dump(df, f)

    print(datetime.now() - now )

