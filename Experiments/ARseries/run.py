
import networkx as nx
import numpy as np
import statsmodels
import tqdm


import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import MMDforGraphs as mg
from importlib import reload  
foo = reload(mg)
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics.pairwise import euclidean_distances
import pickle

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str,metavar='', help='Give path (including filename) to where the data should be saved')

# Number of Iterations specifics
parser.add_argument('-B', '--NrBootstraps',metavar='', type=int, help='Give number of bootstraps')
parser.add_argument('-N', '--NrSampleIterations',metavar='', type=int, help='Give number of sample iterations')
parser.add_argument('-n1', '--NrSamples1', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-n2', '--NrSamples2', type=int,metavar='', help='Number of graphs in sample 2')
parser.add_argument('-ar1', '--ar1', type=float,metavar='', help='Autocorrelation 1')
parser.add_argument('-ar2', '--ar2', type=float,metavar='', help='Autocorrelation 2')
parser.add_argument('-sd1', '--sd1', type=float,metavar='', help='sd 1')
parser.add_argument('-sd2', '--sd2', type=float,metavar='', help='sd 2')

args = parser.parse_args()
# Number of Bootstraps
B = args.NrBootstraps
# Sample and bootstrap N times so that we can estimate the power.
N = args.NrSampleIterations
# Where should the dataframe be saved
path = args.path

n1 = args.NrSamples1
n2 = args.NrSamples2
ar1 = args.ar1
ar2 = args.ar2
sd1 = args.sd1
sd2 = args.sd2



verbose = True



def simpleArima(ar, var, nsamples):

    ar1 = np.array([1, -ar])
    ma1 = np.array([np.sqrt(var)])
    AR_object1 = ArmaProcess(ar1, ma1)
    return AR_object1.generate_sample(nsample=nsamples)


wild_ln = [1, 2, 5]
p_val_wild = {str(i):np.array([-1.0] * N) for i in wild_ln }


block_lengths = [2,3,4,5, 7, 10, 12, 15, 20]
p_val_block_mmdu = {str(i):np.array([-1.0] * N) for i in block_lengths }

p_val_mmdu = np.zeros(N)


if verbose:
    pbar = tqdm.tqdm(disable=(verbose is False), total= N)


for i in range(N):

    p_x = simpleArima(ar1, sd1 ** 2, n1)
    p_y = simpleArima(ar2, sd2 ** 2, n2)


    Z = np.expand_dims(np.r_[p_x, p_y], axis=1)
    D2 = euclidean_distances(Z, squared=True)
    upper = D2[np.triu_indices_from(D2, k=1)]
    kernel_width = np.median(upper, overwrite_input=True)
    bandwidth = np.sqrt(kernel_width / 2)
    kernel_width = 2 * bandwidth**2

    K = np.exp(-D2 * (1/kernel_width))

    for ln in wild_ln:
        p_val_wild[str(ln)][i],_,_ = mg.WildBootstrap(K, ln, n1, n2, B)

    MMD_functions = [mg.MMD_u]
    function_arguments=[dict(n = n1, m = n2 )]
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    for block_length in block_lengths:
        # print(block_length)
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B, method = 'NBB', boot_arg = {'n':n1, 'm':n2, 'l':block_length} )
        #print(f'{parts} {p_val_block_mmdu[str(block_length)]} {block_length}')
        p_val_block_mmdu[str(block_length)][i] = kernel_hypothesis.p_values['MMD_u']

    kernel_hypothesis.Bootstrap(K, function_arguments, B = B )
    p_val_mmdu[i] = kernel_hypothesis.p_values['MMD_u']

    if verbose:
        pbar.update()

if verbose:
    pbar.close()



 # Probability of type 1 error
alphas = np.linspace(0.001, 0.99, 999)


# Store outcome in a data
df = pd.DataFrame()
for alpha in alphas:
        
    # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

    # power of MMD tests (including distribution free test)

    power_wild_mmdu = {str(i):0 for i in wild_ln }
    power_block_mmdu = {str(i):0 for i in block_lengths }
    power_perm_mmdu = np.array([-1.0] * len(alphas))

    for ln in wild_ln:
        power_wild_mmdu[str(ln)]= (np.array(p_val_wild[str(ln)]) < alpha).sum()/float(N)

    for block_length in block_lengths:
        power_block_mmdu[str(block_length)]= (np.array(p_val_block_mmdu[str(block_length)]) < alpha).sum()/float(N)



    power_perm_mmdu = (np.array(p_val_mmdu) < alpha).sum()/float(N)


    # Store the run information in a dataframe,
    tmp = pd.DataFrame({'alpha':alpha,
                    'normalize':0,
                    'ar1':ar1,
                    'ar2':ar2,
                    'var1':sd1**2,
                    'var2':sd2**2,
                    'n':n1,
                    'm':n2,
                    'B':B,
                    'N':N}, index = [0])
    # Add power
    for block_length in block_lengths:
        tmp["MMDu_nbb_" + str(block_length)] = power_block_mmdu[str(block_length)]

    for ln in wild_ln:
        tmp["MMDu_wild_" + str(ln)] = power_wild_mmdu[str(ln)]
    tmp['MMDu'] = power_perm_mmdu


    # add to the main data frame
    df = pd.concat((df, tmp), ignore_index=True)

    # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
with open(path, 'wb') as f:
        pickle.dump(df, f)













