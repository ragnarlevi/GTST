
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap
import seaborn as sns
import string
import pickle # save data frame (results) in a .pkl file
import pandas as pd
from datetime import datetime
import os, sys
import re

import grakel as gk


sys.path.insert(0, 'C:/Users/User/Code/MMDGraph')
sys.path.insert(0, 'C:/Users/User/Code/MMDGraph/myKernels')
#from importlib import reload
#reload(readfoldertopanda)
import mmdutils
import importlib
importlib.reload(sys.modules['mmdutils'])
from mmdutils import readfoldertopanda, PlotROCGeneral, plotVaryingBGDEG, findAUC
import MMDforGraphs as mg
importlib.reload(sys.modules['MMDforGraphs'])
from myKernels import RandomWalk as rw
import WL
from multiprocessing import Pool, freeze_support



import tqdm
import warnings
warnings.filterwarnings("ignore")







n1 = 100
nnode1 = 20
k1 = 5

n2 = 100
nnode2 = 20
k2 = 6






def run_samples_threshold(N, B, n1, nnode1, k1, n2, nnode2, k2):
    import myKernels.RandomWalk as rw
    test_info = pd.DataFrame()
    thresholds = np.linspace(0.8, 1, 20)
    for sample in tqdm.tqdm(range(N)):

        g1 = mg.BinomialGraphs(n1, nnode1, k1, fullyConnected = True, e = 'random_edge_weights', ul = 0.8, uu =0.2)
        g2 = mg.BinomialGraphs(n2, nnode2, k2, fullyConnected = True, e = 'random_edge_weights', ul = 0.8, uu =0.2)
        g1.Generate()
        g2.Generate()
        Gs = g1.Gs + g2.Gs
        for dist_from_disconnection_point in [-1, 0,1,2,3, 20]:
            new_Gs = []
            isconnected = []

            for i in range(len(Gs)):
                A = nx.attr_matrix(Gs[i], edge_attr= 'weight')[0]
                G = mg.gen_fullyconnected_threshold(A, thresholds=thresholds, dist_from_disconnection_point= dist_from_disconnection_point)
                isconnected.append(nx.is_connected(G))

                new_Gs.append(G.copy())


            rw_kernel = rw.RandomWalk(new_Gs, c = 0.01, normalize=0)
            K = rw_kernel.fit_ARKU_plus(r = 6, normalize_adj=False,   edge_attr= None, verbose=False)

            MMD_functions = [mg.MMD_b, mg.MMD_u]

            kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
            function_arguments=[dict(n = g1.n, m = g2.n ), dict(n = g1.n, m = g2.n )]
            kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
            #print(f'p_value {kernel_hypothesis.p_values}')
            #print(f"MMD_u {kernel_hypothesis.sample_test_statistic['MMD_u']}")

            test_info = pd.concat((test_info, pd.DataFrame({
                'p_val':kernel_hypothesis.p_values['MMD_u'],
                'dist_from_critical': dist_from_disconnection_point,
                'sample':sample


            }, index = [0])), ignore_index=True)



    return test_info




if __name__ == '__main__':

    N = 250
    B = 5000
    with Pool() as pool:
        L = pool.starmap(run_samples_threshold, [(N, B, n1, nnode1, k1, n2, nnode2, k2), (N, B, n1, nnode1, k1, n2, nnode2, k2)])
        

        df = pd.concat(L)

    print(df)


    with open('data/threshold_data/binom65.pkl', 'wb') as f:
        pickle.dump(df, f)
