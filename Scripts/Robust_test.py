
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







n = 100
n_1 = 95
n_1_outlier = 5
n_2 = 100
nnode_1 = 50
nnode_2 = 50






def run_samples(N, B, Q):
    n = 100
    n_1 = 95
    n_1_outlier = 5
    n_2 = 100
    nnode_1 = 50
    nnode_2 = 50
    import myKernels.RandomWalk as rw
    test_info = pd.DataFrame()
    for sample in tqdm.tqdm(range(N)):

        bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, l = 'degreelabels')
        bg1_outlier = mg.BinomialGraphs(n_1_outlier, nnode_1, k = 7, fullyConnected = True, l = 'degreelabels')
        bg2 = mg.BinomialGraphs(n_2, nnode_2, k = 4, fullyConnected = True, l = 'degreelabels')
        bg1.Generate()
        bg1_outlier.Generate()
        bg2.Generate()
        Gs = bg1.Gs + bg1_outlier.Gs + bg2.Gs



        rw_kernel = rw.RandomWalk(Gs, c = 0.01, normalize=0)
        K = rw_kernel.fit_ARKU_plus(r = 6, normalize_adj=False, verbose=False)


        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = n, m = n  ), 
                            dict(n = n, m = n ), 
                            dict(Q = Q, y1 = Gs[:n], y2 = Gs[n:] )]
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)

        test_info = pd.concat((test_info, pd.DataFrame({
            'MMD_u':kernel_hypothesis.p_values['MMD_u'],
            'MMD_b':kernel_hypothesis.p_values['MMD_b'],
            'MONK_EST':kernel_hypothesis.p_values['MONK_EST'],
            'sample':sample,
            'test':'degree_test',
            'Q':Q,
            'N':N,
            'B':B,
            'n_1':n_1,
            'n_2':n_2,
            'n_1_outlier':n_1_outlier,
            'n_2_outlier':0

        }, index = [0])), ignore_index=True)



    return test_info



def run_samples_attr(N, B, Q):
    n = 100
    n_1 = 95
    n_1_outlier = 5
    n_2 = 100
    nnode_1 = 50
    nnode_2 = 50
    import myKernels.RandomWalk as rw
    test_info = pd.DataFrame()
    for sample in tqdm.tqdm(range(N)):


        bg1 = mg.BinomialGraphs(n_1, nnode_1, k = 4, fullyConnected = True, a = 'normattr', loc = 0, scale = 0.1)
        bg2 = mg.BinomialGraphs(n_2, nnode_2, k = 4, fullyConnected = True, a = 'normattr', loc = 0, scale = 0.1)
        bg1_outlier = mg.BinomialGraphs(n_1_outlier, nnode_1, k = 4, fullyConnected = True, a = 'normattr', loc = 10, scale = 0.1)
        bg1.Generate()
        bg1_outlier.Generate()
        bg2.Generate()
        Gs = bg1.Gs + bg1_outlier.Gs + bg2.Gs



        graph_list = gk.graph_from_networkx(Gs, node_labels_tag='attr')  # Convert to graphs to Grakel format
        kernel = [{"name": "propagation", "t_max": 5, "w":0.001, "M":'L1', 'with_attributes':True}]

        init_kernel = gk.GraphKernel(kernel= kernel, normalize=0)
        K = init_kernel.fit_transform(graph_list)


        MMD_functions = [mg.MMD_b, mg.MMD_u, mg.MONK_EST]
        kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
        function_arguments = [dict(n = n, m = n  ), 
                            dict(n = n, m = n ), 
                            dict(Q = Q, y1 = Gs[:n], y2 = Gs[n:] )]
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)

        test_info = pd.concat((test_info, pd.DataFrame({
            'MMD_u':kernel_hypothesis.p_values['MMD_u'],
            'MMD_b':kernel_hypothesis.p_values['MMD_b'],
            'MONK_EST':kernel_hypothesis.p_values['MONK_EST'],
            'sample':sample,
            'test':'degree_test',
            'Q':Q,
            'N':N,
            'B':B,
            'n_1':n_1,
            'n_2':n_2,
            'n_1_outlier':n_1_outlier,
            'n_2_outlier':0,
            'kernel':'prop'

        }, index = [0])), ignore_index=True)



    return test_info


if __name__ == '__main__':

    # N = 150
    # B = 3000
    # Q = 5
    # with Pool() as pool:
    #     L = pool.starmap(run_samples, [(N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q)])
        

    #     df = pd.concat(L)

    # print(df)


    # with open('data/Robust/degree_4_4_7_q_5.pkl', 'wb') as f:
    #     pickle.dump(df, f)


    N = 150
    B = 5000
    for Q in [5, 9, 11]:
        print(Q)
        with Pool() as pool:
            L = pool.starmap(run_samples_attr, [(N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q), (N,B,Q)])
            

            df = pd.concat(L)

        print(df)


        with open(f'data/Robust/attributed_q_{Q}.pkl', 'wb') as f:
            pickle.dump(df, f)
