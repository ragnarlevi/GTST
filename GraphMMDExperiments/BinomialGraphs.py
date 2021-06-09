

# This script experiments with MMD for two binomial graphs



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
parser.add_argument('-norm', '--normalize', type=int,metavar='', help='Should kernel be normalized')
parser.add_argument('-n1', '--NrSamples1', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-n2', '--NrSamples2', type=int,metavar='', help='Number of graphs in sample 1')
parser.add_argument('-nnode1', '--NrNodes1', type=int,metavar='', help='Number of nodes in each graph in sample 1')
parser.add_argument('-nnode2', '--NrNodes2', type=int,metavar='', help='Number of nodes in each graph in sample 2')
parser.add_argument('-k1', '--AverageDegree1', type=float,metavar='', help='Average degree of each graph in sample 1')
parser.add_argument('-k2', '--AverageDegree2', type=float,metavar='', help='Average degree of each graph in sample 2')
parser.add_argument('-l1', '--label1', type=str,metavar='', help='Label scheme of sample 1')
parser.add_argument('-l2', '--label2', type=str,metavar='', help='Label scheme of sample 2')
parser.add_argument('-a1', '--attributes1', type=str,metavar='', help='Attribute scheme of sample 1')
parser.add_argument('-a2', '--attributes2', type=str,metavar='', help='Attribute scheme of sample 2')

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_false', help = 'print verbose')


args = parser.parse_args()



class BinomialGraphs():
    """
    Class that generates tvo samples of binomial graphs and compares them.
    """

    def __init__(self, B, n1, n2, nnode1, nnode2, k1, k2, l1 = None, l2 = None, a1 = None, a2 = None) -> None:
        """
        :param kernel: Dictionary with kernel information
        :param B: Number of Bootstraps
        :param path: save data path
        """
        self.B = B
        self.path = path
        self.n1 = n1
        self.n2 = n2
        self.nnode1 = nnode1
        self.nnode2 = nnode2
        self.k1 = k1
        self.k2 = k2
        self.l1 = l1
        self.l2 = l2
        self.a1 = a1
        self.a2 = a2

    @staticmethod
    def samelabels(G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 'a') for i in range(len(G)) ) )

    def Generate(self) -> None:
        """
        
        :param MMD_functions: List of functions that should be applied to the the bootstrapped K matrix
        :param MMD_arguments: List of dictionaries with inputs of for its respective function in MMD_functions,  excluding the kernel matrix K. If no input set as None.
        :param graph_statistics_func: List of functions that calculate graph statistics of a graph
        """

        # Set probability
        p_edge_1 = self.k1/float(self.nnode1-1)
        p_edge_2 = self.k2/float(self.nnode2-1)
        assert not ( self.a1 != None and self.l1 != None), "Sample 1 can't have both attributes and labels"
        assert not (self.l1 != None and self.l2 == None), "One sample can not have labels while the other one has attributes"
        assert not (self.l1 == None and self.l2 != None), "One sample can not have labels while the other one has attributes"
        assert not (self.a1 != None and self.a2 == None), "One sample can not have labels while the other one has attributes"
        assert not (self.a1 == None and self.a2 != None), "One sample can not have labels while the other one has attributes"
        # Set label and attributes 
        if not self.l1 is None:
            label_1 = getattr(self, self.l1)
        if not self.l2 is None:
            label_2 = getattr(self, self.l2)
        if not self.a1 is None:
            attr_1 = getattr(self, self.a1)
        if not self.a2 is None:
            attr_2 = getattr(self, self.a2)
        
    
        # sample binomial graphs
        if not self.l1 is None:
            self.G1 = mg.GenerateBinomialGraph(n = self.n1, nr_nodes = self.nnode1, p = p_edge_1, label = label_1)
            self.G2 = mg.GenerateBinomialGraph(n = self.n2, nr_nodes = self.nnode2, p = p_edge_2, label = label_2)
            self.Gs = self.G1 + self.G2
            self.graph_list = gk.graph_from_networkx(self.Gs, node_labels_tag='label')
        elif not self.a1 is None:
            self.G1 = mg.GenerateBinomialGraph(n = self.n1, nr_nodes = self.nnode1, p = p_edge_1, attributes = attr_1)
            self.G2 = mg.GenerateBinomialGraph(n = self.n2, nr_nodes = self.nnode2, p = p_edge_2, attributes = attr_2)
            self.Gs = self.G1 + self.G2
            self.graph_list = gk.graph_from_networkx(self.Gs, node_labels_tag='attributes')
        else:
            self.G1 = mg.GenerateBinomialGraph(n = self.n1, nr_nodes = self.nnode1, p = p_edge_1)
            self.G2 = mg.GenerateBinomialGraph(n = self.n2, nr_nodes = self.nnode2, p = p_edge_2)
            self.Gs = self.G1 + self.G2
            self.graph_list = gk.graph_from_networkx(self.Gs)

        


        pass




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
    l1 = args.label1
    l2 = args.label2
    a1 = args.attributes1
    a2 = args.attributes2
    k1 = args.AverageDegree1
    k2 = args.AverageDegree2


    # which graph statistics functions should we test?
    Graph_Statistics_functions = [mg.average_degree, mg.median_degree, mg.avg_neigh_degree, mg.avg_clustering, mg.transitivity]

    # functions used for kernel testing
    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)

    # Initialize Graph generator class
    bg = BinomialGraphs(B, n1, n2, nnode1, nnode2, k1, k2, l1, l2, a1, a2 )

    # Probability of type 1 error
    alphas = np.linspace(0.025, 0.975, 39)

    # set seed
    seed = 42
    
    now = datetime.now()
    time = pd.Timestamp(now)
    
    # Kernel specification
    # kernel = [{"name": "WL", "n_iter": 4}]
    kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "vertex_histogram"}]
    # kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "SP"}]
    # kernel = [{"name": "SP", "with_labels": True}]
    # kernel = [{"name": "svm_theta"}]
    # kernel = [{"name": "pyramid_match", "with_labels":False}]
    # kernel = [{"name": "ML", "n_samples":20}]
    
    # Store outcome in a data
    df = pd.DataFrame()

    # Keep p-values and the sample MMD test statistic               
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

    for sample in range(N):
        total_time = datetime.now()
    
        # sample binomial graphs
        bg.Generate()

        # Calculate basic  graph statistics hypothesis testing
        if graphStatistics:
            hypothesis_graph_statistic = mg.BootstrapGraphStatistic(bg.G1, bg.G2, Graph_Statistics_functions)
            hypothesis_graph_statistic.Bootstrap(B = B)
            # match the corresponding p-value for this sample
            for key in test_statistic_p_val.keys():
                test_statistic_p_val[key][sample] = hypothesis_graph_statistic.p_values[key]
        
        # Kernel hypothesis testing
        # Fit a kernel
        init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
        K = init_kernel.fit_transform(bg.graph_list)
        if np.all((K == 0)):
            warnings.warn("all element in K zero")

        function_arguments=[dict(n = n1, m = n2 ), dict(n = n1, m = n2 )]
        
        kernel_hypothesis.Bootstrap(K, function_arguments,B = 1000)
        for i in range(len(MMD_functions)):
            key = MMD_functions[i].__name__
            p_values[key][sample] = kernel_hypothesis.p_values[key]
            mmd_samples[key][sample] = kernel_hypothesis.sample_test_statistic[key]


    # Calculate ROC curve

    for alpha in alphas:
        
        # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

        # power of MMD tests (including distribution free test)
        power_mmd = dict()

        for i in range(len(MMD_functions)):
            key = MMD_functions[i].__name__
            power_mmd[key] = (np.array(p_values[key]) < alpha).sum()/float(N)
            if key == 'MMD_u':
                tmp = mmd_samples[key] < (4*K.max()/np.sqrt(float(n1)))*np.sqrt(np.log(1.0/alpha))
                power_mmd[key + "_distfree"] = 1.0 - float(np.sum(tmp))/float(N)
            if key == 'MMD_b':
                tmp = np.sqrt(mmd_samples[key]) < np.sqrt(2.0*float(K.max())/float(n1))*(1.0 + np.sqrt(2.0*np.log(1/alpha)))
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
                        'run_time':str((datetime.now() - total_time))}, index = [0])
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

    print(datetime.now() - now )








    # # test specification
    # nr_nodes_same = True
    # nr_samples_same = True
    # nr_nodes_1_list = [20, 30, 50, 100, 150, 200]
    # n_list = [5, 10, 15, 20, 50, 100]
    # average_degree = 2.0

    # # are nodes labelled?
    # labelling = True
    # # are nodes attributed?
    # attributed = False
    # assert labelling != attributed, "Graph with both labels and attributes currently not supported"
    # # set seed
    # seed = 42
    
    # # Kernel specification
    # # kernel = [{"name": "WL", "n_iter": 4}]
    # kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "vertex_histogram"}]
    # # kernel = [{"name": "weisfeiler_lehman", "n_iter": 4}, {"name": "SP"}]
    # # kernel = [{"name": "SP", "with_labels": True}]
    # # kernel = [{"name": "svm_theta"}]
    # # kernel = [{"name": "pyramid_match", "with_labels":False}]
    # # kernel = [{"name": "ML", "n_samples":20}]

    # alphas = np.linspace(0.025, 0.975, 39)


    # # store time of run
    # now = datetime.now()
    # time = pd.Timestamp(now)

    # # Store outcome in a data
    # df = pd.DataFrame() #pd.read_pickle("runs.pkl")
    
    # # Bunch of for loops for each test specification, number of nodes, number of samples, the probability etc. The inner-most loop does Bootstrap for N samples
    # for nr_nodes_1 in nr_nodes_1_list:
    #     if nr_nodes_same:
    #         nr_nodes_2_list = [nr_nodes_1]
        
    #     for nr_nodes_2 in nr_nodes_2_list:
    #         print(nr_nodes_1)
    #         print(nr_nodes_2)

    #         for n in n_list:
    #             if nr_samples_same:
    #                 m_list = [n]

    #             for m in m_list:
    #                 print(n)
    #                 print(m)

    #                 for sample_2_margin in tqdm(avg_degree_add):  
                        
    #                     # Set probability
    #                     p_edge_1 = average_degree/float(nr_nodes_1-1)
    #                     p_edge_2 = (average_degree + sample_2_margin)/float(nr_nodes_2-1)
                                
    #                     # Set label (all nodes have same label, just required for some kernels)
    #                     if labelling:
    #                         label_1 = dict( ( (i, 'a') for i in range(nr_nodes_1) ) )
    #                         label_2 = dict( ( (i, 'a') for i in range(nr_nodes_2) ) )
                        
    #                     if attributed:
    #                         attr_1 = dict( ( (i, [1]) for i in range(nr_nodes_1) ) )
    #                         attr_2 = dict( ( (i, [1]) for i in range(nr_nodes_2) ) )

    #                     # keep p values and test statistic values for each MMD test
    #                     p_values = dict()
    #                     mmd_samples = dict()
    #                     for i in range(len(MMD_functions)):
    #                         key = MMD_functions[i].__name__
    #                         p_values[key] = np.array([-1.0] * N)
    #                         mmd_samples[key] = np.array([-1.0] * N)

    #                     # Store the p-values for each test statistic for each test iteration
    #                     test_statistic_p_val = dict()
    #                     for i in range(len(Graph_Statistics_functions)):
    #                         key = Graph_Statistics_functions[i].__name__
    #                         test_statistic_p_val[key] = [0] * N

    #                     for sample in range(N):
    #                         total_time = datetime.now()
                        
    #                         # sample binomial graphs
    #                         if attributed:
    #                             G1 = mg.GenerateBinomialGraph(n = n, nr_nodes = nr_nodes_1, p = p_edge_1, attributes=attr_1)
    #                             G2 = mg.GenerateBinomialGraph(n = m, nr_nodes = nr_nodes_2, p = p_edge_2, attributes=attr_2)
    #                             Gs = G1 +G2
    #                             graph_list = gk.graph_from_networkx(Gs, node_labels_tag='attributes')
    #                         elif labelling:
    #                             G1 = mg.GenerateBinomialGraph(n = n, nr_nodes = nr_nodes_1, p = p_edge_1, label = label_1)
    #                             G2 = mg.GenerateBinomialGraph(n = m, nr_nodes = nr_nodes_2, p = p_edge_2, label = label_2)
    #                             Gs = G1 +G2
    #                             graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')
    #                         else:
    #                             Gs = mg.GenerateBinomialGraph(n = n, nr_nodes = nr_nodes_1, p = p_edge_1)
    #                             G2 = mg.GenerateBinomialGraph(n = m, nr_nodes = nr_nodes_2, p = p_edge_2)
    #                             Gs = G1 +G2
    #                             graph_list = gk.graph_from_networkx(Gs)

    #                         # Calculate basic  graph statistics hypothesis testing
    #                         if graphStatistics:
    #                             hypothesis_graph_statistic = mg.BootstrapGraphStatistic(G1, G2, Graph_Statistics_functions)
    #                             hypothesis_graph_statistic.Bootstrap(B = B)
    #                             # match the corresponding p-value for this sample
    #                             for key in test_statistic_p_val.keys():
    #                                 test_statistic_p_val[key][sample] = hypothesis_graph_statistic.p_values[key]
                            
    #                         # Kernel hypothesis testing
    #                         # Fit a kernel
    #                         init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
    #                         K = init_kernel.fit_transform(graph_list)
    #                         if np.all((K == 0)):
    #                             warnings.warn("all element in K zero")


    #                         function_arguments=[dict(n = n, m = m ), dict(n = n, m = m )]
                            
    #                         kernel_hypothesis.Bootstrap(K, function_arguments,B = 1000)
    #                         for i in range(len(MMD_functions)):
    #                             key = MMD_functions[i].__name__
    #                             p_values[key][sample] = kernel_hypothesis.p_values[key]
    #                             mmd_samples[key][sample] = kernel_hypothesis.sample_test_statistic[key]

   
    #                     # Calculate ROC curve

    #                     for alpha in alphas:
                            
    #                         # type II error is the case when be p_val > alpha so power is 1 - #(p_val>alpha)/N <-> (N - #(p_val>alpha))/N <-> #(p_val<alpha)/N

    #                         # power of MMD tests (including distribution free test)
    #                         power_mmd = dict()

    #                         for i in range(len(MMD_functions)):
    #                             key = MMD_functions[i].__name__
    #                             power_mmd[key] = (np.array(p_values[key]) < alpha).sum()/float(N)
    #                             if key == 'MMD_u':
    #                                 tmp = mmd_samples[key] < (4*K.max()/np.sqrt(float(n)))*np.sqrt(np.log(1.0/alpha))
    #                                 power_mmd[key + "_distfree"] = 1.0 - float(np.sum(tmp))/float(N)
    #                             if key == 'MMD_b':
    #                                 tmp = np.sqrt(mmd_samples[key]) < np.sqrt(2.0*float(K.max())/float(n))*(1.0 + np.sqrt(2.0*np.log(1/alpha)))
    #                                 power_mmd[key + "_distfree"] = 1.0 - float(np.sum(tmp))/float(N)

    #                         # power of "sufficient" statistics
    #                         power_graph_statistics = dict()
    #                         if graphStatistics:
    #                             for i in range(len(Graph_Statistics_functions)):
    #                                 key = Graph_Statistics_functions[i].__name__
    #                                 power_graph_statistics[key] = (np.array(test_statistic_p_val[key]) < alpha).sum()/float(N)

    #                         # Store the run information in a dataframe,
    #                         tmp = pd.DataFrame({'kernel': str(kernel), 
    #                                         'alpha':alpha,
    #                                         'nr_nodes_1': nr_nodes_1,
    #                                         'nr_nodes_2': nr_nodes_2,
    #                                         'p_edge_1': p_edge_1,
    #                                         'p_edge_2': p_edge_2,
    #                                         'degree_1': average_degree ,
    #                                         'degree_2':(average_degree + sample_2_margin),
    #                                         'ratio_p': np.round(p_edge_2/p_edge_1,3),
    #                                         'ratio_degree': np.round((average_degree + sample_2_margin)/average_degree,3),
    #                                         'n':n,
    #                                         'm':m,
    #                                         'timestap':time,
    #                                         'B':B,
    #                                         'N':N,
    #                                         'run_time':str((datetime.now() - total_time))}, index = [0])
    #                         # Add power
    #                         if len(power_graph_statistics) != 0:
    #                             for k,v in power_graph_statistics.items():
    #                                 tmp[k] = v
    #                         if len(power_mmd) != 0:
    #                             for k,v in power_mmd.items():
    #                                 tmp[k] = v

    #                         # add to the main data frame
    #                         df = df.append(tmp, ignore_index=True)

    #                     # Save the dataframe at each iteration each such that if out-of-memory or time-out happen we at least have some of the information.
    #                     with open(path, 'wb') as f:
    #                             pickle.dump(df, f)

















