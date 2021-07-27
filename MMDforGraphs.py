
import networkx as nx
import numpy as np
import grakel as gk
import warnings
import concurrent.futures
import pandas as pd
#import time
from datetime import datetime

from numba import njit
import myKernels.DeepKernel as dk
import myKernels.WWL as wl
#from myKernels import WWL
import myKernels.hashkernel as hk

import SBM


# Biased empirical maximum mean discrepancy
@njit
def MMD_b(K: np.array, n: int, m: int):

    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n ** 2) * Kx.sum() + 1.0 / (n * m) * Ky.sum() - 2.0 / (m ** 2) * Kxy.sum()

# Unbiased empirical maximum mean discrepancy
@njit
def MMD_u(K: np.array, n: int, m: int):
    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n* (n - 1.0)) * (Kx.sum() - np.diag(Kx).sum()) + 1.0 / (m * (m - 1.0)) * (Ky.sum() - np.diag(Ky).sum()) - 2.0 / (n * m) * Kxy.sum()

@njit
def BootstrapPval(B:int, K:np.array, n:int, m:int, seed:int):
    """
    :B number of bootstraps
    :K kernel array
    :n number of samples from sample 1
    :m number of samples from sample 2
    :seed for reproducibility
    """
    # mmd biased value of sample
    mmd_b_sample = MMD_b(K, n, m)
    mmd_u_sample = MMD_u(K, n, m)

    # mmd unbiased value of sample

    # list to store null distribution
    mmd_b_null = np.zeros(B)
    mmd_u_null = np.zeros(B)
    K_i = np.empty(K.shape)

    # Bootstrapp with replacement or without that is the question
    p_b_value = 0
    p_u_value = 0
    for b in range(B):  
        #index = rng.randint(low = 0, high = len(K)-1, size = n+m)
        index = np.random.permutation(n+m)
        #K_i = K[index, index[:, None]] #None sets new axix
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]
        #    K_i[i,] = K[:, :]
        # calculate mmd under the null
        mmd_b_null[b] = MMD_b(K_i, n, m)
        mmd_u_null[b] = MMD_u(K_i, n, m)


    p_b_value =  (mmd_b_null > mmd_b_sample).sum()/float(B)
    p_u_value =  (mmd_u_null > mmd_u_sample).sum()/float(B)

    return p_b_value, p_u_value, mmd_b_null, mmd_u_null, mmd_b_sample, mmd_u_sample


@njit
def Boot_median(statistic: np.array, B:int, n:int, m:int, seed:int):

    """
    Permutate a list B times and calculate the median of index :n minus median of n:(n+m)
    """

    result = np.empty(B)

    for boot in range(B):

        index = np.random.permutation(n+m)

        tmp1 = np.zeros(n)
        tmp2 = np.zeros(m)

        for i in range(n):
            tmp1[i] = statistic[index[i]]
        for i in range(m):
            tmp2[i] = statistic[index[n+i]]

        result[boot] = np.median(tmp1) - np.median(tmp2)
        #result[boot] = np.median(statistic[index[:n]]) - np.median(statistic[index[n:(n+m)]])

    return result


# Kernel Matrix for graphs. Based on the grakel package
def KernelMatrix(graph_list: list, kernel: dict, normalize:bool):
    init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
    K = init_kernel.fit_transform(graph_list)
    return K

def GenerateBinomialGraph(n:int,nr_nodes:int,p:float, label:list = None, attributes:list = None):
    """
    :n Number of samples
    :nr_nodes number of nodes
    :label list for node labelling
    :param attributes: list for attributes
    :return: list of networkx graphs
    """
    Gs = []
    for _ in range(n):
        G = nx.fast_gnp_random_graph(nr_nodes, p)
        if not label is None:
            nx.set_node_attributes(G, label, 'label')
        if not label is None:
            nx.set_node_attributes(G, attributes, 'attributes')
        Gs.append(G)

    return Gs

def generateSBM(n:int, pi:list, P:list, label:list, nr_nodes:int):
    """
    :n Number of samples
    :pi probability of belonging to block, must sum to 1
    :P Block probability matrix
    :label list for node labelling
    :return: list of networkx graphs
    """

    Gs = []
    for _ in range(n):
        G = SBM.SBM(P, pi, nr_nodes)
        nx.set_node_attributes(G, label, 'label')
        Gs.append(G)

    return Gs

def generateSBM2(n:int, sizes:list, P:list, label:list):
    """
    :n Number of samples
    :sizes number of node in each block
    :P Block probability matrix
    :label list for node labelling
    :return: list of networkx graphs
    """

    Gs = []
    for _ in range(n):
        G = nx.stochastic_block_model(sizes, P )
        nx.set_node_attributes(G, label, 'label')
        Gs.append(G)

    return Gs



def CalculateGraphStatistics(Gs, n,m):
    """


    :param Gs: list of networkx graphs
    :param n: number of graphs in first sample
    :param m: number of graphs in second sample
    """

    # average degree
    avg_degree_list = np.array(list(map(lambda x: np.average(x.degree, axis = 0)[1], Gs)))
    avg_degree_sample = np.median(avg_degree_list[:n]) - np.median(avg_degree_list[n:(n+m)])
    # median degree
    median_degree_list = np.array(list(map(lambda x: np.median(x.degree, axis = 0)[1], Gs)))
    median_degree_sample = np.median(avg_degree_list[:n]) - np.median(avg_degree_list[n:(n+m)])
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

    test_statistic_sample = dict()
    test_statistic_sample['avg_degree'] = avg_degree_sample
    test_statistic_sample['median_degree'] = median_degree_sample
    test_statistic_sample['max_degree'] = max_degree_sample
    test_statistic_sample['avg_neigh_degree'] = avg_neigh_degree_sample
    test_statistic_sample['avg_clustering'] = avg_clustering_sample
    test_statistic_sample['transitivity'] = transitivity_sample

    test_statistic_list = dict()
    test_statistic_list['avg_degree'] = avg_degree_list
    test_statistic_list['median_degree'] = median_degree_list
    test_statistic_list['max_degree'] = max_degree_list
    test_statistic_list['avg_neigh_degree'] = avg_neigh_degree_list
    test_statistic_list['avg_clustering'] = avg_clustering_list
    test_statistic_list['transitivity'] = transitivity_list


    return test_statistic_list, test_statistic_sample



def GenerateScaleFreeGraph(power:float, nr_nodes:int):
    """
    create a graph with degrees following a power law distribution
    https://stackoverflow.com/questions/28920824/generate-a-scale-free-network-with-a-power-law-degree-distributions
    """
#create a graph with degrees following a power law distribution

    # the outer loop makes sure that we have an even number of edges so that we can use the configuration model
    while True: 
        # s keeps the degree sequence
        s=[]
        while len(s)<nr_nodes:
            nextval = int(nx.utils.powerlaw_sequence(1, power)[0]) #100 nodes, power-law exponent 2.5
            if nextval!=0:
                s.append(nextval)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G= nx.Graph(G) # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def GenerateSamplesOfScaleFreeGraphs(n:int,nr_nodes:int,power:float, label:list = None, attributes:list = None):
    """
    :n Number of samples
    :nr_nodes: number of nodes
    :param power: power law parameter
    :param label: list for node labelling
    :param attributes: list for attributes
    :return: list of networkx graphs
    """
    Gs = []
    for _ in range(n):
        G = GenerateScaleFreeGraph(power, nr_nodes)
        if not label is None:
            nx.set_node_attributes(G, label, 'label')
        if not label is None:
            nx.set_node_attributes(G, attributes, 'attributes')
        Gs.append(G)

    return Gs


class BoostrapMethods():
    """
    Various Bootstrap/Permutation Functions

    Class for permutation testing
    """

    def __init__(self, list_of_functions:list) -> None:
        """
        :param list_of_functions: List of functions that should be applied to the the permutated K matrix
        """

        self.list_of_functions = list_of_functions

    @staticmethod
    def issymmetric(a, rtol=1e-05, atol=1e-08):
        """
        Check if matrix is symmetric
        """
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    @njit
    def PermutationScheme(K) -> np.array:
        """
        :param K: Kernel Matrix
        """

        K_i = np.empty(K.shape)
        index = np.random.permutation(K.shape[0])
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]

        return K_i

    @staticmethod
    @njit
    def BootstrapScheme(K) -> np.array:
        """
        :param K: Kernel Matrix
        """

        K_i = np.empty(K.shape)
        index = np.random.choice(K.shape[0])
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]

        return K_i


    def Bootstrap(self,K,function_arguments,B:int, method:str = "PermutationScheme", check_symmetry:bool = False) -> None:
        """
        :param K: Kernel matrix that we want to permutate
        :param function_arguments: List of dictionaries with inputs of for its respective function in list_of_functions,  excluding K. If no input set as None.
        :param B: Number of Bootstraps
        :param method: Which permutation method should be applied?
        :param check_symmetry: Should the scheme check if the matrix is symmetirc, each time? Adds time complexity
        """
        self.K = K
        self.function_arguments = function_arguments
        assert self.issymmetric(self.K), "K is not symmetric"

        # keep p-value result from each MMD function
        p_value_dict = dict()
        

        # get arguments of each function ready for evaluation
        inputs = [None] * len(self.list_of_functions)
        for i in range(len(self.list_of_functions)):
            if self.function_arguments[i] is None:
                continue
            inputs[i] =  ", ".join("=".join((k,str(v))) for k,v in sorted(self.function_arguments[i].items()))

        # Get the evaluation method
        evaluation_method = getattr(self, method)

        # Calculate sample mmd statistic, and create a dictionary for bootstrapped statistics
        sample_statistic = dict()
        boot_statistic = dict()
        for i in range(len(self.list_of_functions)):
            # the key is the name of the MMD (test statistic) function
            key = self.list_of_functions[i].__name__
            # string that is used as an input to eval, the evaluation function
            if self.function_arguments[i] is None:
                eval_string = key + "(K =self.K" + ")"
            else:
                eval_string = key + "(K =self.K, " + inputs[i] + ")"

            sample_statistic[key] = eval(eval_string)
            boot_statistic[key] = np.zeros(B)



        # Now Perform Bootstraping
        for boot in range(B):
            K_i = evaluation_method(self.K)
            if check_symmetry:
                if self.issymmetric(K_i):
                    warnings.warn("Not a Symmetric matrix", Warning)

            # apply each statistic, and keep the bootstraped/permutated value
            for i in range(len(self.list_of_functions)):
                eval_string = self.list_of_functions[i].__name__ + "(K =K_i, " + inputs[i] + ")"
                boot_statistic[self.list_of_functions[i].__name__][boot] = eval(eval_string)
                

        # calculate p-value
        for key in sample_statistic.keys():
            p_value_dict[key] =  (boot_statistic[key] >= sample_statistic[key]).sum()/float(B)


        self.p_values = p_value_dict
        self.sample_test_statistic = sample_statistic
        self.boot_test_statistic = boot_statistic

class BootstrapGraphStatistic():
    """
    class that bootstrap graph statistics

    """

    def __init__(self, G1:list, G2:list, list_of_functions:list ) -> None:
        """
        :param G1: list of networkx graphs, sample 1
        :param G2: list of networkx graphs, sample 2
        :param list_of_functions: List of functions that calculate graph statistics of a graph
        """
        self.G1 = G1
        self.G2 = G2
        # combine G1 and G2 into one list
        self.Gs = G1 + G2
        self.n1 = len(G1)
        self.n2 = len(G2)
        self.list_of_functions = list_of_functions


    def Statistic(self, func):
        """
        calculate the sample statistic based on func, the median is used
        """
        statistics = np.array(list(map(lambda x: func(x), self.Gs)))
        return np.median(statistics[:self.n1]) - np.median(statistics[self.n1:(self.n1+self.n2)]), statistics
        

    @staticmethod
    def permutate_statistic(statistic, n1, n2) -> np.array:

        index = np.random.permutation(n1 + n2)

        return statistic[index]


    def Bootstrap(self, B:int, method = "permutate_statistic"):

        """
        Permutate a list B times and calculate the median of index :n minus median of n:(n+m)

        :param method: Method used to shuffle the statistics
        """

        # keep p-value result from each MMD function
        p_value_dict = dict()


        # Get the evaluation method
        evaluation_method = getattr(self, method)

        # Calculate sample mmd statistic, and create a dictionary for bootstrapped statistics
        # the test statistic
        sample_test_statistic = dict()
        # the graph statistic value of each graph
        graph_statistics = dict()
        # bootstrapped test statistic
        boot_test_statistic = dict()

        for i in range(len(self.list_of_functions)):
            # the key is the name of the graph statistic function
            key = self.list_of_functions[i].__name__

            sample_test_statistic[key], graph_statistics[key] = self.Statistic(self.list_of_functions[i])
            boot_test_statistic[key] = np.zeros(B)



        # Now Perform Bootstraping for each graph statistic
        for boot in range(B):
            
            for i in range(len(self.list_of_functions)):
                key = self.list_of_functions[i].__name__
                statistic_boot = evaluation_method(graph_statistics[key], self.n1, self.n2)
                boot_test_statistic[key][boot] = np.median(statistic_boot[:self.n1]) - np.median(statistic_boot[self.n1:(self.n1+self.n2)])
                

        # calculate p-value, it is a two-tailed test
        for key in sample_test_statistic.keys():
            p_value_dict[key] =  2*np.min([(boot_test_statistic[key] > sample_test_statistic[key]).sum(),(boot_test_statistic[key] < sample_test_statistic[key]).sum()])/float(B)


        self.p_values = p_value_dict
        self.sample_test_statistic = sample_test_statistic
        self.boot_test_statistic = boot_test_statistic


def average_degree(G):
    return np.average(G.degree, axis = 0)[1]
def median_degree(G):
    return float(np.median(G.degree, axis = 0)[1])
def avg_neigh_degree(G):
    return np.average(list(nx.average_neighbor_degree(G).values()))
def avg_clustering(G):
    return nx.average_clustering(G)
def transitivity(G):
    return nx.transitivity(G)



class DegreeGraphs():
    """
    Wrapper for graph generator which are fully specified by degree
    """

    def __init__(self, n, nnode, k, l = None,  a = None, fullyConnected = False, **kwargs) -> None:
        """
        :param kernel: Dictionary with kernel information
        :param n: Number of samples
        :param nnode: Number of nodes
        :param k: Degree
        :param l: Labelling scheme
        :param a: Attribute scheme
        :param **kwargs: Arguments for the labelling functions
        :param path: save data path
        """
        self.n = n
        self.nnode = nnode
        self.k = k
        self.l = l
        self.a = a
        self.kwargs = kwargs
        self.fullyConnected = fullyConnected


    def samelabels(self, G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 'a') for i in range(len(G)) ) )

    def degreelabels(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """

        nodes_degree = dict(G.degree)
        return {key: str(value) for key, value in nodes_degree.items()}

    def normattr(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """
        loc = self.kwargs.get('loc', 0)
        scale = self.kwargs.get('scale', 1)
        return dict( ( (i, np.random.normal(loc = loc, scale = scale, size = (1,))) for i in range(len(G)) ) )

    def rnglabels(self, G):
        """
        labelling Scheme. Nodes labelled accodring to a discrete pmf

        :param G: Networkx graph
        :param pmf: pmf as list. If None then uniform over all entries
        :return: Dictionary
        """
        import string
        assert not self.kwargs['nr_letters'] is None, "Number of letters (nr_letters) has to be specified"
        assert  np.sum(self.kwargs['pmf']) >0.999, "pmf has to sum to 1"

        # check if the pmf of labels has been given
        if not 'pmf' in self.kwargs.keys():
            pmf = None
        else:
            pmf = self.kwargs['pmf']

        letters = list(string.ascii_lowercase[:self.kwargs['nr_letters']])
        return dict( ( (i, np.random.choice(letters, p = pmf)) for i in range(len(G)) ) )


class BinomialGraphs(DegreeGraphs):
    """
    Class that generates tvo samples of binomial graphs and compares them.
    """
    def __init__(self,  n, nnode, k, l = None,  a = None, **kwargs):
        super().__init__( n, nnode, k, l ,  a, **kwargs )
        self.p = k/float(nnode-1)

    def Generate(self) -> None:
        """
        :return: list of networkx graphs
        """
        self.Gs = []
        for _ in range(self.n):
            if self.fullyConnected:
                while True:
                    G = nx.fast_gnp_random_graph(self.nnode, self.p)
                    if nx.is_connected(G):
                        break
            else:
                G = nx.fast_gnp_random_graph(self.nnode, self.p)

            if (not self.l is None) and (not self.a is None):
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            elif not self.l is None:
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
            elif not self.a is None:
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            self.Gs.append(G)



class SBMGraphs():
    """
    Class that generates tvo samples of SBM graphs and compares them.
    """
    def __init__(self,  n, sizes, P, params = {}, fullyConnected = True, l = None,  a = None):
        self.sizes = sizes
        self.P = P
        self.n = n
        self.l = l
        self.a = a
        self.params = params
        self.fullyConnected = fullyConnected

        assert len(self.sizes) == self.P.shape[0]
        assert len(self.sizes) == self.P.shape[1]
        assert self.params.get('label_pmf', np.zeros(self.P.shape)).shape[0] == self.P.shape[0]
    

    def samelabels(self, G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 'a') for i in range(len(G)) ) )

    def degreelabels(self, G):
        """
        labelling Scheme. Nodes labelled with their degree

        :param G: Networkx graph
        :return: Dictionary
        """

        nodes_degree = dict(G.degree)
        return {key: str(value) for key, value in nodes_degree.items()}

    def BlockLabelling(self, G):
        """
        Labelling Scheme. Nodes are labelled according to their block but are allowed to have imputations, that is some nodes can be labelled according to other blocks.
        The params dictionary should have a key called label_pmf, which is a B times B matrix where each row corresponds to the pmf of labels of block i, where i is the row number.
        Very import that the first index of the pmfs is the probability for label 1, second index is the probability that a node has label 2, etc..
        """

        import string

        letters = list(string.ascii_lowercase[:len(self.sizes)])
        
        nr_blocks = self.P.shape[0]

        label_pmf = self.params['label_pmf']
        return {v[0]:np.random.choice(letters[:nr_blocks],p = label_pmf[v[1]["block"],:] )  for v in G.nodes(data=True) } 



    def Generate(self):
        """
        :return: list of networkx graphs

        self.params should include:
            sizes (list of ints) – Sizes of blocks
            p (list of list of floats) – Element (r,s) gives the density of edges going from the nodes of group r to nodes of group s. Must match the number of groups
        """



        self.Gs = []
        for _ in range(self.n):
            if self.fullyConnected:
                while True:
                    G = nx.stochastic_block_model(self.sizes, self.P)
                    if nx.is_connected(G):
                        break
            else:
                G = nx.fast_gnp_random_graph(self.sizes, self.P)

            # label scheme
            if (not self.l is None) and (not self.a is None):
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            elif not self.l is None:
                label = getattr(self, self.l)
                label_dict = label(G)
                nx.set_node_attributes(G, label_dict, 'label')
            elif not self.a is None:
                attributes = getattr(self, self.a)
                attribute_dict = attributes(G)
                nx.set_node_attributes(G, attribute_dict, 'attr')
            self.Gs.append(G)


        

        




def iterationGraphStat(N:int, Graph_Statistics_functions, bg1, bg2, B:int):
    """
    Function That generates samples according to the graph generators bg1 and bg2 and calculates graph statistics

    :param N: Number of samples
    :param kernel: Dictionary with the kernel arguments
    :param normalize: Should the kernel be normalized
    :param Graph_Statistics_functions: List of functions that calculate graph statistics
    :param bg1: Class that generates sample 1
    :param bg2: Class that generates sample 1
    :param B: Number of bootstraps
    :param kernel_hypothesis: The class that handes bootstrapping
    :param kernel_library: which package or which own kernel: Grakel, deepkernel (own function), 


    :return test_statistic_p_val: Dictionary that stores the p-value of each Graph Statistic test according to the kernel_hypothesis bootstrap result
    """
    # Store the p-values for each test statistic for each test iteration
    test_statistic_p_val = dict()
    for i in range(len(Graph_Statistics_functions)):
        key = Graph_Statistics_functions[i].__name__
        test_statistic_p_val[key] = np.array([-1.0] * N)

    # Calculate basic  graph statistics hypothesis testing
    for sample in range(N):

        bg1.Generate()
        bg2.Generate()


        hypothesis_graph_statistic = BootstrapGraphStatistic(bg1.Gs, bg2.Gs, Graph_Statistics_functions)
        hypothesis_graph_statistic.Bootstrap(B = B)
        # match the corresponding p-value for this sample
        for key in test_statistic_p_val.keys():
            test_statistic_p_val[key][sample] = hypothesis_graph_statistic.p_values[key]



    return test_statistic_p_val



def iteration(N:int, kernel:dict, normalize:bool, MMD_functions, bg1, bg2, B:int, kernel_hypothesis, kernel_library="Grakel", node_labels_tag='label'):
    """
    Function That generates samples according to the graph generators bg1 and bg2 and calculates graph statistics

    :param N: Number of samples
    :param kernel: Dictionary with the kernel arguments
    :param normalize: Should the kernel be normalized
    :param MMD_functions: List of functions do MMD hypothesis testing
    :param bg1: Class that generates sample 1
    :param bg2: Class that generates sample 1
    :param B: Number of bootstraps
    :param kernel_hypothesis: The class that handes bootstrapping
    :param kernel_library: which package or which own kernel: Grakel, deepkernel (own function), 
    :param node_labels_tag: Node label tak for Grakel kernels, usually label for labeled graphs and attr if attributed. Should be in concordance with the label/attribute generation mechanism in the generation scheme of bg1 and bg2.

    :return Kmax: List that stores the maximum kernel value for each sample iteration
    :return p_values: Dictionary that stores the p-value of each MMD tset according to the kernel_hypothesis bootstrap result
    :return mmd_samples: Dictionary that stores the sample MMD value function for each function in the MMD_function list
    :return test_statistic_p_val: Dictionary that stores the p-value of each Graph Statistic test according to the kernel_hypothesis bootstrap result
    """
        # Keep p-values and the sample MMD test statistic               
    p_values = dict()
    mmd_samples = dict()
    for i in range(len(MMD_functions)):
        key = MMD_functions[i].__name__
        p_values[key] = np.array([-1.0] * N)
        mmd_samples[key] = np.array([-1.0] * N)


    # Store K max for acceptance region
    Kmax = np.array([0] * N, dtype = np.float64)

    for sample in range(N):
    
        # sample binomial graphs
        bg1.Generate()
        bg2.Generate()
        Gs = bg1.Gs + bg2.Gs
        graph_list = gk.graph_from_networkx(Gs, node_labels_tag = node_labels_tag)

        # # Calculate basic  graph statistics hypothesis testing
        # if graphStatistics:
        #     hypothesis_graph_statistic = BootstrapGraphStatistic(bg1.Gs, bg2.Gs, Graph_Statistics_functions)
        #     hypothesis_graph_statistic.Bootstrap(B = B)
        #     # match the corresponding p-value for this sample
        #     for key in test_statistic_p_val.keys():
        #         test_statistic_p_val[key][sample] = hypothesis_graph_statistic.p_values[key]
        
        # Kernel hypothesis testing
        # Fit a kernel, Note the Grakel uses graph_list while myKernels use Gs
        if kernel_library == "Grakel":
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
            K = init_kernel.fit_transform(graph_list)
        elif kernel_library == "deepkernel":
            init_kernel = dk.DK(params = kernel)
            K = init_kernel.fit_transform(Gs)
        elif kernel_library == "wwl":
            init_kernel = wl.WWL(param = {'discount':kernel['discount'],'h':kernel['h'], 'sinkhorn':kernel['sinkhorn'] })
            K = init_kernel.fit_transform(Gs)
        else:
            raise ValueError(f"{kernel_library} not defined")

        # print(K)
        Kmax[sample] = K.max()
        if np.all((K == 0)):
            warnings.warn("all element in K zero")

        # bootstrap function argument
        function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
        
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        for i in range(len(MMD_functions)):
            key = MMD_functions[i].__name__
            p_values[key][sample] = kernel_hypothesis.p_values[key]
            mmd_samples[key][sample] = kernel_hypothesis.sample_test_statistic[key]

    return dict(Kmax = Kmax, p_values = p_values, mmd_samples = mmd_samples)#), test_statistic_p_val = test_statistic_p_val)


if __name__ == '__main__':
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 5
    m = 5

    average_degree = 6
    print('what is up')


    bg1 = BinomialGraphs(n, nr_nodes_1,average_degree, l = None)
