
import networkx as nx
import numpy as np
import grakel as gk
import warnings
#import time
from datetime import datetime

from numba import njit

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
    for i in range(n):
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
    for i in range(n):
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
    for i in range(n):
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
    for i in range(n):
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

    def Bootstrap(self, K:np.array, function_arguments:list, B:int, method:str = "PermutationScheme", check_symmetry:bool = False) -> None:
        """
        :param K: Kernel matrix that we want to permutate
        :param function_arguments: List of dictionaries with inputs of for its respective function in list_of_functions,  excluding K. If no input set as None.
        :param B: Number of Bootstraps
        :param method: Which permutation method should be applied?
        :param check_symmetry: Should the scheme check if the matrix is symmetirc, each time? Adds time complexity
        """
        assert self.issymmetric(K), "K is not symmetric"

        # keep p-value result from each MMD function
        p_value_dict = dict()
        

        # get arguments of each function ready for evaluation
        inputs = [None] * len(self.list_of_functions)
        for i in range(len(self.list_of_functions)):
            if function_arguments[i] is None:
                continue
            inputs[i] =  ", ".join("=".join((k,str(v))) for k,v in sorted(function_arguments[i].items()))

        # Get the evaluation method
        evaluation_method = getattr(self, method)

        # Calculate sample mmd statistic, and create a dictionary for bootstrapped statistics
        sample_statistic = dict()
        boot_statistic = dict()
        for i in range(len(self.list_of_functions)):
            # the key is the name of the MMD (test statistic) function
            key = self.list_of_functions[i].__name__
            # string that is used as an input to eval, the evaluation function
            if function_arguments[i] is None:
                eval_string = key + "(K =K" + ")"
            else:
                eval_string = key + "(K =K, " + inputs[i] + ")"

            sample_statistic[key] = eval(eval_string)
            boot_statistic[key] = np.zeros(B)



        # Now Perform Bootstraping
        for boot in range(B):
            K_i = evaluation_method(K)
            if check_symmetry:
                if self.issymmetric(K_i):
                    warnings.warn("Not a Symmetric matrix", Warning)

            # apply each statistic, and keep the bootstraped/permutated value
            for i in range(len(self.list_of_functions)):
                eval_string = self.list_of_functions[i].__name__ + "(K =K_i, " + inputs[i] + ")"
                boot_statistic[self.list_of_functions[i].__name__][boot] = eval(eval_string)
                

        # calculate p-value
        for key in sample_statistic.keys():
            p_value_dict[key] =  (boot_statistic[key] > sample_statistic[key]).sum()/float(B)


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
        # Shuffled boostrap_statistics
        boot_graph_statistic = dict()
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
    return np.float(np.median(G.degree, axis = 0)[1])
def avg_neigh_degree(G):
    return np.average(list(nx.average_neighbor_degree(G).values()))
def avg_clustering(G):
    return nx.average_clustering(G)
def transitivity(G):
    return nx.transitivity(G)



if __name__ == "__main__":
    time = datetime.now()
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 60
    m = 100

    Block_Matrix_1 = np.array([[0.15, 0.05, 0.05],
                            [0.05, 0.15, 0.05],
                            [0.05, 0.05, 0.15]])

    Block_Matrix_2 = np.array([[0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1]])

    pi = [1/3] * 3



    # Kernel specification
    kernel = [{"name": "WL-OA", "n_iter": 3}]

    

            
    # Set block probabilities
    p1 = np.array(Block_Matrix_1)
    l = 0.11

    p2 = (1-l)*np.array(Block_Matrix_1) + l*np.array(Block_Matrix_2)

    print(p1)
    print(p2)

            
    # Set label (all nodes have same label, just required for some kernels)
    label_1 = dict( ( (i, 'a') for i in range(nr_nodes_1) ) )
    label_2 = dict( ( (i, 'a') for i in range(nr_nodes_2) ) )


            
    # sample binomial graphs
    G1 = generateSBM(n, pi, p1, label_1, nr_nodes_1)
    G2 = generateSBM(m, pi, p2, label_2, nr_nodes_2)
    # Gs = mg.generateSBM2(n, sizes, p1, label_1)
    # G2 = mg.generateSBM2(m, sizes, p2, label_2)
    Gs = G1 + G2


    graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')
    print(datetime.now() - time)           
    # Fit a kernel
    time = datetime.now()
    K = KernelMatrix(graph_list, kernel, True)



    list_of_functions = [MMD_b, MMD_u]
    function_arguments=[dict(n = n, m = m ), dict(n = n, m = m )]

    hypothesis = BoostrapMethods(list_of_functions)
    time = datetime.now()
    hypothesis.Bootstrap(K,function_arguments, B = 1000)
    print(datetime.now() - time)

    time = datetime.now()
    hypothesis.Bootstrap(K,function_arguments, B = 1000)
    print(datetime.now() - time)
    # print(hypothesis.p_values)
    # print(hypothesis.sample_test_statistic)

        # sample binomial graphs
    G1 = generateSBM(n, pi, p1, label_1, nr_nodes_1)
    G2 = generateSBM(m, pi, p2, label_2, nr_nodes_2)
    # Gs = mg.generateSBM2(n, sizes, p1, label_1)
    # G2 = mg.generateSBM2(m, sizes, p2, label_2)
    Gs = G1 + G2


    graph_list = gk.graph_from_networkx(Gs, node_labels_tag='label')          
    # Fit a kernel
    K = KernelMatrix(graph_list, kernel, True)
    
    time = datetime.now()
    hypothesis.Bootstrap(K,function_arguments, B = 1000)
    print(datetime.now() - time)



    # list_of_functions = [average_degree, median_degree, avg_neigh_degree, avg_clustering, transitivity]
    # hypothesis_graph_statistic = BootstrapGraphStatistic(G1,G2, list_of_functions)
    # hypothesis_graph_statistic.Bootstrap(B = 100)
    # print(hypothesis_graph_statistic.p_values)
    # print(hypothesis_graph_statistic.sample_test_statistic)
    # print(hypothesis_graph_statistic.boot_test_statistic)




