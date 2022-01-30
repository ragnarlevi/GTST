
import networkx as nx
import numpy as np
import grakel as gk
import warnings
from scipy.stats import norm


import pandas as pd
#import time


from numba import njit
from scipy.sparse.sputils import validateaxis

import MONK

# Biased empirical maximum mean discrepancy
def MMD_b(K: np.array, n: int, m: int):

    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n ** 2) * Kx.sum() + 1.0 / (n * m) * Ky.sum() - 2.0 / (m ** 2) * Kxy.sum()

# Unbiased empirical maximum mean discrepancy
def MMD_u(K: np.array, n: int, m: int):
    Kx = K[:n, :n]
    Ky = K[n:, n:]
    Kxy = K[:n, n:]
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n* (n - 1.0)) * (Kx.sum() - np.diag(Kx).sum()) + 1.0 / (m * (m - 1.0)) * (Ky.sum() - np.diag(Ky).sum()) - 2.0 / (n * m) * Kxy.sum()

def MONK_EST(K, Q, y1, y2):
    """
    Wrapper for MONK
    """

    mmd =  MONK.MMD_MOM(Q = Q, kernel_type = 'matrix', kernel=K)
    return mmd.estimate(y1, y2)



def factorial_k(m, k):
    """
    Calculate (m)_k := m(m-1)*...* (m-k+1)
    
    """
    
    base = m
    for i in range(1,k):
        m *= base - i
    return m


def power_ratio(K, mmd_stat, threshold, m):
    """
    The function calculates the power ratio of a specific kernel

    Parameters
    --------------------
    K: numpy array, The ernel matrix
    mmd_stat: float, the sample mmd value. Note this is the squared unbiased mmd value
    threshold: float, the test thershold for a given type I error (alpha value)
    m: int, size of samples

    Returns
    --------------------
    ratio: float, the power ratio, the bigger the better
    power: float, The power of the test
    


    Variance is found in Unbiased estimators for the variance of MMD estimators Danica J. Sutherland https://arxiv.org/pdf/1906.02104.pdf
    """

    Kxx = K[:m, :m]
    Kxy = K[:m, m:]
    Kyy = K[m:, m:]

    Ktxx = Kxx.copy()
    Ktxy = Kxy.copy()
    Ktyy = Kyy.copy()
    np.fill_diagonal(Ktxx, 0)
    np.fill_diagonal(Ktxy, 0)
    np.fill_diagonal(Ktyy, 0)



    e = np.ones(m)
    # Calculate variance
    V = (
         (4/factorial_k(m, 4)) * (np.inner(np.matmul(Ktxx,e),np.matmul(Ktxx,e))  + np.inner(np.matmul(Ktyy,e),np.matmul(Ktyy,e)) )
        + ((4*(m**2 - m - 1)) / (m**3 * (m-1)**2)) * (np.inner(np.matmul(Kxy,e),np.matmul(Kxy,e))  + np.inner(np.matmul(Kxy.T,e),np.matmul(Kxy.T,e)) )
        - (8 / ((m**2) * (m**2 - 3*m + 2))) * (np.dot(e, Ktxx).dot(Kxy).dot(e) + np.dot(e, Ktyy).dot(Kxy.T).dot(e))
        + (8 / (m**2 * factorial_k(m, 3))) * ((np.dot(e, Ktxx).dot(e) + np.dot(e, Ktyy).dot(e))*np.dot(e,Kxy).dot(e))
        - ((2*(2*m-3))/(factorial_k(m,2)*factorial_k(m,4))) * (np.dot(e, Ktxx).dot(e)**2 + np.dot(e, Ktyy).dot(e)**2)
        - ((4*(2*m-3))/(m**3 * (m-1)**3)) * np.dot(e,Kxy).dot(e)**2
        - (2/(m *(m**3 - 6*m**2 +11*m -6))) *(np.linalg.norm(Ktxx, ord = 'fro')**2 + np.linalg.norm(Ktyy, ord = 'fro')**2)
        + ((4*(m-2))/(m**2 * (m-1)**3)) * np.linalg.norm(Kxy, ord='fro')**2
    )

    
    # K_XX = K[:m, :m]
    # K_XY = K[:m, m:]
    # K_YY = K[m:, m:]



    # diag_X = np.diag(K_XX)
    # diag_Y = np.diag(K_YY)

    # sum_diag_X = diag_X.sum()
    # sum_diag_Y = diag_Y.sum()

    # sum_diag2_X = diag_X.dot(diag_X)
    # sum_diag2_Y = diag_Y.dot(diag_Y)

    # Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    # Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    # K_XY_sums_0 = K_XY.sum(axis=0)
    # K_XY_sums_1 = K_XY.sum(axis=1)

    # Kt_XX_sum = Kt_XX_sums.sum()
    # Kt_YY_sum = Kt_YY_sums.sum()
    # K_XY_sum = K_XY_sums_0.sum()

    # Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    # Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    # K_XY_2_sum  = (K_XY ** 2).sum()


    # mmd2 = (Kt_XX_sum / (m * (m-1))
    #         + Kt_YY_sum / (m * (m-1))
    #         - 2 * K_XY_sum / (m * m))

    # V = (
    #       2 / (m**2 * (m-1)**2) * (
    #           2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
    #         + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
    #     - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
    #     + 4*(m-2) / (m**3 * (m-1)**2) * (
    #           K_XY_sums_1.dot(K_XY_sums_1)
    #         + K_XY_sums_0.dot(K_XY_sums_0))
    #     - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
    #     - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
    #     + 8 / (m**3 * (m-1)) * (
    #           1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    #         - Kt_XX_sums.dot(K_XY_sums_1)
    #         - Kt_YY_sums.dot(K_XY_sums_0))
    # )

    ratio = (mmd_stat / np.sqrt(V)) - (threshold/(m*np.sqrt(V)))
    power = norm.cdf(ratio)

    return ratio, power, V





    





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
        index = np.random.choice(K.shape[0], size = K.shape[0])
    
        for i in range(len(index)):
            for j in range(len(index)):
                K_i[i,j] = K[index[i], index[j]]

        return K_i


    def Bootstrap(self, K, function_arguments,B:int, method:str = "PermutationScheme", check_symmetry:bool = False) -> None:
        """
        :param K: Kernel matrix that we want to permutate
        :param function_arguments: List of dictionaries with inputs for its respective function in list_of_functions,  excluding K. If no input set as None.
        :param B: Number of Bootstraps
        :param method: Which permutation method should be applied?
        :param check_symmetry: Should the scheme check if the matrix is symmetirc, each time? Adds time complexity
        """
        self.K = K.copy()
        #print(self.K)
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

            sample_statistic[key] =  self.list_of_functions[i](K, **self.function_arguments[i]) #eval(eval_string)
            boot_statistic[key] = np.zeros(B)



        # Now Perform Bootstraping
        for boot in range(B):
            K_i = evaluation_method(self.K)
            if check_symmetry:
                if self.issymmetric(K_i):
                    warnings.warn("Not a Symmetric matrix", Warning)

            # apply each test defined in list_if_functions, and keep the bootstraped/permutated value
            for i in range(len(self.list_of_functions)):
                eval_string = self.list_of_functions[i].__name__ + "(K =K_i, " + inputs[i] + ")"
                boot_statistic[self.list_of_functions[i].__name__][boot] = self.list_of_functions[i](K_i, **self.function_arguments[i])#eval(eval_string)

        # calculate p-value
        for key in sample_statistic.keys():
            p_value_dict[key] =  (boot_statistic[key] >= sample_statistic[key]).sum()/float(B)
            #print(f' boot_stat {boot_statistic[key][:10]}')
            #print(f' sample_statistic {sample_statistic[key]}')


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
            p_value_dict[key] =  2*np.min([(boot_test_statistic[key] >= sample_test_statistic[key]).sum(),(boot_test_statistic[key] <= sample_test_statistic[key]).sum()])/float(B)


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
    Wrapper for a graph generator. Defines the labels and attribute generation. Used as a parent class.
    """

    def __init__(self, n, nnode, k = None, l = None,  a = None, fullyConnected = False, **kwargs) -> None:
        """
        :param kernel: Dictionary with kernel information
        :param n: Number of samples
        :param nnode: Number of nodes
        :param k: Degree
        :param l: Labelling scheme
        :param a: Attribute scheme
        :param **kwargs: Arguments for the labelling/attribute functions
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

    def samelabels_float(self, G):
        """
        labelling Scheme. All nodes get same label

        :param G: Networkx graph
        """
        return dict( ( (i, 0.0) for i in range(len(G)) ) )

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
        labelling Scheme. Nodes labelled according to a discrete pmf

        :param G: Networkx graph
        :param pmf: pmf as list. If None then uniform over all entries
        :return: Dictionary
        """
        import string
        assert not self.kwargs['nr_letters'] is None, "Number of letters (nr_letters) has to be specified"
        

        # check if the pmf of labels has been given
        if not 'pmf' in self.kwargs.keys():
            pmf = None
        else:
            pmf = self.kwargs['pmf']
            assert  np.sum(self.kwargs['pmf']) >0.999, "pmf has to sum to 1"

        letters = list(string.ascii_lowercase[:self.kwargs['nr_letters']])
        return dict( ( (i, np.random.choice(letters, p = pmf)) for i in range(len(G)) ) )

    def normal_conditional_on_latent_mean_rv(self, G):
        """
        Generate a random variable where the mean follows another normal distribution
        scale: sd of normal
        scale_latent: sd of the latent normal
        """

        loc = np.random.normal(self.kwargs.get('loc_latent', 1), self.kwargs.get('scale_latent', 1), size = self.nnode)
        scale = self.kwargs.get('scale', 1)
        return dict( ( (i, np.random.normal(loc = loc[i], scale = scale, size = (1,))) for i in range(len(G)) ) )

    def edges(self, G):
        """
        concatenate edge labels of node and set it as the node label
        """

        return dict(( (i, ''.join(map(str,sorted([ info[2] for info in G.edges(i, data = 'sign')]))) ) for i in range(len(G))))


def scale_free(n, exponent):
    """
    Parameters:
    -------------------
    n - number of nodes
    exponent - power law exponent
    
    """
    while True:  
        s=[]
        while len(s)<n:
            nextval = int(nx.utils.powerlaw_sequence(1, exponent)[0]) #100 nodes
            if nextval!=0:
                s.append(nextval)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G = nx.Graph(G) # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


class ScaleFreeGraph(DegreeGraphs):
    """
    Generate a powerlaw graph
    """

    def __init__(self,  n, nnode, exponent, l = None,  a = None, **kwargs):
        """
        Parameters:
        ---------------------
        n - number of samples
        nnode - number of nodes


        balance_target - ratio of balanced triangles to unbalanced ones.
        exponent - power law exponent

        
        """
        super().__init__( n = n, nnode = nnode, l = l ,  a = a, **kwargs )

        self.exponent = exponent


    def Generate(self):

        self.Gs = []

        for _ in range(self.n):
            
            G = scale_free(self.nnode, self.exponent)

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





class SignedGraph(DegreeGraphs):
    """
    Generate a powerlaw or erdos_renyi graph with signed edges
    """

    def __init__(self,  n, nnode, l = None,  a = None, **kwargs):
        """
        Parameters:
        ---------------------
        n - number of samples
        nnode - number of nodes

        **kwargs
            balance_target - ratio of balanced triangles to unbalanced ones.
            exponent - power law exponent
            powerlaw - boolean
            k - degree if graph generation is not scale free
        
        """
        super().__init__( n = n, nnode = nnode, l = l ,  a = a, **kwargs )

        self.balance_target = kwargs.get('balance_target',1.0)
        self.exponent = kwargs.get('exponent',1.0)
        self.powerlaw = kwargs.get('powerlaw',False)
        self.k = kwargs.get('k',5)

    def Generate(self):

        self.Gs = []
        for _ in range(self.n):
            
            G = self.generate_single_Graph()

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


    def generate_single_Graph(self):
        
        nr_triangles = 0
        while nr_triangles <= 0:
            
            if self.powerlaw:
                G = scale_free(self.nnode, self.exponent)
            else:
                G = nx.erdos_renyi_graph(self.nnode, self.k / (self.nnode - 1))
                
            nr_triangles = len([c for c in nx.cycle_basis(G) if len(c)==3])

        # Label all edges with -, so balance ratio = 1
        nx.set_edge_attributes(G, {(n1, n2): np.random.choice([-1,1]) for n1, n2 in G.edges()}, "sign")

        cnt_balanced, cnt_unbalanced = self.cnt_balance(G)


        if self.balance_target + 0.05 > cnt_balanced /(cnt_balanced + cnt_unbalanced) > self.balance_target - 0.05:
            pass
        elif cnt_balanced /(cnt_balanced + cnt_unbalanced) > self.balance_target:
            G = self.balance_down(G, self.balance_target, cnt_balanced, cnt_unbalanced)
        else:
            G = self.balance_up(G, self.balance_target, cnt_balanced, cnt_unbalanced)

        return G

    def cnt_balance(self,G):
        # Loop thorugh triangles and set as unbalanced until under certain threshold
        triangles = [c for c in nx.cycle_basis(G) if len(c)==3]
        cnt_unbalanced = 0
        cnt_balanced = 0
        for cycle in triangles:
            
            # This cycle is balanced
            if G.edges[cycle[0],cycle[1]]['sign']*G.edges[cycle[1],cycle[2]]['sign']*G.edges[cycle[0],cycle[2]]['sign'] == 1.0:
                cnt_balanced += 1
            else:
                cnt_unbalanced +=1


        return cnt_balanced, cnt_unbalanced


    def balance_up(self, G, balance_target, cnt_balanced0, cnt_unbalanced0):

        # Loop thorugh triangles and set as unbalanced until under certain threshold
        triangles = [c for c in nx.cycle_basis(G) if len(c)==3]
        
        balance_ratio = cnt_balanced0/(cnt_unbalanced0 + cnt_balanced0)
        cnt_balanced = cnt_balanced0
        cnt_unbalanced =cnt_unbalanced0

        itr = 0
        while (balance_ratio + 0.05 <= balance_target) and itr < 1000:
            itr += 1
            
            for node in np.random.permutation(G.nodes()):

                # find all cycles rooted at this node
                triangles = [c for c in nx.cycle_basis(G, root = node) if len(c)==3]

                if len(triangles) <= 0:
                    continue

                # randomly pick one triangle
                tri_pick = np.random.choice(len(triangles))
                cycle = triangles[tri_pick]

                # is this triangle unbalanced?
                if G.edges[cycle[0],cycle[1]]['sign']*G.edges[cycle[1],cycle[2]]['sign']*G.edges[cycle[0],cycle[2]]['sign'] == -1.0:

                    perm = np.random.permutation([0,1,2])
                
                    if G.edges[cycle[perm[0]],cycle[perm[1]]]['sign'] == -1.0:
                        nx.set_edge_attributes(G, {(cycle[perm[0]], cycle[perm[1]]): 1.0}, "sign")
                    elif G.edges[cycle[perm[2]],cycle[perm[1]]]['sign'] == -1.0:
                        nx.set_edge_attributes(G, {(cycle[perm[2]], cycle[perm[1]]): 1.0}, "sign")
                    else:
                        nx.set_edge_attributes(G, {(cycle[perm[0]], cycle[perm[2]]): 1.0}, "sign")

                    cnt_balanced, cnt_unbalanced = self.cnt_balance(G)
                    balance_ratio = cnt_balanced/(cnt_unbalanced + cnt_balanced)
                    break

                else:
                    continue
        
        return G

    def balance_down(self, G, balance_target, cnt_balanced0, cnt_unbalanced0):

        # Loop thorugh triangles and set as unbalanced until under certain threshold
        triangles = [c for c in nx.cycle_basis(G) if len(c)==3]
        
        balance_ratio = cnt_balanced0/(cnt_unbalanced0 + cnt_balanced0)
        cnt_balanced = cnt_balanced0
        cnt_unbalanced =cnt_unbalanced0

        itr = 0
        while (balance_ratio - 0.05 >= balance_target) and itr < 1000:
            itr += 1
            
            for node in np.random.permutation(G.nodes()):

                # find all cycles rooted at this node
                triangles = [c for c in nx.cycle_basis(G, root = node) if len(c)==3]

                if len(triangles) <= 0:
                    continue

                # randomly pick one triangle
                tri_pick = np.random.choice(len(triangles))
                cycle = triangles[tri_pick]

                # is this triangle unbalanced?
                if G.edges[cycle[0],cycle[1]]['sign']*G.edges[cycle[1],cycle[2]]['sign']*G.edges[cycle[0],cycle[2]]['sign'] == 1.0:

                    perm = np.random.permutation([0,1,2])
                
                    if G.edges[cycle[perm[0]],cycle[perm[1]]]['sign'] == 1.0:
                        nx.set_edge_attributes(G, {(cycle[perm[0]], cycle[perm[1]]): -1.0}, "sign")
                    elif G.edges[cycle[perm[2]],cycle[perm[1]]]['sign'] == 1.0:
                        nx.set_edge_attributes(G, {(cycle[perm[2]], cycle[perm[1]]): -1.0}, "sign")
                    else:
                        nx.set_edge_attributes(G, {(cycle[perm[0]], cycle[perm[2]]): -1.0}, "sign")

                    cnt_balanced, cnt_unbalanced = self.cnt_balance(G)
                    balance_ratio = cnt_balanced/(cnt_unbalanced + cnt_balanced)
                    break

                else:
                    continue
        
        return G



class CliqueGraph(DegreeGraphs):
    """
    a Class that generates clique graphs
    """

    def __init__(self,  n, nnode, l = None,  a = None, **kwargs):
        super().__init__( n = n, nnode = nnode, l = l ,  a = a, **kwargs )

    def Generate(self):

        self.Gs = []
        for _ in range(self.n):
            
            G = nx.Graph()
            G.add_nodes_from(range(self.nnode))
            from itertools import product
            G.add_edges_from((a,b) for a,b in product(list(G.nodes()), list(G.nodes())) if a != b)

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

    def blockmean(self,G):
        """
        Attribute Scheme. Nodes get attributes according to a normal distribution with standar deviation 1. Their mean is governed by their block membership.

        Assumes that the class was initalized with list called block_mean. First item is mean of block 0, next is mean of block 1 etc..
        """

        block_mean = self.params['block_mean']
        return {v[0]:np.array([np.random.normal(block_mean[v[1]['block']])]) for v in G.nodes(data=True) }

    def blockmean2(self, G):

        label_pmf = self.params['label_pmf']
        # Mean of each node
        return {v[0]:np.array([np.random.normal(np.random.choice(self.params['block_mean'],p = label_pmf[v[1]["block"],:] ))]) for v in G.nodes(data=True)}



    def Generate(self):
        """
        :return: list of networkx graphs

        self.params should include:
            sizes: (list of ints) Sizes of blocks
            p: (list of list of floats) Element (r,s) gives the density of edges going from the nodes of group r to nodes of group s. Must match the number of groups
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



def iteration(N:int, kernel:dict, normalize:bool, MMD_functions, bg1, bg2, B:int, kernel_hypothesis, kernel_library="Grakel", node_labels_tag='label', edge_labels_tag = None, label_list = None, edge_labels = None, rw_attributes = False):
    """
    Function That generates samples according to the graph generators bg1 and bg2 and calculates graph statistics

    Parameters
    --------------------------------
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
    :param label_list, label list for random walk kernel
    edge_labels - edge label list for random walk kernel
    rw_attributes - Only used for random walk kernel and attributed graphs (one attribute)

    Returns
    -------------------------------
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

        if sample % 10 == 0:
            print(f'{sample} ')
    
        # sample binomial graphs
        bg1.Generate()
        bg2.Generate()
        Gs = bg1.Gs + bg2.Gs
        graph_list = gk.graph_from_networkx(Gs, node_labels_tag = node_labels_tag, edge_labels_tag = edge_labels_tag)

        # calculate label_list
        if (kernel_library == 'randomwalk') and (label_list is None) and (kernel['calc_type'] == 'ARKL'):
            label_list = []
            for G in Gs:
                label_list.append(np.unique(list(nx.get_node_attributes(G, 'label').values())))
            label_list = np.unique(np.concatenate(label_list))       

        # if we are using attributes and random walk
        if rw_attributes and (kernel_library == 'randomwalk'):
            p = [np.array([i[1][0] for i in G.nodes('attr') ]) for G in Gs ]
        else:
            p = None

        
        # Kernel hypothesis testing
        # Fit a kernel, Note the Grakel uses graph_list while myKernels use Gs
        if kernel_library == "Grakel":
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=normalize)
            K = init_kernel.fit_transform(graph_list)
        elif kernel_library == "randomwalk":
            import myKernels.RandomWalk as rw
            init_kernel = rw.RandomWalk(Gs, c = kernel['c'], p = p)
            K = init_kernel.fit(calc_type = kernel['calc_type'], 
                                r = kernel['r'], 
                                k = kernel['k'],
                                mu_vec = kernel['mu_vec'],
                                normalize_adj = kernel.get('normalize_adj',False), 
                                row_normalize_adj = kernel.get('row_normalize_adj',False),
                                label_list = label_list,
                                edge_labels=edge_labels,
                                verbose = False
                                )
        elif kernel_library == "deepkernel":
            import myKernels.DeepKernel as dk
            init_kernel = dk.DK(params = kernel)
            K = init_kernel.fit_transform(Gs)
        elif kernel_library == "wwl":
            import myKernels.WWL as wl
            #import myKernels.hashkernel as hk
            init_kernel = wl.WWL(param = {'discount':kernel['discount'],'h':kernel['h'], 'sinkhorn':kernel['sinkhorn'], 'normalize':kernel['normalize']})
            K = init_kernel.fit_transform(Gs)
        elif kernel_library == "gik":
            import myKernels.GraphInvariant as gi
            init_kernel = gi.GIK(local = True, label_name = 'label', attr_name= 'attr', params = {'wl_itr':kernel['wl_itr'], 
                                                                                            'distances':kernel['distances'],  
                                                                                            'c':kernel['c'],
                                                                                            'normalize':kernel['normalize']})
            K = init_kernel.fit_transform(Gs)
        elif kernel_library == 'hash':
            import myKernels.hashkernel as hashkernel
            init_kernel = hashkernel.HashKernel(base_kernel = kernel['base_kernel'], param = {'iterations':kernel['iterations'],
                                                                                         'lsh_bin_width':kernel['lsh_bin_width'], 
                                                                                         'sigma':kernel['sigma'],
                                                                                         'normalize':kernel['normalize'],
                                                                                         'scale_attributes':kernel['scale_attributes'],
                                                                                         'attr_name': 'attr',
                                                                                         'label_name':'label',
                                                                                         'wl_iterations':kernel['wl_iterations'],
                                                                                         'normalize':kernel['normalize']})
            K = init_kernel.fit_transform(Gs)
        else:
            raise ValueError(f"{kernel_library} not defined")

        # print(K)
        Kmax[sample] = K.max()
        if np.all((K == 0)):
            warnings.warn("all element in K zero")

        #print(K)
        #print("")

        # bootstrap function argument
        function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
        
        kernel_hypothesis.Bootstrap(K, function_arguments, B = B)
        for i in range(len(MMD_functions)):
            key = MMD_functions[i].__name__
            # print(f'key {kernel_hypothesis.p_values[key]}' )
            p_values[key][sample] = kernel_hypothesis.p_values[key]
            mmd_samples[key][sample] = kernel_hypothesis.sample_test_statistic[key]

    return dict(Kmax = Kmax, p_values = p_values, mmd_samples = mmd_samples)


if __name__ == '__main__':
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 5
    m = 5

    average_degree = 6
    bg1 = BinomialGraphs(n, nr_nodes_1,average_degree, l = None)
