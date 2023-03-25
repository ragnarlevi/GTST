



import numpy as np
import warnings
import networkx as nx
from GTST.MONK import MMD_MONK
import tqdm

from GTST.kernels import RandomWalk, WWL, GNTK, DeepKernel
import GTST.glasso as glasso

# Biased empirical maximum mean discrepancy
def MMD_b(K: np.array, n1: int, n2: int):
    '''
    Biased empirical maximum mean discrepancy

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)
    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy
    
    '''

    if (n1 + n2) != K.shape[0]:
        raise ValueError("n + m have to equal the size of K")

    Kx = K[:n1, :n1]
    Ky = K[n1:, n1:]
    Kxy = K[:n1, n1:]
    
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n1 ** 2) * Kx.sum() + 1.0 / (n1 * n2) * Ky.sum() - 2.0 / (n2 ** 2) * Kxy.sum()

# Unbiased empirical maximum mean discrepancy
def MMD_u(K: np.array, n1: int, n2: int):
    '''
    Unbiased empirical maximum mean discrepancy

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)

    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy
    
    '''

    if (n1 + n2) != K.shape[0]:
        raise ValueError("n + m have to equal the size of K")
    Kx = K[:n1, :n1]
    Ky = K[n1:, n1:]
    Kxy = K[:n1, n1:]
    # important to write 1.0 and not 1 to make sure the outcome is a float!
    return 1.0 / (n1* (n1 - 1.0)) * (Kx.sum() - np.diag(Kx).sum()) + 1.0 / (n2 * (n2 - 1.0)) * (Ky.sum() - np.diag(Ky).sum()) - 2.0 / (n1 * n2) * Kxy.sum()

def MONK_EST(K, Q, n1, n2):
    """
    Wrapper for MONK class

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)

    Q: int,
        Number of partitions for each sample.

    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy


    """

    mmd =  MMD_MONK(Q = Q,  K=K)
    return mmd.estimate(n1, n2)

def MMD_l(K: np.array, n1: int, n2: int) -> float:
    '''
    Unbiased estimate of the maximum mean discrepancy using fewer Kernel evaluations

    Parameters
    ---------------------
    K: np.array,
        Kernel matrix size K (n+m) x (n+m)
    n1: int,
        Number of observations in sample 1

    n2: int,
        Number of observations in sample 1

    Returns
    ----------------------------
    float
        Unbiased estimate of the maximum mean discrepancy using fewer Kernel evaluations
    
    '''

    assert n1 == n2, "n has to be equal to m"

    Kxx = K[:n1,:n1]
    Kyy = K[n1:,n1:]
    Kxy = K[:n1,n1:]
    Kyx = K[n1:,:n1]

    if n1 %2 != 0:
        n1 = n1-1
    return np.mean(Kxx[range(0,n1-1,2), range(1,n1,2)]) +\
            np.mean(Kyy[range(0,n1-1,2), range(1,n1,2)]) -\
            np.mean(Kxy[range(0,n1-1,2), range(1,n1,2)]) -\
            np.mean(Kyx[range(0,n1-1,2), range(1,n1,2)])



class BoostrapMethods():
    """
    Various Bootstrap/Permutation Functions

    Class for permutation testing
    """

    def __init__(self, list_of_functions:list,function_arguments:dict,) -> None:
        """

        Parameters
        ---------------------------
        list_of_functions: list,
            list containing function that represent a MMD calculation

        function_arguments: dict
            dict of dictionaries with inputs for its respective function in list_of_functions,  excluding K. 
            The key of function_arguments should be the name of the functions in list_of_functions

        """

        self.list_of_functions = list_of_functions
        self.function_arguments = function_arguments

    @staticmethod
    def issymmetric(a, rtol:float=1e-05, atol:float=1e-08) -> bool:
        """
        Check if matrix is symmetric
        """
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    def PermutationScheme(K:np.array) -> np.array:
        """
        Permutate matrix without replacement

        Parameters
        ------------------
        :param K: numpy array
            Kernel Matrix
        """
        index = np.random.permutation(K.shape[0])
        return K[np.ix_(index,index)]

    @staticmethod
    def BootstrapScheme(K:np.array) -> np.array:
        """
        Permutate matrix with replacement

        Parameters
        ------------------
        :param K: numpy array
            Kernel Matrix
        """
        index = np.random.choice(K.shape[0], size = K.shape[0])
        return K[np.ix_(index,index)]

    
    def Bootstrap(self, K:np.array, B:int, method:str = "PermutationScheme", check_symmetry:bool = False, verbose:bool = False) -> None:
        """

        Permutate/bootstrap to estimate the p-value of a mmd test

        Parameters
        --------------------
        K: numpy array
            Kernel matrix that we want to permutate
        B: int,
            Number of Bootstraps
        method: str,
            Which permutation method should be applied? Should match the function name
        check_symmetry: bool,
            Should the scheme check if the matrix is symmetirc, each time? Adds time complexity
        verbose: bool,
            Should a progress bar show the progress?
        """

        assert self.issymmetric(K), "K is not symmetric"

        # keep p-value result from each MMD function
        p_value_dict = dict()
        
        # Calculate sample mmd statistic, and create a dictionary for bootstrapped statistics
        sample_statistic = dict()
        boot_statistic = dict()
        for i in range(len(self.list_of_functions)):
            # the key is the name of the MMD (test statistic) function
            key = self.list_of_functions[i].__name__
            sample_statistic[key] =  self.list_of_functions[i](K, **self.function_arguments[key]) 
            boot_statistic[key] = np.zeros(B)


        # Get bootstrap evaluation method
        evaluation_method = getattr(self, method)

        if verbose:
            pbar = tqdm.tqdm(total = B)
        # Now Perform Bootstraping
        for boot in range(B):
            K_i = evaluation_method(K)
            if check_symmetry:
                if self.issymmetric(K_i):
                    warnings.warn("Not a Symmetric matrix", Warning)

            # apply each test defined in list_if_functions, and keep the bootstraped/permutated value
            for i in range(len(self.list_of_functions)):
                key = self.list_of_functions[i].__name__
                boot_statistic[key][boot] = self.list_of_functions[i](K_i, **self.function_arguments[key])

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        # calculate p-value
        for key in sample_statistic.keys():
            p_value_dict[key] =  (boot_statistic[key] >= sample_statistic[key]).sum()/float(B)

        self.p_values = p_value_dict
        self.sample_test_statistic = sample_statistic
        self.boot_test_statistic = boot_statistic




class MMD():
    """
    Class to calculate Graph MMD. Either real-data which contains 
    
    """

    def __init__(self) -> None:
        self.G1 = None
        self.G2 = None

    @staticmethod
    def label_fun(G, s):
        """"
        G: networkx graph,
        s: str
        """

        if s == "degree":
            nodes_degree = dict(G.degree)
            labels = {key: str(value) for key, value in nodes_degree.items()} 
        else:
            raise ValueError(f'Labelling scheme {s} not defined')

        return labels


    def get_mmd_func_pair(self, estimator, Q = None):
        """
        estimator: str,
            which estimator to use
        Q: int,
            Number of partitions for MONK
        
        """

        if estimator == 'MMD_u':
            ret = (MMD_u, {'MMD_u':dict(n1=self.n1, n2=self.n2)})
        elif estimator == 'MMD_l':
            ret = (MMD_l, {'MMD_l':dict(n1=self.n1, n2=self.n2)})
        elif estimator == 'MMD_b':
            ret = (MMD_b, {'MMD_b':dict(n1=self.n1, n2=self.n2)})
        elif estimator == 'MONK_EST':
            if Q is None:
                raise ValueError("Q has to be given for MONK estimator")
            ret = (MONK_EST, {'MONK_EST':dict(Q = Q, n1=self.n1, n2=self.n2)})
        else:
            raise ValueError(f"No estimator called {estimator}")
        
        return ret
        
    def estimate_graphs(self,X1, X2, window_size:int,alpha:float, beta:float, nonparanormal:bool = False, scale:bool = False, set_attributes = None, set_labels = None ):
        """
        X1,X2: araray like,
            real data for sample 1 and 2
        window_size: int,
            number of data points to estimate each graph
        alpha: list, float,
            Regularization parameters in a list or a single float
        beta: float:
            EBIC hyperparameter
        nonparanormal: bool
            Should data be nonparanormally transformed
        scale:bool,
            Should data be scaled
        get_attribute: function or None,
            Set attribute of nodes. If a function should take numpy array  (which will be a submatrix of X1,X2) as input and output attributes for each node/parameter, used as an attribute for a graph kernel.
        set_labels: function, str or None:
            Set labels of nods: If a function should take networkx graph as input and output labels for each node/parameter.  Should return a dict {node_i:label,..}
            If string should be degree and the labels will be labeled as degree
            If None no labelling

        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.nonparanormal = nonparanormal
        self.scale = scale
        
        # store if attributes, labels are being set
        if set_attributes is not None:
            self.node_attr = 'attr'
        
        if set_labels is not None:
            self.node_label = 'label'

        # weight names are set with this method
        self.weight_name = 'weight'



        if X1.shape[0] != X2.shape[0]:
            warnings.warn("X1 and X2 have different number of data points")
        if X1.shape[1] != X2.shape[1]:
            warnings.warn("X1 and X2 have different number of nodes, be careful of interpretation")
        if beta <0 or beta >1:
            raise ValueError("beta should be in the interval [0,1]")

        self.G1 = []
        self.G2 = []

         # Estimate precision 1
        for i in range(window_size,X1.shape[0]+1,window_size):
            # Extract data points used to estimate a precision matrix
            tmp_X1= X1[(i-window_size):(i)].copy()

            if tmp_X1.shape[0] != window_size:
                break
        
            info_G1 = glasso.glasso_wrapper(alpha =self.alpha, beta = self.beta, nonparanormal=self.nonparanormal, scale = self.scale ).fit(tmp_X1)
            precision1 = info_G1.precision_.copy()
            np.fill_diagonal(precision1,0)
            G1i = nx.from_numpy_array(precision1)
            if callable(set_attributes):
                attributes_i = set_attributes(tmp_X1)
                nx.set_node_attributes(G1i, {k:attributes_i[k] for k in range(len(attributes_i))}, 'attr')
            if set_labels is not None:
                if type(set_labels) == str:
                    labels1 = self.label_fun(G1i, set_labels)
                    nx.set_node_attributes(G1i, labels1, 'label')
                elif callable(set_labels):
                    labels1 = set_labels(tmp_X1)
                    nx.set_node_attributes(G1i, labels1, 'label')
                elif type(set_labels) == dict:
                    nx.set_node_attributes(G1i, set_labels['1'], 'label')
                else:
                    raise ValueError(f"labelling not defined for type {type(set_labels)}")
            self.G1.append(G1i)

        # Estimate Precision 2
        for i in range(window_size,X2.shape[0]+1,window_size):
            # Extract data points used to estimate a precision matrix
            tmp_X2= X2[(i-window_size):(i)].copy()

            if tmp_X2.shape[0] != window_size:
                break
             # Estimate precision2
            info_G2 = glasso.glasso_wrapper(alpha =self.alpha, beta = self.beta, nonparanormal=self.nonparanormal, scale = self.scale ).fit(tmp_X2)
            precision2 = info_G2.precision_.copy()
            np.fill_diagonal(precision2,0)
            G2i = nx.from_numpy_array(precision2)
            if callable(set_attributes):
                attributes_i = set_attributes(tmp_X2)
                nx.set_node_attributes(G2i, {k:attributes_i[k] for k in range(X2.shape[1])}, 'attr')
            if set_labels is not None:
                if type(set_labels) == str:
                    labels2 = self.label_fun(G2i, set_labels)
                    nx.set_node_attributes(G2i, labels2, 'label')
                elif callable(set_labels):
                    labels2 = set_labels(tmp_X2)
                    nx.set_node_attributes(G2i, labels2, 'label')
                elif type(set_labels) == dict:
                    nx.set_node_attributes(G2i, set_labels['2'], 'label')
                else:
                    raise ValueError(f"labelling not defined for type {type(set_labels)}")
            self.G2.append(G2i)

        return self

    def fit(self, kernel, mmd_estimators, G1 = None, G2 = None, **kwargs):

        """
        Parameters
        --------------------
        Run a MMD test and estimate p-val
        G1, G2: list of networkx graphs or None.
            If None than estimate_graphs has to been called first (which estimates graphs from numpy arrays)
        kernel: list, str, np.array,
            If str then one of kernels provided by the MMDGraph: RW_ARKU_plus, RW_ARKU, RW_ARKU_edge, RW_ARKL, GNTK, WWL, DK
            If list then a GraKel kernel is used so the list should be formatted as one would do for the GraKel package
            If np.array then a pre-calculate kernel is used.
        mmd_estimators:str or list of strings, 
            example MMD_u, MMD_b, MONK, MMD_l (or own defined function)
        
        **kwargs: Additional argumtens to kernel function and bootstrap function

        Notes
        -----------------

        """

        # I do not want to overwrite these attributes if estimate_graphs has been called which creates these attributes
        # so I carefully include these if sentences
        if kwargs.get('edge_attr', None) is not None:
            self.edge_attr = kwargs.get('edge_attr', None)
        if kwargs.get('node_attr', None) is not None:
            self.node_attr = kwargs.get('node_attr', None)
        if kwargs.get('node_label', None) is not None:
            self.node_label = kwargs.get('node_label', None)
        if kwargs.get('edge_label', None) is not None:
            self.edge_label = kwargs.get('edge_label', None)



        if hasattr(self, 'edge_attr'):
            print(f'Using {self.edge_attr} as edge attributes')
        if hasattr(self, 'node_attr'):
            print(f'Using {self.node_attr} as node attributes')
        if hasattr(self, 'node_label'):
            print(f'Using {self.node_label} as node labels')
        if hasattr(self, 'edge_label'):
            print(f'Using {self.edge_label} as edge labels')
  
        


        # Check inputs
        if G1 is None and self.G1 is None:
            raise ValueError("G1 can not be None if estimate_graphs has not been called. Either input list of graphs G1, G2 or an array of data via estimate_graphs ")

        if G1 is not None and self.G1 is not None:
            warnings.warn(f'estimate_graphs has been called but user is also inputing G1. G1 will overwrite the estimated graphs, also be careful that edge_attr, edge_label, node_label, node_attr are correctly set ')

        if self.G1 is None:
            self.G1 = G1
            self.G2 = G2

        
        self.n1 = len(self.G1)
        self.n2 = len(self.G2)

        if self.n1 <2:
            raise ValueError("length of G1 has to be bigger than or equal to 2")
        if self.n2 <2:
            raise ValueError("length of G2 has to be bigger than or equal to 2")

        # Set up bootstrap
        if type(mmd_estimators) == str:
            mmd_estimators = [mmd_estimators]

        function_arguments = dict()
        list_of_functions = []
        for estimator in mmd_estimators:
            fun, fun_args = self.get_mmd_func_pair(estimator, Q= kwargs.get('Q',None))
            list_of_functions.append(fun)
            function_arguments[estimator] = fun_args[estimator]


        pval = BoostrapMethods(list_of_functions=list_of_functions, function_arguments=function_arguments)

        # Caluclate kernel
        if kernel == "RW_ARKU_plus":
            rw_kernel = RandomWalk.RandomWalk(self.G1+self.G2, r = kwargs['r'],c = kwargs['c'], 
                                                edge_attr=getattr(self, 'edge_attr', None), 
                                                normalize=kwargs.get('normalize',False), 
                                                node_attr= getattr(self, 'node_attr', None))
            rw_kernel.fit(calc_type = "ARKU_plus",  
                            verbose = kwargs.get('verbose',False), 
                            check_psd = kwargs.get('check_psd',True))

            self.K = rw_kernel.K
            del rw_kernel
        elif kernel == "RW_ARKU":
            rw_kernel = RandomWalk.RandomWalk(self.G1+self.G2, r = kwargs['r'],c = kwargs['c'], 
                                                edge_attr=getattr(self, 'edge_attr', None), 
                                                normalize=kwargs.get('normalize',False), 
                                                node_attr= getattr(self, 'node_attr', None))
            rw_kernel.fit(calc_type = "ARKU",  
                            verbose = kwargs.get('verbose',False), 
                            check_psd = kwargs.get('check_psd',True))

            self.K = rw_kernel.K
            del rw_kernel
        elif kernel == "RW_ARKU_edge":
            rw_kernel = RandomWalk.RandomWalk(self.G1+self.G2, r = kwargs['r'],c = kwargs['c'], 
                                    edge_attr=getattr(self, 'edge_attr', None), 
                                    normalize=kwargs.get('normalize',False), 
                                    edge_label = kwargs['edge_label'],
                                    unique_edge_labels=kwargs['unique_edge_labels'])
            rw_kernel.fit(calc_type = "ARKU_edge",  
                            verbose = kwargs.get('verbose',False), 
                            check_psd = kwargs.get('check_psd',True))

            self.K = rw_kernel.K
            del rw_kernel
        elif kernel == "RW_ARKL":
            rw_kernel = RandomWalk.RandomWalk(self.G1+self.G2, r = kwargs['r'],c = kwargs['c'], 
                                    edge_attr=getattr(self, 'edge_attr', None), 
                                    normalize=kwargs.get('normalize',False), 
                                    node_label = kwargs['node_label'],
                                    unique_node_labels=kwargs['unique_node_labels'])
            rw_kernel.fit(calc_type = "ARKL",  
                            verbose = kwargs.get('verbose',False), 
                            check_psd = kwargs.get('check_psd',True))

            self.K = rw_kernel.K
            del rw_kernel
        elif kernel == "GNTK":
            gntk_kernel = GNTK.GNTK(num_layers= kwargs['num_layers'], num_mlp_layers = kwargs['num_layers'], 
                                        jk = kwargs['jk'], scale = kwargs['scale'], normalize=kwargs.get('normalize',False))
            
            gntk_kernel.fit_all(self.G1+self.G2,degree_as_tag=kwargs.get('degree_as_tag',True),features= getattr(self, 'node_attr', None))
            self.K = gntk_kernel.K
            del gntk_kernel
        elif kernel == "WWL":
            wwl_kernel = WWL.WWL( param =dict(discount=kwargs['discount'], h=kwargs['h'], sinkhorn=kwargs.get('sinkhorn', False), 
                                    sinkhorn_lambda = kwargs.get('sinkhorn_lambda', 1), normalize = kwargs.get('normalize',False)),
                                    label_name=self.node_label)
            self.K = wwl_kernel.fit_transform(self.G1+self.G2)
            del wwl_kernel
        elif kernel == 'DK':
            dk_kernel = DeepKernel.DK(type = kwargs['type'], wl_it = kwargs.get('wl_it',2),opt_type= kwargs.get('opt_type',None),
                                      vector_size= kwargs.get('vector_size',2), window=kwargs.get('window',2), min_count=kwargs.get('min_count',0), 
                                      normalize = kwargs.get('normalize',False), nodel_label=self.node_label)
            self.K = dk_kernel.fit_transform(self.G1+self.G2)
            del dk_kernel
        elif type(kernel) == list:
            import grakel as gk
            # Grakel kernels
            # if there are attributes they will overwrite label_name
            if getattr(self, 'node_attr', None) is not None: 
                    node_labels_tag = self.node_attr
            elif hasattr(self, 'node_label'):
                node_labels_tag = self.node_label
            else:
                node_labels_tag = None
            print(node_labels_tag)
            init_kernel = gk.GraphKernel(kernel= kernel, normalize=kwargs.get('normalize',False))
            graph_list = gk.graph_from_networkx(self.G1+self.G2, node_labels_tag=node_labels_tag, 
                                                    edge_weight_tag=getattr(self, 'edge_attr', None), 
                                                    edge_labels_tag=getattr(self, 'edge_label', None))
            self.K = init_kernel.fit_transform(graph_list)
            del init_kernel


        elif type(kernel) == np.ndarray:
            self.K = kernel
        else: 
            raise ValueError(f'{kernel} not defined ')



        pval.Bootstrap(self.K, kwargs.get('B',1000))
        self.p_values = pval.p_values
        self.sample_mmd = pval.sample_test_statistic











                
            




            



