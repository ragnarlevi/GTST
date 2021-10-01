



import numpy as np
import networkx as nx
import scipy.sparse as sparse
from sklearn import preprocessing as pre

import itertools as it
import numpy as np
import scipy.sparse.lil as lil
import scipy as sp
import os, sys


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
import log_primes_list as log_pl

# Code taken from https://github.com/chrsmrrs/hashgraphkernel
# The code is adjusted

class HashKernel():


    def __init__(self, base_kernel, param) -> None:
        """
        :param param: iterations:int
                        lsh_bin_width:float
                        sigma:float
                        normalize:bool
                        scale_attributes:bool
                        attr_name:str
                        label_name:str

        """
        self.base_kernel = base_kernel
        self.param = param
        pass

    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, k.dot(k.T))

    @staticmethod
    def locally_sensitive_hashing(m, d, w, sigma=1.0):
        # Compute random projection vector
        v = np.random.randn(d, 1) * sigma  # / np.random.randn(d, 1)

        # Compute random offset
        b = w * np.random.rand() * sigma

        # Compute hashes
        labels = np.floor((np.dot(m, v) + b) / w)

        # Compute label
        _, indices = np.unique(labels, return_inverse=True)

        return indices

    @staticmethod
    def wl_coloring(M, colors, log_primes):
        """
        Each color gets its unique prime number, The adjancy matrix = neighbours are multiplied with the prime numbers.
        The color will be unique because we are using prime numbers.
        """

        log_prime_colors = np.array([log_primes[i] for i in colors], dtype=np.float64)
        colors = colors + M.dot(log_prime_colors)

        # Round numbers to avoid numerical problems
        colors = np.round(colors, decimals=10)

        _, colors = np.unique(colors, return_inverse=True)

        return colors



    def fit_transform(self, X):

        num_vertices = 0
        for g in X:
            num_vertices += g.number_of_nodes()
        n = len(X)

        g = X[0]
        tmp = nx.get_node_attributes(g,self.param['attr_name'])
        dim_attributes = len(tmp[0])
        # print(f'dimension is {dim_attributes}')
        colors_0 = np.zeros([num_vertices, dim_attributes])
        offset = 0

        gram_matrix = np.zeros([n, n])

        # Get attributes from all graph instances
        graph_indices = []
        for g in X:
            for i, attr in enumerate(nx.get_node_attributes(g,self.param['attr_name']).values()):
                colors_0[i + offset] = attr

            graph_indices.append((offset, offset + g.number_of_nodes() - 1))
            offset += g.number_of_nodes()

        # Normalize attributes: center to the mean and component wise scale to unit variance
        if self.param['scale_attributes']:
            colors_0 = pre.scale(colors_0, axis=0)
        #print(colors_0)

        # retrieve base kernel from parameter input
        base_kernel_func = getattr(self, self.base_kernel)

        for it in range(self.param['iterations']):
            colors_hashed = self.locally_sensitive_hashing(colors_0, dim_attributes, self.param['lsh_bin_width'], sigma=self.param['sigma'])

            # caclulate feature vector based on base_kernel
            tmp = base_kernel_func(X, colors_hashed, self.param)

            if it == 0:
                feature_vectors = tmp
            else:
                feature_vectors = sparse.hstack((feature_vectors, tmp)) # Note feature_vectors is a matrix, so we are stacking matrices horizontally

        feature_vectors = feature_vectors.tocsr()

        #if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = np.sqrt(1.0 / self.param['iterations']) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        gram_matrix = gram_matrix.toarray()

        if self.param.get('normalize', False):
            gram_matrix = self.normalize_gram_matrix(gram_matrix)

        return gram_matrix


    def shortest_path_kernel(self,graph_db, hashed_attributes, param):
        label_name = param.get('label_name', None)

        num_vertices = 0
        for g in graph_db:
            num_vertices += g.number_of_nodes()

        offset = 0
        graph_indices = []
        colors_0 = np.zeros(num_vertices, dtype=np.int64)

        # Get labels (colors) from all graph instances
        offset = 0
        for g in graph_db:
            graph_indices.append((offset, offset + g.number_of_nodes() - 1))

            if label_name:
                for i, label in enumerate(nx.get_node_attributes(g,label_name).values()):
                    colors_0[i + offset] = label

            offset += g.number_of_nodes()
        _, colors_0 = np.unique(colors_0, return_inverse=True)

        colors_1 = hashed_attributes

        triple_indices = []
        triple_offset = 0
        triples = []

        # Solve APSP problem for every graphs in graph data base
        for i, g in enumerate(graph_db):
            M = dict(nx.all_pairs_shortest_path_length(g))

            # index is a tuple giving index of first and last node for graph h
            index = graph_indices[i]

            if label_name:
                l = colors_0[index[0]:index[1] + 1]
                h = colors_1[index[0]:index[1] + 1]
            else:
                h = colors_1[index[0]:index[1] + 1]
            d = len(M)
            # For each pair of vertices collect labels, hashed attributes, and shortest-path distance
            pairs = list(it.product(range(d), repeat=2))
            if label_name:
                t = [hash((l[k], h[k], l[j], h[j], M[k][j])) for (k, j) in pairs if (k != j and ~np.isinf(M[k].get(j, np.inf)))]
            else:
                t = [hash((h[k], h[j], M[k][j])) for (k, j) in pairs if (k != j and ~np.isinf(M[k].get(j, np.inf)))]

            triples.extend(t)

            triple_indices.append((triple_offset, triple_offset + len(t) - 1))
            triple_offset += len(t)

        _, colors = np.unique(triples, return_inverse=True)
        m = np.amax(colors) + 1

        # Compute feature vectors
        feature_vectors = []
        for i, index in enumerate(triple_indices):
            feature_vectors.append(np.bincount(colors[index[0]:index[1] + 1], minlength=m))

        return lil.lil_matrix(feature_vectors, dtype=np.float64) # each feature vector will be row


    def WL_kernel(self,graph_db, hashed_attributes, param):
        label_name = param.get('label_name', None)
        wl_iterations = param.get('wl_iterations')

        # Create one empty feature vector for each graph
        feature_vectors = []
        for _ in graph_db:
            feature_vectors.append(np.zeros(0, dtype=np.float64))

        # Construct block diagonal matrix of all adjacency matrices
        adjacency_matrices = []
        for g in graph_db:
            adjacency_matrices.append(np.array(nx.adjacency_matrix(g).todense()))
        M = sp.sparse.block_diag(tuple(adjacency_matrices), dtype=np.float64, format="csr")
        num_vertices = M.shape[0]

        # Load list of precalculated logarithms of prime numbers
        log_primes = log_pl.log_primes[0:num_vertices]

        # Color vector representing labels
        colors_0 = np.zeros(num_vertices, dtype=np.float64)
        # Color vector representing hashed attributes
        colors_1 = hashed_attributes

        # Get labels (colors) from all graph instances
        offset = 0
        graph_indices = []


        for g in graph_db:
            if label_name:
                for i, label in enumerate(nx.get_node_attributes(g,label_name).values()):
                    colors_0[i + offset] = label

            graph_indices.append((offset, offset + g.number_of_nodes() - 1))
            offset += g.number_of_nodes()

        # Map labels to [0, number_of_colors)
        if label_name:
            _, colors_0 = np.unique(colors_0, return_inverse=True)

        for it in range(0, wl_iterations + 1):

            if label_name:
                # Map colors into a single color vector
                colors_all = np.array([colors_0, colors_1])
                colors_all = [hash(tuple(row)) for row in colors_all.T]
                _, colors_all = np.unique(colors_all, return_inverse=True)
                max_all = int(np.amax(colors_all) + 1)
                # max_all = int(np.amax(colors_0) + 1)

                feature_vectors = [
                    np.concatenate((feature_vectors[i], np.bincount(colors_all[index[0]:index[1] + 1], minlength=max_all)))
                    for i, index in enumerate(graph_indices)]

                # Avoid coloring computation in last iteration
                if it < wl_iterations:
                    colors_0 = self.wl_coloring(M, colors_0, log_primes[0:len(colors_0)])
                    colors_1 = self.wl_coloring(M, colors_1, log_primes[0:len(colors_1)])
            else:
                max_1 = int(np.amax(colors_1) + 1)

                feature_vectors = [
                    np.concatenate((feature_vectors[i], np.bincount(colors_1[index[0]:index[1] + 1], minlength=max_1))) for
                    i, index in enumerate(graph_indices)]

                # Avoid coloring computation in last iteration
                if it < wl_iterations:
                    colors_1 = self.wl_coloring(M, colors_1, log_primes[0:len(colors_1)])

        return lil.lil_matrix(feature_vectors, dtype=np.float64) # each feature vector will be row


if __name__ == "__main__":
    # add perant dir 
    import os, sys
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    parentdir = os.path.dirname(parentdir)
    sys.path.append(parentdir)
    import MMDforGraphs as mg
    nr_nodes_1 = 20
    nr_nodes_2 = 20
    n = 50
    m = 50


    average_degree = 6


    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'samelabels')
    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, a = 'normattr', l = 'degreelabels' )
    bg1.Generate()
    #bg2 = mg.BinomialGraphs(n, nr_nodes_1, average_degree+3, a = 'normattr', l = 'degreelabels' )
    bg2 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, a = 'normattr', l = 'degreelabels', loc = 0 )
    bg2.Generate()

    Gs = bg1.Gs + bg2.Gs
    #print(len(Gs))
    test = bg2.Gs[0]
    #print(test.nodes(True))

    # kernel = HashKernel(base_kernel = 'shortest_path_kernel', param = {'iterations':20,
    #                                                                     'lsh_bin_width':0.1, 
    #                                                                     'sigma':1,
    #                                                                     'normalize':True,
    #                                                                      'scale_attributes':True,
    #                                                                      'attr_name': 'attr',
    #                                                                      'label_name':'label'})
    kernel = HashKernel(base_kernel = 'WL_kernel', param = {'iterations':20,
                                                            'lsh_bin_width':0.1, 
                                                            'sigma':1,
                                                            'normalize':True,
                                                            'scale_attributes':True,
                                                            'attr_name': 'attr',
                                                            'label_name':'label',
                                                            'wl_iterations':0})
    K = kernel.fit_transform(Gs)
    print(K)


    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = 100)
    print(kernel_hypothesis.p_values)
