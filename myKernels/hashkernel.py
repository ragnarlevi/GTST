



import numpy as np
import networkx as nx
import scipy.sparse as sparse
from sklearn import preprocessing as pre

import itertools as it
import numpy as np
import scipy.sparse.csr as csr
import scipy.sparse.lil as lil

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


    def fit_transform(self, X):

        num_vertices = 0
        for g in X:
            num_vertices += g.number_of_nodes()
        n = len(X)

        g = X[0]
        tmp = nx.get_node_attributes(g,self.param['attr_name'])
        dim_attributes = len(tmp[0])
        print(f'dimension is {dim_attributes}')
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


        base_kernel_func = getattr(self, self.base_kernel)
        for it in range(self.param['iterations']):
            colors_hashed = self.locally_sensitive_hashing(colors_0, dim_attributes, self.param['lsh_bin_width'], sigma=self.param['sigma'])

            tmp = base_kernel_func(X, colors_hashed, self.param)

            if it == 0: #and not use_gram_matrices:
                feature_vectors = tmp
            else:
                # if use_gram_matrices:
                #     feature_vectors = tmp
                #     feature_vectors = feature_vectors.tocsr()
                #     feature_vectors = np.sqrt(1.0 / iterations) * (feature_vectors)
                #     gram_matrix += feature_vectors.dot(feature_vectors.T).toarray()

                #else:
                feature_vectors = sparse.hstack((feature_vectors, tmp)) # Note feature_vectors is a matrix, so we are stacking matrices horizontally

        feature_vectors = feature_vectors.tocsr()

        #if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = np.sqrt(1.0 / self.param['iterations']) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        gram_matrix = gram_matrix.toarray()

        if self.param['normalize']:
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


        #if not compute_gram_matrix:
        return lil.lil_matrix(feature_vectors, dtype=np.float64) # each feature vector will be row
        # else:
        #     # Make feature vectors sparse
        #     gram_matrix = csr.csr_matrix(feature_vectors, dtype=np.float64) # each feature vector will be row
        #     # Compute gram matrix
        #     gram_matrix = gram_matrix.dot(gram_matrix.T)

        #     gram_matrix = gram_matrix.toarray()

        #     if normalize_gram_matrix:
        #         return normalize_gram_matrix(gram_matrix)
        #     else:
        #         return gram_matrix

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
    bg2 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, a = 'normattr', l = 'degreelabels', loc = 5 )
    bg2.Generate()

    Gs = bg1.Gs + bg2.Gs
    print(len(Gs))
    test = bg2.Gs[0]
    print(test.nodes(True))

    kernel = HashKernel(base_kernel = 'shortest_path_kernel', param = {'iterations':20,
                                                                        'lsh_bin_width':0.1, 
                                                                        'sigma':1,
                                                                        'normalize':True,
                                                                         'scale_attributes':True,
                                                                         'attr_name': 'attr',
                                                                         'label_name':'label'})
    K = kernel.fit_transform(Gs)
    print(K)


    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = 100)
    print(kernel_hypothesis.p_values)
