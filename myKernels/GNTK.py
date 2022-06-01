import math
import numpy as np
import scipy as sp
import tqdm

class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale, normalize):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        normalize: normalize kernel matrix?
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(scale in ['uniform', 'degree'])
        self.normalize = normalize

    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, np.outer(k,k))
    
    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = np.sqrt(np.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g
        A: adjacency matrix
        """
        N = A.shape[0]
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))

        diag_list = []
        adj_block = sp.sparse.kron(A, A)

        # input covariance
        sigma = np.matmul(g.node_features, g.node_features.T)
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
        ntk = np.copy(sigma)
		
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)
        return diag_list

    def gntk(self, g1, g2, diag_list1, diag_list2, A1, A2):
        """
        compute the GNTK value \Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """
        
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))
        
        adj_block = sp.sparse.kron(A1, A2)
        
        jump_ntk = 0
        sigma = np.matmul(g1.node_features, g2.node_features.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
        ntk = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)
        if self.jk:
            return np.sum(jump_ntk) * 2
        else:
            return np.sum(ntk) * 2


    def preprocess(self, g_list_nx, degree_as_tag, features = None):

        g_list = []

        for i, g in enumerate(g_list_nx):
            g_list.append(S2VGraph(g, i, node_tags = None))



        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])


        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

        if features is None:
            #Extracting unique tag labels   
            tagset = set([])
            for g in g_list:
                tagset = tagset.union(set(g.node_tags))

            tagset = list(tagset)
            tag2index = {tagset[i]:i for i in range(len(tagset))}

            for g in g_list:
                g.node_features = np.zeros([len(g.node_tags), len(tagset)])
                g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        else:
            for g in g_list:
                g.node_features = np.array([i[1] for i in g.g.nodes(features)])


        self.g_list = g_list

        A_list = []
        diag_list = []

        # procesing the data
        for i in range(len(g_list)):
            n = len(g_list[i].neighbors)
            for j in range(n):
                g_list[i].neighbors[j].append(j)
            edges = g_list[i].g.edges
            m = len(edges)

            row = [e[0] for e in edges]
            col = [e[1] for e in edges]

            A_list.append(sp.sparse.coo_matrix(([1] * len(edges), (row, col)), shape = (n, n), dtype = np.float32))
            A_list[-1] = A_list[-1] + A_list[-1].T + sp.sparse.identity(n)
            diag = self.diag(g_list[i], A_list[i])
            diag_list.append(diag)

        self.diag_list = diag_list
        self.A_list = A_list



    def fit_all(self, verbose = True):

        n = len(self.g_list)

        K = np.zeros((n, n))

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total= n*(n+1)/2)

        for i in range(n):
            for j in range(i, n):

                K[i,j] = self.gntk(self.g_list[i], self.g_list[j], self.diag_list[i], self.diag_list[j], self.A_list[i], self.A_list[j])

                if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        K = K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            K = self.normalize_gram_matrix(K)


        return K


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0

        self.max_neighbor = 0