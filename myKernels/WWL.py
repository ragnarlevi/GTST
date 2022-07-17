
"""
Author: Pinar Yanardag (ypinar@purdue.edu)
Code from https://github.com/BorgwardtLab/WWL/blob/master/experiments/main.py and adjusted
"""


import networkx as nx
import numpy as np
import copy
from typing import List
from collections import defaultdict
from sklearn.metrics import DistanceMetric
import ot

class WeisfeilerLehman():
    """
    Class that implements the Weisfeiler-Lehman transform
    Credits: Christian Bock and Bastian Rieck
    """
    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: list, label_name = 'label'):
        """
        Pre-process so labels go from 0,1,...
        """
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
                 
            # get label of graph
            labels = list(nx.get_node_attributes(x,label_name).values())
            
            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels.append(self._preprocess_relabel_dict[label])
            nx.set_node_attributes(x, {i:l for i, l in enumerate(new_labels)}, label_name)
            self._results[0][i] = (labels, new_labels)
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X, num_iterations: int=3,label_name = 'label'):
        """
        Returns a dictionary of dicitonaries where first key is wl iteration number, next key is the index of a graph in the sample which gives a tuple 
        where the first element in the tuple is the previous labels (or initial labes) and the next elment in the new labelling according to the wl scheme
        """
        # Pre-process so labels go from 0,1,...
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = list(nx.get_node_attributes(g,label_name).values())

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(merged_labels)
                self._relabel_steps[i][it] = { idx: {old_label: new_labels[idx]} for idx, old_label in enumerate(current_labels) }
                nx.set_node_attributes(g, {i:l for i, l in enumerate(new_labels)}, label_name)
                self._results[it][i] = (merged_labels, new_labels)
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        return self._results

    def _relabel_graph(self,  merged_labels: List[list]):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str,merged))])
        return new_labels

    def _append_label_dict(self, merged_labels):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X:nx.Graph, sort: bool=True):
        neighbor_indices = [[n_v for n_v in X.neighbors(v)] for v in list(X.nodes)]
        neighbor_labels = []
        for n_indices in neighbor_indices:
            if sort:
                
                neighbor_labels.append( sorted([X.nodes[v]['label'] for v in n_indices]) )
            else:
                neighbor_labels.append( [X.nodes[v]['label'] for v in n_indices] )
                    
        return neighbor_labels


class WWL():



    def __init__(self, labelled = True, param:dict = None) -> None:
        """
        :param param: Dictionary with hyperparameter arguments: discount, h, sinkhorn:bool, sinkhorn_lambda
        """
        self.type = type
        self.param = param
        self.labelled = labelled

    
    def parse_input(self, X):
        if self.labelled:
            label_sequence = self.compute_wl_embeddings_discrete(X, self.param.get('h'))
        else:
            raise ValueError('labels only supported for now')

        return label_sequence


            

    def fit_transform(self, X):
        """
        Calculate the kernel matrix
        Generate the Wasserstein distance matrix for the graphs embedded 
        in label_sequences
                      

        :param X: List of nx graphs.
        """




        # Input validation and parsing
        if X is None:
            raise ValueError('`fit` input cannot be None')
        else:
            label_sequences  = self.parse_input(X)
    

        h = self.param['h']
        # Get the iteration number from the embedding file
        n = len(label_sequences)
        emb_size = label_sequences[0].shape[1]
        n_feat = int(emb_size/(h+1))


        M = np.zeros((n,n))
        # Iterate over pairs of graphs
        for graph_index_1, graph_1 in enumerate(label_sequences):
            # Only keep the embeddings for the first h iterations
            labels_1 = label_sequences[graph_index_1][:,:n_feat*(h+1)]
            for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
                # graph_index_2 + graph_index_1 just take current graph and all graphs after, as we just need to calculate the upper triangle of the kernel matrix
                labels_2 = label_sequences[graph_index_2 + graph_index_1][:,:n_feat*(h+1)] 
                # Get cost matrix
                ground_distance = 'hamming' if self.labelled else 'euclidean'
                
                # costs = ot.dist(labels_1, labels_2, metric=ground_distance)
                costs = DistanceMetric.get_metric(ground_distance).pairwise(labels_1, labels_2)
                if self.param['sinkhorn']:
                    mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                        np.ones(len(labels_2))/len(labels_2), costs, self.param['sinkhorn_lambda'], 
                                        numItermax=50)
                    M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
                else:
                    M[graph_index_1, graph_index_2 + graph_index_1] = \
                        ot.emd2([], [], costs)
                        
        M = (M + M.T)

        K = np.exp(-self.param['discount']*M)

        if self.param.get('normalize', False):
            K = self.normalize_gram_matrix(K)

        return  K

    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, k.dot(k.T))

    def compute_wl_embeddings_discrete(self, graphs, h):

        wl = WeisfeilerLehman()
        label_dicts = wl.fit_transform(graphs, h)

        # Each entry in the list represents the label sequence of a single
        # graph. The label sequence contains the vertices in its rows, and
        # the individual iterations in its columns.
        #
        # Hence, (i, j) will contain the label of vertex i at iteration j.
        label_sequences = [
            np.full((len(graph.nodes()), h + 1), np.nan) for graph in graphs
        ]   

        for iteration in sorted(label_dicts.keys()):
            for graph_index, graph in enumerate(graphs):
                labels_raw, labels_compressed = label_dicts[iteration][graph_index]

                # Store label sequence of the current iteration, i.e. *all*
                # of the compressed labels.
                label_sequences[graph_index][:, iteration] = labels_compressed

        return label_sequences



if __name__ == "__main__":
    # add perant dir 
    import os, sys
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    parentdir = os.path.dirname(parentdir)
    sys.path.append(parentdir)
    import MMDforGraphs as mg
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 20
    m = 20


    average_degree = 6


    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'samelabels')
    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1.Generate()
    bg2 = mg.BinomialGraphs(m, nr_nodes_2, average_degree+0.5, l = 'degreelabels')
    bg2.Generate()

    Gs = bg1.Gs + bg2.Gs
    print(len(Gs))
    kernel = WWL(param = {'discount':1,'h':8, 'sinkhorn':False })
    K = kernel.fit_transform(Gs)
    print(K)


    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = 100)
    print(kernel_hypothesis.p_values)



