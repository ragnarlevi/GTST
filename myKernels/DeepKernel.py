
"""
Author: Pinar Yanardag (ypinar@purdue.edu)
Please refer to: http://web.ics.purdue.edu/~ypinar/kdd for more details.
"""

from collections.abc import Iterable
import networkx as nx
import numpy as np
import copy



class DK():



    def __init__(self, params:dict) -> None:
        """
        :param param: Dictionary with hyperparameter arguments
        """

        self.params = params

        
    @staticmethod
    def normalize_kernel_matrix(K):
        D = np.diag(np.reciprocal(np.sqrt(np.diag(K))))
        return D.dot(K).dot(D)


    def ShortestPathSimilarity(self, X):
        """
        Creates shortest path similarity for the Deep Kernel

        :param X: List of nx graphs
        """
        vocabulary = set()
        prob_map = {}
        corpus = []

        for gidx, graph in enumerate(X):

            prob_map[gidx] = {}
            # label of each node
            label_map = list(nx.get_node_attributes(graph,'label').values())
            # get all pairs shortest paths
            all_shortest_paths = nx.all_pairs_shortest_path(graph) # nx.floyd_warshall(G)
            # traverse all paths and subpaths
            tmp_corpus = []
            # source is node we are going from
            # sink is node that we are walking to
            for source, sink_map in all_shortest_paths:
                for sink, path in sink_map.items():
                    sp_length = len(path)-1
                    label = "_".join(map(str, sorted([label_map[source],label_map[sink]]))) + "_" + str(sp_length) 
                    tmp_corpus.append(label)
                    prob_map[gidx][label] = prob_map[gidx].get(label, 0) + 1
                    vocabulary.add(label)
            corpus.append(tmp_corpus)
            # Normalize frequency
        prob_map = {gidx: {path: count/float(sum(paths.values())) for path, count in paths.items()} for gidx, paths in prob_map.items()}

        return prob_map, vocabulary, corpus

    def WLSimilarity(self, X, max_h):
        labels = {}
        label_lookup = {}
        # labels are usually strings so we relabel them as integers for sorting purposes
        label_counter = 0 
        vocabulary = set()
        num_graphs = len(X)

        # it stands for wl iteration. the initial labelling is indexed at [-1]
        # Contains features, which are coiunts
        wl_graph_map = {it: {gidx: dict() for gidx in range(num_graphs)} for it in range(-1, max_h)} # if key error, return 0

      
        # initial labeling
        # label_lookup is a dictionary where key is the label. This loops count how often the 
        for gidx in range(num_graphs):
            labels[gidx] = np.zeros(len(X[gidx]))
            current_graph_labels = nx.get_node_attributes(X[gidx],'label')
            for node in X[gidx].nodes():
                label = current_graph_labels.get(node, -1) # label of current node, if not labelled it gets -1
                if not label in label_lookup:
                    # if we have not observed this label we relabel at the current label_counter
                    label_lookup[label] = label_counter 
                    labels[gidx][node] = label_counter
                    label_counter += 1
                else:
                    labels[gidx][node] = label_lookup[label]
                # add feature, features are counts
                wl_graph_map[-1][gidx][label_lookup[label]] = wl_graph_map[-1][gidx].get(label_lookup[label], 0) + 1
        # we are constantly changing the label dictionary so we do a deepcopy
        compressed_labels = copy.deepcopy(labels)


        # WL iterations
        for it in range(max_h):
            label_lookup = {}
            label_counter = 0
            for gidx in range(num_graphs):
                for node in X[gidx].nodes():
                    node_label = tuple([labels[gidx][node]])
                    neighbors = list(X[gidx].neighbors(node))
                    if len(neighbors) > 0:
                        neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                        #node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                        node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                    if not node_label in label_lookup:
                        label_lookup[node_label] = str(label_counter)
                        compressed_labels[gidx][node] = str(label_counter)
                        label_counter += 1
                    else:
                        compressed_labels[gidx][node] = label_lookup[node_label]
                    wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label], 0) + 1

            labels = copy.deepcopy(compressed_labels)

        # end = timer() 
        # #print(f' WL it took {end - start}')
        # start = timer()
        # create the corpus, frequency map and vocabulary
        graphs = {}
        prob_map = {}
        corpus = []
        for it in range(-1, max_h):
            for gidx, label_map in wl_graph_map[it].items():
                if gidx not in graphs:
                    graphs[gidx] = []
                    prob_map[gidx] = {}
                for label_, count in label_map.items():
                    label = str(it) + "+" + str(label_)
                    for _ in range(count):
                        graphs[gidx].append(label)
                    vocabulary.add(label)
                    prob_map[gidx][label] = count

        prob_map = {gidx: {path: count/float(sum(paths.values())) for path, count in paths.items()} for gidx, paths in prob_map.items()}

        corpus = [graph for graph in graphs.values()]
        vocabulary = sorted(vocabulary)

        #end = timer() 
        #print(f' Corpus took {end - start}')



        return prob_map, vocabulary, corpus


    def parse_input(self, X):
        """
        Creates features of the Deep Kernel

        :param X: List of nx graphs
        :param type: Type of similarity
        :param wl_it: Number of wl iterations
        """

        if not isinstance(X, Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            if self.params['type'] == "sp":
                return self.ShortestPathSimilarity(X)
            if self.params['type']:
                return self.WLSimilarity(X, self.params['wl_it'])
            else:
                raise ValueError("type has to be sp for shortest path, wl for weisfeiler lehman ")


    
            

    def fit_transform(self, X):
        """
        Calculate the kernel matrix

        :param X: List of nx graphs.
        :param wl_it: Number of wl iterations
        :param vector_size: Dimensionality of the word vectors.
        :param window: Maximum distance between the current and predicted word within a sentence.
        :param min_count: Ignores all words with total frequency lower than this
        :param workers: Use these many worker threads to train the model (=faster training with multicore machines)
        """


        # Parameter initialization
        #self.initialize()

        # Input validation and parsing
        if X is None:
            raise ValueError('`fit` input cannot be None')
        else:
            prob_map, vocabulary, corpus  = self.parse_input(X)


        if self.params.get('opt_type', None) == 'word2vec':
            from gensim.models import Word2Vec
            model = Word2Vec(corpus, vector_size=self.params['vector_size'], window=self.params['window'], min_count=0, workers = self.params['workers'])
        
        num_voc = len(X)

        if self.params.get('opt_type', None) == 'word2vec':
            P = np.zeros((num_voc, len(vocabulary)))
            for i in range(num_voc):
                for jdx, j in enumerate(vocabulary):
                    P[i][jdx] = prob_map[i].get(j,0)
            M = np.zeros((len(vocabulary), len(vocabulary)))
            for idx, i in enumerate(vocabulary):
                for jdx, j in enumerate(vocabulary):
                    M[idx, jdx] = np.dot(model.wv[i], model.wv[j])
            K = (P.dot(M)).dot(P.T)
        else:
            P = np.zeros((num_voc, len(vocabulary)))
            for i in range(num_voc):
                for jdx, j in enumerate(vocabulary):
                    P[i][jdx] = prob_map[i].get(j,0)
            K = P.dot(P.T)

    
        if self.params.get('normalize', False):
            K = self.normalize_kernel_matrix(K)

        return K



if __name__ == "__main__":
    # add perant dir 
    import os, sys
    from timeit import default_timer as timer
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    parentdir = os.path.dirname(parentdir)
    sys.path.append(parentdir)
    import MMDforGraphs as mg
    nr_nodes_1 = 100
    nr_nodes_2 = 100
    n = 3
    m = 3


    average_degree = 6


    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'samelabels')
    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1.Generate()
    bg2 = mg.BinomialGraphs(m, nr_nodes_2, average_degree+2, l = 'degreelabels')
    bg2.Generate()

    Gs = bg1.Gs + bg2.Gs
    kernel = DK(params = {'type': 'wl', 'normalize':True, 'wl_it':4})
    start = timer()
    K = kernel.fit_transform(Gs)
    print(K)

    end = timer() 
    print(f' K took {end - start}')

    MMD_functions = [mg.MMD_b, mg.MMD_u]
    
    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = 100)
    print(kernel_hypothesis.p_values)



