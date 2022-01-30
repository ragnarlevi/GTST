
import networkx as nx
import numpy as np
import copy
from typing import List
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

class WeisfeilerLehmanv2():
    """
    Class that implements the Weisfeiler-Lehman transform. Now allows nodes to be ordered/labelled arbitrarily
    Credits: Christian Bock and Bastian Rieck, 
    """
    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        
        self._label_dicts = {}

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label
    
    def _reset_label_generation(self):
        self._last_new_label = -1

    def _relabel_graphs(self, X: list, label_name = 'label'):
        """
        Pre-process so labels go from 0,1,...
        """
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
                 
            # get label of graph
            labels = nx.get_node_attributes(x,label_name)
            
            new_labels = dict()
            for node, label in labels.items():
                if label in self._preprocess_relabel_dict.keys():
                    new_labels[node] = self._preprocess_relabel_dict[label]
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels[node] = self._preprocess_relabel_dict[label]
            #print(new_labels)
            nx.set_node_attributes(x, {i:l for i, l in new_labels.items()}, label_name)
            self._results[i][0] = (labels, new_labels)
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X, num_iterations: int=3, label_name = 'label'):
        """
        Returns a dictionary of dicitonaries where first key is wl iteration number, next key is the index of a graph in the sample which gives a tuple 
        where the first element in the tuple is the previous labels (or initial labes) and the next elment in the new labelling according to the wl scheme
        """
        # Pre-process so labels go from 0,1,...
        self._results = defaultdict(dict)

        if type(X) != list:
            X = [X]
        X = self._relabel_graphs(X)

        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):

                # Get labels of current interation
                current_labels = nx.get_node_attributes(g,label_name)

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels.values())]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = dict()
                new_labels = self._relabel_graph(current_labels, merged_labels)

                self._relabel_steps[i][it] =  { idx: (old_label, merged_labels[idx], list(new_labels.values())[idx]) for idx, old_label in enumerate(current_labels.values()) }
                nx.set_node_attributes(g, {i:l for i, l in zip(current_labels.keys(), new_labels.values())}, label_name)
                self._results[i][it] = (merged_labels, new_labels)
                
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        
        return self._results



    def _relabel_graph(self, current_labels,  merged_labels: List[list]):
        new_labels = dict()
        for node, merged in zip(current_labels.keys(), merged_labels):
            new_labels[node] = self._label_dict['-'.join(map(str,merged))]
        return new_labels

    def _append_label_dict(self, merged_labels):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X:nx.Graph):
        neighbor_indices = [[n_v for n_v in X.neighbors(v)] for v in list(X.nodes)]
        neighbor_labels = []
        for n_indices in neighbor_indices:
            neighbor_labels.append( sorted([X.nodes[v]['label'] for v in n_indices]) )
        return neighbor_labels






class GIK():

    def __init__(self) -> None:
        pass

    def __init__(self, label_name, local = True, attr_name = None,  params:dict = None) -> None:
        """
        :param param: Dictionary with hyperparameter arguments: discount, h
        """
        self.local = local
        self.label_name = label_name
        self.attr_name = attr_name
        self.params = params

    @staticmethod
    def semi_bfs(G, source, distance):
        """
        breadth first search such that searches within a radius of a source node
        """

        node_list = set()
        node_list.add(source)
        
        # Mark all the vertices as not visited

        visited = {i:False for i in  list(G.nodes())}

        # Create a queue for BFS
        current_queue = []

        # Mark the source node as
        # visited and enqueue it
        current_queue.append(source)
        next_queue = []
        visited[source] = True

        if len([n for n in G.neighbors(source)]) == 0:
            return source

        More_things_to_visit = True
        r = 0

        while r < distance and More_things_to_visit:
            # print(visited)
            for i in current_queue:
                neighbours = [n for n in G.neighbors(i)]
                for j in neighbours:
                    node_list.add(j)
                    if visited[j] == False:
                        visited[j] = True
                        next_queue.append(j)

            r += 1
            # print(next_queue)
            if len(next_queue) == 0:
                # print("More_things_to_visit")
                More_things_to_visit = False

            current_queue = next_queue.copy()
            next_queue = []

        

        return list(node_list)

    @staticmethod
    def normalize_gram_matrix(x):
        k = np.reciprocal(np.sqrt(np.diag(x)))
        k = np.resize(k, (len(k), 1))
        return np.multiply(x, k.dot(k.T))


    def parse_input(self, X):
        """
        Create subgraphs of increasing radius for each vertex for each graph and apply WL
        """

        num_graphs = len(X)
        convolution_pattern = {graph_nr: {node_nr: list() for node_nr in range(X[graph_nr].number_of_nodes())} for graph_nr in range(num_graphs)}

        #convolution_pattern = {graph_nr:list() for graph_nr in range(num_graphs)}
   

        wl_itr = self.params['wl_itr']
        wl = WeisfeilerLehmanv2()
        all_sub_patterns = []
        graph_id = []
        node_id = []
        distance_id = []
        for idx, G in enumerate(X):
            for n in G.nodes:
                for r in range(self.params['distances']):

                    sub_g = G.subgraph(self.semi_bfs(G, source = n, distance = r)).copy()
                    #convolution_pattern[idx].append((r, n, sub_g ))
                    convolution_pattern[idx][n].append( sub_g)
                    all_sub_patterns.append(sub_g)
                    graph_id.append(idx)
                    node_id.append(n)
                    distance_id.append(r)

                    
        if self.local:
            vertex_invariant_pattern = {graph_nr: {node_nr: list() for node_nr in range(X[graph_nr].number_of_nodes())} for graph_nr in range(num_graphs)} 
            out = wl.fit_transform(all_sub_patterns,   wl_itr)

            for i in range(len(all_sub_patterns)):
                    vertex_invariant_pattern[graph_id[i]][node_id[i]].append(out[i])

        else:
            out = wl.fit_transform(X,   wl_itr)
            vertex_invariant_pattern = {graph_nr: out[graph_nr] for graph_nr in range(num_graphs)} 

        return convolution_pattern, vertex_invariant_pattern


    def fit_transform(self, X):
        """
        Calculate the kernel matrix
        Generate the Wasserstein distance matrix for the graphs embedded 
        in label_sequences

                    

        :param X: List of nx graphs.
        """

        # Note vertex_invariant_pattern has a different structure for local or non local
        convolution_pattern, vertex_invariant_pattern = self.parse_input(X)


        
        graph_db_size = len(X)

        K = np.zeros(shape = (graph_db_size, graph_db_size), dtype=float)

        for idx1, G1 in enumerate(X):
            for idx2, G2 in enumerate(X):

                if idx1> idx2:
                    continue


                weights_matrix = np.zeros((G1.number_of_nodes(), G2.number_of_nodes())) 

                if self.attr_name:
                    K_attr = np.zeros((G1.number_of_nodes(), G2.number_of_nodes())) 

                for v_1 in range(G1.number_of_nodes()):
                    for v_2 in range(G2.number_of_nodes()):

                        # compare patters, only compare if r = r
                        for r, (pattern_1, pattern_2) in enumerate(zip(convolution_pattern[idx1][v_1],convolution_pattern[idx2][v_2])):

                            # Graph invariant

                            if pattern_1.number_of_nodes() != pattern_2.number_of_nodes():
                                continue
                            if self.local:
                                # No break so now we extract the WL features (label at each wl iteration) and calculate the vertex invariant
                                v_1_score = []
                                for _,v in vertex_invariant_pattern[idx1][v_1][r].items():
                                    v_1_score.append(v[1][v_1])
                            
                                v_2_score = []
                                for _,v in vertex_invariant_pattern[idx2][v_2][r].items():
                                    v_2_score.append(v[1][v_2])

                            if self.local:
                                weights_matrix[v_1,v_2]  += np.sum([float(v_1_score[i] == v_2_score[i]) for i in range(len(v_1_score))], dtype=float) / float(len(pattern_1) * len(pattern_2))
                            else:
                                weights_matrix[v_1,v_2] += 1.0/ float(len(pattern_1) * len(pattern_2))

                        if not self.local:
                            weights_matrix[v_1,v_2] *= np.sum([float(vertex_invariant_pattern[idx1][i][1][v_1] == vertex_invariant_pattern[idx2][i][1][v_2])   for i in range(self.params['wl_itr'])], dtype=float)

                        
                        if self.attr_name:
                            x = G1.nodes[v_1][self.attr_name]
                            y = G2.nodes[v_2][self.attr_name]
                            sqdist_X = np.dot(x,x) - 2 * np.dot(x,y) + np.dot(y,y)
                            K_attr[v_1, v_2] = np.exp(-sqdist_X / self.params['c'])

                if self.attr_name:
                    K[idx1, idx2] = np.sum(np.multiply(weights_matrix,K_attr))
                else:
                    K[idx1, idx2] = np.sum(weights_matrix) #np.dot(ones.T,weights_matrix).dot(ones) #np.trace(weights_matrix.dot(np.ones(weights_matrix.shape)))


        K = K + K.T - np.diag(np.diag(K))
  
        if self.params.get('normalize', False):
            K = self.normalize_gram_matrix(K)

        return K





        #                 for g_1 in convolution_pattern[idx1]:
        #                     for g_2 in convolution_pattern[idx2]:
                                

        #                         # Graph invariant
        #                         if g_1[0] != g_2[0]:
        #                             continue

        #                         # Graph invariant
        #                         if g_1[2].number_of_nodes() != g_2[2].number_of_nodes():
        #                             continue
                                
        #                         # indicator function
        #                         if not (v_1 in g_1[2].nodes() and v_2 in g_2[2].nodes()):
        #                             continue
                                
        #                         # local vertex invariant for this substructure?
        #                         if self.local:
        #                             v_1_score = []
        #                             for _,v in vertex_invariant_pattern[idx1][v_1][g_1[0]].items():
        #                                 #print(v)
        #                                 v_1_score.append(v[1][v_1])
                                
        #                             v_2_score = []
        #                             for _,v in vertex_invariant_pattern[idx2][v_2][g_2[0]].items():
        #                                 v_2_score.append(v[1][v_2])

        #                             weights_matrix[v_1,v_2]  += np.sum([float(v_1_score[i] == v_2_score[i]) for i in range(len(v_1_score))], dtype=float) / float(len(g_1[2]) * len(g_2[2]))
        #                         else:
        #                             weights_matrix[v_1,v_2] += 1.0/ float(len(g_1[2]) * len(g_2[2]))


        #                 if not self.local:
        #                     weights_matrix[v_1,v_2] *= np.sum([float(vertex_invariant_pattern[idx1][i][1][v_1] == vertex_invariant_pattern[idx2][i][1][v_2])   for i in range(self.params['wl_itr'])], dtype=float)

        #                 if self.attr_name:
        #                     x = G1.nodes[v_1][self.attr_name]
        #                     y = G2.nodes[v_2][self.attr_name]
        #                     sqdist_X = np.dot(x,x) - 2 * np.dot(x,y) + np.dot(y,y)
        #                     K_attr[v_1, v_2] = np.exp(-sqdist_X / self.params['c'])

                    
        #         if self.attr_name:
        #             K[idx1, idx2] = np.sum(np.multiply(weights_matrix,K_attr))
        #         else:
        #             K[idx1, idx2] = np.sum(weights_matrix) #np.dot(ones.T,weights_matrix).dot(ones) #np.trace(weights_matrix.dot(np.ones(weights_matrix.shape)))

  
        # return K + K.T - np.diag(np.diag(K))
                                    

                        

                        # # compare patters, only compare if r = r
                        # for r, (pattern_1, pattern_2) in enumerate(zip(convolution_pattern[idx1][v_1],convolution_pattern[idx2][v_2])):

                        #     # Graph invariant

                        #     if pattern_1.number_of_nodes() != pattern_2.number_of_nodes():
                        #         continue
                        #     if self.local:
                        #         # No break so now we extract the WL features (label at each wl iteration) and calculate the vertex invariant
                        #         v_1_score = []
                        #         for _,v in vertex_invariant_pattern[idx1][v_1][r].items():
                        #             print(v)
                        #             v_1_score.append(v[1][v_1])
                            
                        #         v_2_score = []
                        #         for _,v in vertex_invariant_pattern[idx2][v_2][r].items():
                        #             v_2_score.append(v[1][v_2])

                        #     if self.local:
                        #         weights_matrix[v_1,v_2]  += np.sum([float(v_1_score[i] == v_2_score[i]) for i in range(len(v_1_score))], dtype=float) / float(len(pattern_1) * len(pattern_2))
                        #     else:
                        #         weights_matrix[v_1,v_2] += 1.0/ float(len(pattern_1) * len(pattern_2))

                        # if not self.local:
                        #     weights_matrix[v_1,v_2] *= np.sum([float(vertex_invariant_pattern[idx1][i][1][v_1] == vertex_invariant_pattern[idx2][i][1][v_2])   for i in range(self.params['wl_itr'])], dtype=float)

                        
        #                 if self.attr_name:
        #                     x = G1.nodes[v_1][self.attr_name]
        #                     y = G2.nodes[v_2][self.attr_name]
        #                     sqdist_X = np.dot(x,x) - 2 * np.dot(x,y) + np.dot(y,y)
        #                     K_attr[v_1, v_2] = np.exp(-sqdist_X / self.params['c'])

        #         if self.attr_name:
        #             K[idx1, idx2] = np.sum(np.multiply(weights_matrix,K_attr))
        #         else:
        #             K[idx1, idx2] = np.sum(weights_matrix) #np.dot(ones.T,weights_matrix).dot(ones) #np.trace(weights_matrix.dot(np.ones(weights_matrix.shape)))

  


        # return K + K.T - np.diag(np.diag(K))




if __name__ == '__main__':
    import os, sys
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    parentdir = os.path.dirname(parentdir)
    sys.path.append(parentdir)
    import MMDforGraphs as mg

    nr_nodes_1 = 20
    nr_nodes_2 = 20
    n = 10
    m = 10


    average_degree = 2


    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'samelabels')
    # bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, l = 'degreelabels')
    bg1 = mg.BinomialGraphs(n, nr_nodes_1, average_degree, a = 'normattr', l = 'degreelabels', loc = 1, fullyConnected=True )
    bg1.Generate()
    bg2 = mg.BinomialGraphs(m, nr_nodes_2, average_degree, a = 'normattr', l = 'degreelabels', loc = 2, fullyConnected=True)
    bg2.Generate()

    Gs = bg1.Gs + bg2.Gs
    X = Gs



    #print(len(Gs))
    kernel = GIK(local = True, label_name = 'label', attr_name= 'attr', params = {'wl_itr':4,'distances':3, 'c':0.1 })
    #kernel = GIK(local = False, label_name = 'label',  params = {'wl_itr':4,'distances':3 })
    K = kernel.fit_transform(Gs)
    #print(K)



    MMD_functions = [mg.MMD_b, mg.MMD_u]

    # initialize bootstrap class, we only want this to be initalized once so that numba njit
    # only gets compiled once (at first call)
    kernel_hypothesis = mg.BoostrapMethods(MMD_functions)
    function_arguments=[dict(n = bg1.n, m = bg2.n ), dict(n = bg1.n, m = bg2.n )]
    kernel_hypothesis.Bootstrap(K, function_arguments, B = 100)
    print(kernel_hypothesis.p_values)