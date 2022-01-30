import networkx as nx
import numpy as np
import copy
from typing import List
from collections import defaultdict


class WL():
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
        X list of graphs

        Returns a list of list of graphs, where index i is a list of graphs resulting from the WL iteration of the original graph nr. i
        """
        # Pre-process so labels go from 0,1,...
        self._results = defaultdict(dict)

        if type(X) != list:
            X = [X]
        X = self._relabel_graphs(X)

        X_new = [[G] for G in X]
        new_labels_graphs = [[nx.get_node_attributes(G, label_name)] for G in X]

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

                g_new = X[i].copy()
                nx.set_node_attributes(g_new, new_labels, name=label_name)
                X_new[i].append(g_new)
                new_labels_graphs[i].append(new_labels)


            #     self._relabel_steps[i][it] =  { idx: (old_label, merged_labels[idx], list(new_labels.values())[idx]) for idx, old_label in enumerate(current_labels.values()) }
            #     nx.set_node_attributes(g, {i:l for i, l in zip(current_labels.keys(), new_labels.values())}, label_name)
            #     self._results[i][it] = (merged_labels, new_labels)
                
            # self._label_dicts[it] = copy.deepcopy(self._label_dict)
        
        return X_new, new_labels_graphs



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