import numpy as np
import networkx as nx

import sys, os
sys.path.append('src')
from GTST import MMD 
from tests.utils import generate_H0_false, assert_low_p_val,  generate_H0_false_but_same_topology, generate_H0_false_directed, generate_Xs_H0_false



def test_MMD():

    # Test various kernels, similar to kernels_test

    # Test Random Walk
    g1, g2 = generate_H0_false()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_plus', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l', 'MONK_EST'], r = 6, c = 0.001, Q = 5, verbose = True)
    assert_low_p_val(MMD_out.p_values, "RW ARKU_plus diff topology")

    g1, g2 = generate_H0_false_but_same_topology()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_plus', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 6, c = 1e-4, verbose = False, node_attr = 'attr',
                bootstrap_method = 'BootstrapScheme', check_symmetry = True)
    assert_low_p_val(MMD_out.p_values, "RW ARKU_plus same topology")

    g1, g2 = generate_H0_false_but_same_topology()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_edge', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 6, c = 1e-4, verbose = False, 
                edge_label = 'edge_label', unique_edge_labels = ['a', 'b'], check_symmetry = True)
    assert_low_p_val(MMD_out.p_values, "RW ARKU_edge same topology")

    g1, g2 = generate_H0_false_but_same_topology()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKL', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 6, c = 1e-4, verbose = False, 
                node_label = 'label', unique_node_labels = ['a', 'b', 'c'])
    assert_low_p_val(MMD_out.p_values, "RW ARKL same topology")

    g1, g2 = generate_H0_false_directed()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 6, c = 1e-4, verbose = False, edge_attr = 'weight')
    assert_low_p_val(MMD_out.p_values, "RW ARKU same topology")

    # Test GNTK
    g1, g2 = generate_H0_false()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = True, scale = 'uniform', degree_as_tag = True, verbose =False)
    assert_low_p_val(MMD_out.p_values, "GNTK same topology uniform")
    g1, g2 = generate_H0_false_but_same_topology()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = False, scale = 'degree', node_attr = 'attr', verbose =False)
    assert_low_p_val(MMD_out.p_values, "GNTK same topology degree")

    # Test DK
    g1, g2 = generate_H0_false_but_same_topology()
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'wl', wl_it = 4, node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "DK same topology wl")
    
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'sp', node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "DK same topology sp")


    # Test WWL
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = 'WWL', mmd_estimators = 'MMD_u', discount = 0.1, h = 2, node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "WWL same topology")

    # Test one Grakel kernel
    kernel = [{"name": "weisfeiler_lehman", "n_iter": 1}, {"name": "vertex_histogram"}]
    MMD_out = MMD()
    MMD_out.fit(G1 = g1, G2 = g2, kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "Grakel WL same topology")



    # Test graph fitting

    X1,X2 = generate_Xs_H0_false()
    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False)
    MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 6, c = 0.01, edge_attr = 'weight')
    assert_low_p_val(MMD_out.p_values, "Graph fitting WL same topology")

    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=True,scale = False)
    MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 6, c = 0.01, edge_attr = 'weight')
    assert_low_p_val(MMD_out.p_values, "Graph fitting WL same topology nonparanormal")

    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = True)
    MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 6, c = 0.01, edge_attr = 'weight')
    assert_low_p_val(MMD_out.p_values, "Graph fitting WL same topology scale")


    label_dict = {'1':{j:i for j,i in enumerate(['a']*8 + ['b']*3)}, 
              '2':{j:i for j,i in enumerate(['a']*4 + ['b']*7)}}
    kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_dict)
    MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
    assert_low_p_val(MMD_out.p_values)
    assert_low_p_val(MMD_out.p_values, "label_dict")


    def label_function(X):
        m = np.mean(X,axis = 0)
        return {i:str(np.round(m[i],1)) for i in range(len(m))}

    kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_function)
    MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "label_function")


    kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= 'degree')
    MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
    assert_low_p_val(MMD_out.p_values, "degree")


    def attr_function(X):
        return np.expand_dims(np.mean(X,axis = 0),axis=1)

    MMD_out = MMD()
    MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_attributes = attr_function)
    MMD_out.fit(kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = False, scale = 'degree', node_attr = 'attr', verbose =True)
    assert_low_p_val(MMD_out.p_values, "attr_function")



























