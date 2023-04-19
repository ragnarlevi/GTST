import numpy as np
import networkx as nx

import sys, os
sys.path.append('../src')
from GTST import MMD 
from tests.utils import generate_H0_false, assert_low_p_val,  generate_H0_false_but_same_topology, generate_H0_false_directed, generate_Xs_H0_false



# def test_MMD():

#     # Test various kernels, similar to kernels_test

#     # Test Random Walk
#     g1, g2 = generate_H0_false()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_plus', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l', 'MONK_EST'], r = 4, c = 0.001, Q = 5, verbose = True)
#     assert_low_p_val(MMD_out.p_values)

#     g1, g2 = generate_H0_false_but_same_topology()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_plus', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 3, c = 1e-4, verbose = True, node_attr = 'attr',
#                 bootstrap_method = 'BootstrapScheme', check_symmetry = True)
#     assert_low_p_val(MMD_out.p_values)

#     g1, g2 = generate_H0_false_but_same_topology()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU_edge', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 3, c = 1e-4, verbose = True, 
#                 edge_label = 'edge_label', unique_edge_labels = ['a', 'b'], check_symmetry = True)
#     assert_low_p_val(MMD_out.p_values)

#     g1, g2 = generate_H0_false_but_same_topology()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKL', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 3, c = 1e-4, verbose = True, 
#                 node_label = 'label', unique_node_labels = ['a', 'b', 'c'])
#     assert_low_p_val(MMD_out.p_values)

#     g1, g2 = generate_H0_false_directed()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'RW_ARKU', mmd_estimators = ['MMD_u', 'MMD_b', 'MMD_l'], r = 3, c = 1e-4, verbose = True, edge_attr = 'weight')
#     assert_low_p_val(MMD_out.p_values)

#     # Test GNTK
#     g1, g2 = generate_H0_false()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = True, scale = 'uniform', degree_as_tag = True, verbose =True)
#     assert_low_p_val(MMD_out.p_values)
#     g1, g2 = generate_H0_false_but_same_topology()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = False, scale = 'degree', node_attr = 'attr', verbose =True)
#     assert_low_p_val(MMD_out.p_values)

#     # Test DK
#     g1, g2 = generate_H0_false_but_same_topology()
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'wl', wl_it = 4, node_label = 'label')
#     assert_low_p_val(MMD_out.p_values)
    
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'DK', mmd_estimators = 'MMD_u', type = 'sp', node_label = 'label')
#     assert_low_p_val(MMD_out.p_values)


#     # Test WWL
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = 'WWL', mmd_estimators = 'MMD_u', discount = 0.1, h = 2, node_label = 'label')
#     assert_low_p_val(MMD_out.p_values)

#     # Test one Grakel kernel
#     kernel = [{"name": "weisfeiler_lehman", "n_iter": 1}, {"name": "vertex_histogram"}]
#     MMD_out = MMD()
#     MMD_out.fit(G1 = g1, G2 = g2, kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
#     assert_low_p_val(MMD_out.p_values)



#     # Test graph fitting

#     X1,X2 = generate_Xs_H0_false()
#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False)
#     MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 3, c = 0.1, edge_attr = 'weight')
#     assert_low_p_val(MMD_out.p_values)

#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=True,scale = False)
#     MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 3, c = 0.1, edge_attr = 'weight')
#     assert_low_p_val(MMD_out.p_values)

#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = True)
#     MMD_out.fit( kernel = 'RW_ARKU_plus', mmd_estimators = 'MMD_u', r = 3, c = 0.1, edge_attr = 'weight')
#     assert_low_p_val(MMD_out.p_values)


#     label_dict = {'1':{j:i for j,i in enumerate(['a']*8 + ['b']*3)}, 
#               '2':{j:i for j,i in enumerate(['a']*4 + ['b']*7)}}
#     kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_dict)
#     MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
#     assert_low_p_val(MMD_out.p_values)
#     assert_low_p_val(MMD_out.p_values, "label_dict")


#     def label_function(X):
#         m = np.mean(X,axis = 0)
#         return {i:str(np.round(m[i],1)) for i in range(len(m))}

#     kernel = [{"name": "weisfeiler_lehman", "n_iter": 2}, {"name": "vertex_histogram"}]
#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_labels= label_function)
#     MMD_out.fit(kernel = kernel, mmd_estimators = 'MMD_u', node_label = 'label')
#     assert_low_p_val(MMD_out.p_values, "label_function")


#     def attr_function(X):
#         return np.expand_dims(np.mean(X,axis = 0),axis=1)

#     MMD_out = MMD()
#     MMD_out.estimate_graphs(X1,X2,window_size=400, alpha = np.exp(np.linspace(-5,-2,100)),beta = 0.5, nonparanormal=False,scale = False, set_attributes = attr_function)
#     MMD_out.fit(kernel = 'GNTK', mmd_estimators = 'MMD_u', num_layers = 2, num_mlp_lauers = 2, jk = False, scale = 'degree', node_attr = 'attr', verbose =True)
#     assert_low_p_val(MMD_out.p_values, "attr_function")



























