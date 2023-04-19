import sys, os
sys.path.append('../src')
import numpy as np
import networkx as nx
from GTST import RandomWalk, GNTK, DK, WWL, BoostrapMethods, MMD_b, MMD_l, MMD_u, MONK_EST  
from tests.utils import generate_H0_false,assert_positive_eig,  generate_H0_false_but_same_topology, generate_H0_false_directed, assert_low_p_val_from_K

def test_RandomWalk_ARKU_plus():
    g1, g2 = generate_H0_false()
    rw = RandomWalk(g1 + g2,r = 6, c = 1e-5)
    rw.fit('ARKU_plus', verbose=True)
    assert_positive_eig(rw.K)

    # Test MMD, should give very low p-val
    pval = BoostrapMethods(list_of_functions=[MMD_u, MMD_b, MMD_l, MONK_EST], function_arguments={'MMD_u':{'n1':len(g1), 'n2':len(g2)}, 
                                                                                        'MMD_b':{'n1':len(g1), 'n2':len(g2)}, 
                                                                                        'MMD_l':{'n1':len(g1), 'n2':len(g2)}, 
                                                                                        'MONK_EST':{'n1':len(g1), 'n2':len(g2), 'Q':5}})
    pval.Bootstrap(rw.K, 1000)
    assert np.all(np.fromiter(pval.p_values.values(), dtype=float) < 0.01), f"Some p value too high {pval.p_values}"

    
    g1, g2 = generate_H0_false_but_same_topology()
    rw = RandomWalk(g1 + g2,r = 6, c = 1e-5,edge_attr='weight')
    rw.fit('ARKU_plus', verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)

    rw = RandomWalk(g1 + g2,r = 6, c = 1e-8,node_attr='attr')
    rw.fit('ARKU_plus', verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)




def test_RandomWalk_ARKL():
    g1, g2 = generate_H0_false()
    unique_node_labels= set(np.concatenate([list(nx.get_node_attributes(g, 'label').values()) for g in g1+g2]))
    rw = RandomWalk(g1 + g2, r = 6, c = 0.001, node_label='label', unique_node_labels = unique_node_labels)
    rw.fit('ARKL', verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)

    g1, g2 = generate_H0_false_but_same_topology()
    unique_node_labels= ['a', 'b', 'c']
    rw = RandomWalk(g1 + g2, r = 6, c = 0.001, node_label='label', unique_node_labels = unique_node_labels)
    rw.fit('ARKL', verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)


def test_RandomWalk_ARKU():
    g1, g2 = generate_H0_false()
    rw = RandomWalk(g1 + g2, r = 6, c = 0.001)
    rw.fit('ARKU',verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)
    # Directed
    g1, g2 =  generate_H0_false_directed()
    rw = RandomWalk(g1 + g2, r = 6, c = 0.001)
    rw.fit('ARKU',verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)


def test_RandomWalk_ARKU_edge():
    g1, g2 = generate_H0_false_but_same_topology()
    rw = RandomWalk(g1 + g2, r = 6, c = 0.0001, edge_label='edge_label',unique_edge_labels= ['a', 'b'])
    rw.fit('ARKU_edge', verbose=True)
    assert_positive_eig(rw.K)
    assert_low_p_val_from_K(rw.K, g1,g2)


def test_GNTK():
    g1, g2 = generate_H0_false()

    gntk_kernel = GNTK(num_layers= 2, num_mlp_layers = 2, 
                                jk = True, scale = 'uniform', normalize=1)
    gntk_kernel.fit_all(g1 + g2,degree_as_tag=True,features= None)
    assert_positive_eig(gntk_kernel.K)
    assert_low_p_val_from_K(gntk_kernel.K, g1,g2)

    gntk_kernel = GNTK(num_layers= 2, num_mlp_layers = 2, 
                            jk = False, scale = 'uniform', normalize=1)
    gntk_kernel.fit_all(g1 + g2,degree_as_tag=True,features= None)
    assert_positive_eig(gntk_kernel.K)
    assert_low_p_val_from_K(gntk_kernel.K, g1,g2)

    # Same graphs but different attributes
    g1, g2 = generate_H0_false_but_same_topology()
    gntk_kernel = GNTK(num_layers= 2, num_mlp_layers = 2, 
                            jk = False, scale = 'degree', normalize=1)
    gntk_kernel.fit_all(g1 + g2,degree_as_tag=False,features= 'attr')
    assert_positive_eig(gntk_kernel.K)
    assert_low_p_val_from_K(gntk_kernel.K, g1,g2)


def test_DK():
    g1, g2 = generate_H0_false_but_same_topology()
    dk_kernel = DK(type = 'wl', wl_it = 2, opt_type = None, nodel_label='label')
    dk_kernel.fit_transform(g1+g2)
    assert_positive_eig(dk_kernel.K)
    assert_low_p_val_from_K(dk_kernel.K, g1,g2)


    dk_kernel = DK(type = 'sp', opt_type = None,
                            vector_size= 2, window=2, min_count=0, 
                            normalize = 0, nodel_label='label')
    dk_kernel.fit_transform(g1+g2)
    assert_positive_eig(dk_kernel.K)
    assert_low_p_val_from_K(dk_kernel.K, g1,g2)

    dk_kernel = DK(type = 'sp', opt_type = 'word2vec',
                        vector_size= 2, window=2, min_count=0, 
                        normalize = 0, nodel_label='label')
    dk_kernel.fit_transform(g1+g2)
    assert_positive_eig(dk_kernel.K)
    assert_low_p_val_from_K(dk_kernel.K, g1,g2)


def test_WWL():
    g1, g2 = generate_H0_false_but_same_topology()

    wwl_kernel = WWL(param =dict(discount=0.1, h=2, sinkhorn=False, label_name='label'))
    wwl_kernel.fit_transform(g1+g2)
    assert_positive_eig(wwl_kernel.K)
    assert_low_p_val_from_K(wwl_kernel.K, g1,g2)


    







