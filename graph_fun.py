import numpy as np
import networkx as nx
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

def rolling_correlation(data, window_length, period, method, prec = False):
    """
    Correlation and 
    
    """

    corrs = []
    inv_corrs = []

    index = np.array(range(data.shape[0]))[::period]


    for i in index:
        tmp_data = np.array(data.iloc[i: window_length + i])

        if tmp_data.shape[0]<window_length:
            break

        tmp_corr = pd.DataFrame(tmp_data).corr(method= method)
        corrs.append(np.array(tmp_corr))
        if prec:
            inv_corrs.append(np.linalg.inv(np.array(tmp_corr)))


    return corrs, inv_corrs


def gen_fullyconnected_threshold(corrs_sector, edge_type = None):
    """
    edge_type: None, pos, neg
    
    """
    fully_connected_graphs = []
    fully_connected_graphs_signed = []
    fully_connected_graphs_weighted = []
    thresholds = np.linspace(0, 1, 500)

    pbar = tqdm.tqdm(total= len(corrs_sector))

    for i in range(len(corrs_sector)):    
        cnt = 0
        A = corrs_sector[i].copy()

        if edge_type == "pos":
            A[A <0] = 0
        if edge_type == "neg":
            A[A >0] = 0



        np.fill_diagonal(A,0)
        G = nx.from_numpy_matrix(A)
        while nx.is_connected(G):
            A = corrs_sector[i].copy()

            if edge_type == "pos":
                A[A <0] = 0
            if edge_type == "neg":
                A[A >0] = 0

            A[np.abs(A) < thresholds[cnt]] = 0
            A[np.abs(A) > thresholds[cnt]] = 1
            G = nx.from_numpy_matrix(A)
            cnt +=1 
        pbar.update()

        if cnt == 0:
            #raise ValueError("Already 2 components")
            A = corrs_sector[i].copy()
            if edge_type == "pos":
                A[A <0] = 0
            if edge_type == "neg":
                A[A >0] = 0


            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(A.copy())
            fully_connected_graphs_weighted.append(G)


            A[np.abs(A) > 0] = 1
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(A)
            fully_connected_graphs.append(G)

            A = corrs_sector[i].copy()
            if edge_type == "pos":
                A[A <0] = 0
            if edge_type == "neg":
                A[A >0] = 0

            A[A > 0] = 1
            A[A < 0] = -1
            G  = nx.from_numpy_matrix(A)
            edges = {(i[0], i[1]): i[2] for i in list(G.edges(data = 'weight'))}
            A = np.abs(A)
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(np.abs(A))
            nx.set_edge_attributes(G, edges, "sign")
            fully_connected_graphs_signed.append(G)
        else:
            A = corrs_sector[i].copy()
            if edge_type == "pos":
                A[A <0] = 0
            if edge_type == "neg":
                A[A >0] = 0

            A[np.abs(A) < thresholds[cnt-2]] = 0
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(A.copy())
            fully_connected_graphs_weighted.append(G)


            A[np.abs(A) > thresholds[cnt-2]] = 1
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(A)
            fully_connected_graphs.append(G)

            A = corrs_sector[i].copy()
            if edge_type == "pos":
                A[A <0] = 0
            if edge_type == "neg":
                A[A >0] = 0

            A[np.abs(A) < thresholds[cnt-2]] = 0
            A[A > thresholds[cnt-2]] = 1
            A[A < -thresholds[cnt-2]] = -1
            G  = nx.from_numpy_matrix(A)
            edges = {(i[0], i[1]): i[2] for i in list(G.edges(data = 'weight'))}
            A = np.abs(A)
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(np.abs(A))
            nx.set_edge_attributes(G, edges, "sign")
            fully_connected_graphs_signed.append(G)


    pbar.close()

    return fully_connected_graphs_weighted, fully_connected_graphs, fully_connected_graphs_signed

    
def plot_corr(corrs, idx = None):

    corr_diag = np.zeros((len(corrs), len(corrs[0][np.triu_indices(corrs[0].shape[0], k = 1)])))

    corrs[0][np.triu_indices(corrs[0].shape[0], k = 1)]
    for i in range(len(corrs)):
        corr_diag[i,:] = corrs[i][np.triu_indices(corrs[0].shape[0], k = 1)]
        # corr_diag[i,np.abs()]


    fig, ax = plt.subplots(1,1, figsize = (30, 10))


    if idx is None:
        for i in range(corr_diag.shape[1]):
            ax.plot(range(corr_diag.shape[0]), corr_diag[:,i])
    else:
        for i in idx:
            ax.plot(range(corr_diag.shape[0]), corr_diag[:,i])


def avg_degree(G, normalize = False, weight = None):
    
    if normalize:
        avg = np.average(list(dict(G.degree(weight=weight)).values()))/(G.number_of_nodes()-1)
    else:
        avg = np.average(list(dict(G.degree(weight=weight)).values()))
    return avg

def avg_degree_list(Gs, normalize = False, weight = None):
   return [avg_degree(G, normalize = normalize, weight = weight) for G in Gs]


def cnt_balance(G):
    # Loop thorugh triangles and set as unbalanced until under certain threshold
    triangles = [c for c in nx.cycle_basis(G) if len(c)==3]
    cnt_unbalanced = 0
    cnt_balanced = 0
    for cycle in triangles:
        
        # This cycle is balanced
        if G.edges[cycle[0],cycle[1]]['sign']*G.edges[cycle[1],cycle[2]]['sign']*G.edges[cycle[0],cycle[2]]['sign'] == 1.0:
            cnt_balanced += 1
        else:
            cnt_unbalanced +=1
    
    return cnt_balanced, cnt_unbalanced

def cnt_pos_neg(G, pos = 1):
    return len([(edge[0], edge[1]) for edge in G.edges(data = 'sign') if edge[2] == pos])


def plot_sign_network(Gs, index, node_attributes, figsize, edge_name = 'sign', pos = None):

    fig, ax = plt.subplots(1,len(index), figsize = figsize )


    cnt = 0
    if pos is None:
        pos = nx.spring_layout(Gs[0])
    for i in index:

        G = Gs[i]

        pos_edge = [(edge[0], edge[1]) for edge in G.edges(data = edge_name) if edge[2] > 0]
        pos_width = np.array([edge[2] for edge in G.edges(data = edge_name) if edge[2] > 0])
        neg_edge = [(edge[0], edge[1]) for edge in G.edges(data = edge_name) if edge[2] < 0]
        neg_width = np.abs(np.array([edge[2] for edge in G.edges(data = edge_name) if edge[2] < 0]))
        nx.draw_networkx_nodes(G, pos = pos, ax = ax[cnt])
        nx.draw_networkx_labels(G, pos = pos, labels =node_attributes, ax = ax[cnt])
        if len(pos_edge)>= 1:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=pos_edge,
                width=4*pos_width/np.max(pos_width),
                alpha=0.5,
                edge_color="tab:green",
                ax = ax[cnt]
            )
        if len(neg_width)>= 1:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=neg_edge,
                width=4*neg_width/np.max(neg_width),
                alpha=0.5,
                edge_color="tab:red",
                ax = ax[cnt])

        cnt += 1


