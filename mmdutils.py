
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import matplotlib
from matplotlib.cm import get_cmap
import pandas as pd
import os
from scipy import linalg
from scipy import sparse
from scipy import stats


def readfoldertopanda(path):
    """
    Read files in a folder and concatenate them into an pandas data frame
    """
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # read and append
    df = []
    for file in onlyfiles:
        df.append(pd.read_pickle(os.path.join(path, file)))

    return pd.concat(df)

def plot_corr(A, ax=None, dw=0.125, cbar_length=0.8, max_w = None):

    # Normalize colormap
    min_w = np.min(np.triu(A))
    if max_w is None:
        max_w = np.max(np.triu(A))

    disc_min_w = dw * np.floor(min_w / dw)
    disc_max_w = dw * np.ceil(max_w / dw)
    bounds = np.linspace(
        disc_min_w, disc_max_w, np.round((disc_max_w - disc_min_w) / dw).astype(int) + 1
    )
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = sns.color_palette("rocket_r", as_cmap=True) #sns.diverging_palette(220, 10, as_cmap=True)

    # Draw heatmap
    ax = sns.heatmap(
        A,
        cmap=cmap,
        center=0,
        vmax=max_w,
        vmin=min_w,
        square=True,
        mask=A == 0,
        cbar_kws=dict(use_gridspec=False, location="bottom", shrink=cbar_length),
        norm=norm,
        ax=ax,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Draw frame
    ax.axhline(y=0, color="k", linewidth=2)
    ax.axhline(y=A.shape[1], color="k", linewidth=2)
    ax.axvline(x=0, color="k", linewidth=2)
    ax.axvline(x=A.shape[0], color="k", linewidth=2)

    return ax



def PlotROCGeneral(df, power_measure, comparison, n1 , nr_nodes_1, n2 = None, nr_nodes_2 = None , figsize = (20,10), graph_statistic_list = None, graph_kernel_list = None, title = True, legend_title = None):
    """
    Plot ROC curves


    :param df: Data frame s

    """
    if n2 is None:
        n2 = n1
    if nr_nodes_2 is None:
        nr_nodes_2 = nr_nodes_1

    fig, ax = plt.subplots(figsize = figsize)

    markers = ['o', 'v', 's', 'D','' ]
    linestyles = ['', '', '', '', '-']
    label = []

    name = "Dark2" 
    cmap = get_cmap(name, lut = 8) # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list

    # Which graph difference are we plotting, etc the average degree ratio of the two graphs

    v = sorted(np.unique(df[comparison[0]]))[comparison[1]]

    ab_line_plotted = False

    cnt = 0
    for graph_kernel in graph_kernel_list:

        tmp = df.loc[df['GraphKernel'] == graph_kernel]

        for stat in power_measure:
            # print(comparison[0])
            # print(v)
            # print(n1)
            # print(n2)
            # print(nr_nodes_1)
            # print(nr_nodes_2)
            # print(tmp['alpha'].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)])

            if not ab_line_plotted:
                ax.plot(tmp['alpha'].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                tmp['alpha'].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                color = 'black')
                ab_line_plotted = True
            

            ax.plot(tmp['alpha'].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                    tmp[stat].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                    color =colors[cnt], linestyle =linestyles[cnt], marker = markers[cnt], label=str(v))
            
            label.append(stat)

            cnt +=1
            if cnt >= len(markers):
                print("No more markers defined")
                break

        if cnt >= len(markers):
            #print("No more markers defined")
            break

        # check if graph statistic is found in the current graph kernel run

        len_graph_stat = len(graph_statistic_list)
        if  len_graph_stat > 0:
            this_round = []
            for i in range(len_graph_stat):
                print(graph_statistic_list[i])
                if graph_statistic_list[i] in tmp.columns:
                    ax.plot(tmp['alpha'].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                            tmp[graph_statistic_list[i]].loc[(tmp[comparison[0]] == v) & (tmp['n'] == n1) & (tmp['m'] == n2) & (tmp['nr_nodes_1'] == nr_nodes_1) & (tmp['nr_nodes_2'] == nr_nodes_2)], 
                            color =colors[cnt], linestyle =linestyles[cnt], marker = markers[cnt], label=str(v))
                    label.append(graph_statistic_list[i])
                    this_round.append(graph_statistic_list[i])
                    cnt +=1
                    if cnt >= len(markers):
                        print("No more markers defined")
                        break
                else:
                    print(str(graph_statistic_list[i]) + " Not found")
            
            for i in this_round:
                graph_statistic_list.remove(i)
            



    h, l = ax.get_legend_handles_labels()
    # ax.legend(handles=zip(h[::3], h[1::3], h[2::3]), labels=l[::3], 
    #         handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, title = 'Degree ratio')
    # legend to indicate which ratio is used
    first_legend = ax.legend(handles=h[::cnt], labels=l[::cnt], 
            handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, title = 'Degree ratio', bbox_to_anchor=(1, 0.7))

    # Add the legend manually to the current Axes.
    plt.gca().add_artist(first_legend)
    # legend to indicate which marker is what
    ax.legend(handles=h[:cnt], labels=label, 
            handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, title = 'Type', bbox_to_anchor=(1, 0.4))

    ax.set_xlabel('alpha')
    ax.set_ylabel('Power')


    plt.show()


def plotVaryingBGDEG(df, param_vary_name, params_fixed, mmd_stat = "MMD_b", color_name = "viridis", set_legend = True, disp_title = True, legend_title = None):

        _, ax = plt.subplots(figsize = (20,12))

        
        # select the fixed parameters
        bool_index = np.array([1] * df.shape[0]) 
        for k,v in params_fixed.items():
                bool_index = bool_index * np.array(df[k] == v, dtype = int)
        
        df = df.loc[bool_index != 0]

        #print(df.head())

        if param_vary_name in df.columns:
            df[param_vary_name].fillna(value=99999, inplace=True)

            params_vary = sorted(np.unique(df[param_vary_name]))
        else:
            params_vary = []
        
        cmap = get_cmap(sns.color_palette(color_name, as_cmap=True), lut = len(params_vary)) # type: matplotlib.colors.ListedColormap
        colors = cmap.colors[::int(len(cmap.colors)/(len(params_vary)+1))]  # type: list

        label = []


        ab_line_plotted = False

        no_param = len(params_vary) == 0

        if no_param:
            params_vary = [1]


        # plot each varying parameter
        for cnt, param in enumerate(params_vary):

                if no_param:
                    tmp = df
                else:
                    tmp = df.loc[df[param_vary_name] == param]

                if not ab_line_plotted:
                        y = tmp['alpha']
                        if len(y)>0:
                                ax.plot(y, y, color = 'black')
                                ab_line_plotted = True
                                #label.append('line')

                ax.plot(tmp['alpha'],tmp[mmd_stat], color =colors[cnt], label=str(param))

                label.append(str(param))
                



        #ax.legend(label)
        if set_legend:
            h, l = ax.get_legend_handles_labels()

            leg = ax.legend(handles=h, labels=label, 
                    handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, bbox_to_anchor=(1, 0.6), fontsize = 30)

            if legend_title is None:
                leg.set_title(param_vary_name, prop={'size':40})
            else:
                leg.set_title(legend_title, prop={'size':40})

        ax.set_xlabel('Type I error', fontsize = 20)
        ax.set_ylabel('Power', fontsize = 20)

        if disp_title:
            ax.set_title(str(params_fixed), fontsize = 20)


        plt.show()



def findAUC(keys:list, params:list, stats:list, df):
    """
    
    :param keys: graph specific statitics. number of nodes, number of samples,...
    :param params: kernel specific parameter names
    :param stats: The statistic the auc should be calculated for
    """
    from scipy.integrate import simps

    all_keys = keys + params

    unique_combination = df[all_keys].drop_duplicates()
    auc = []

    #  For each unique combination calculate the auc
    for i in range(unique_combination.shape[0]):

        row = unique_combination.iloc[[i]].copy()
        tmp = df.reset_index().merge(row, on = all_keys)

        for stat in stats:
            newname = stat + "_auc"
            row[newname] = simps(y = tmp[stat], x = tmp['alpha'])

        auc.append(row)

    return pd.concat(auc)



def _fast_mat_inv_lapack(Mat):
    """
        Compute the inverse of a positive semidefinite matrix.
    
        This function exploits the positive semidefiniteness to speed up
        the matrix inversion.
            
        Parameters
        ----------
        Mat : 2D numpy.ndarray, shape (N, N)
            A positive semidefinite matrix.
    
        Returns
        -------
        inv_Mat : 2D numpy.ndarray, shape (N, N)
            Inverse of Mat.
        """

    zz, _ = linalg.lapack.dpotrf(Mat, False, False)
    inv_Mat, info = linalg.lapack.dpotri(zz)
    inv_Mat = np.triu(inv_Mat) + np.triu(inv_Mat, k=1).T
    return inv_Mat



def _comp_EBIC(W, C_samp, C_null, L, beta, Knull, input_matrix_type):
    """
        Compute the extended Bayesian Information Criterion (BIC) for a network. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix.
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
        L : int
            Number of samples.
        beta : float
            Parameter for the extended BIC. 
        K_null: int
            Number of parameters of the null correlation matrix.
        input_matrix_type: string
	    Type of matrix to be given (covariance or precision)
    
        Returns
        -------
        EBIC : float
            The extended BIC value for the generated network.
        """
    k = Knull + np.count_nonzero(np.triu(W, 1))  + np.count_nonzero(np.diag(W))
    EBIC = (
        np.log(L) * k
        - 2 * L * _comp_loglikelihood(W, C_samp, C_null, input_matrix_type)
        + 4 * beta * k * np.log(W.shape[0])
    )
    return EBIC


def _comp_loglikelihood(W, C_samp, C_null, input_matrix_type):
    """
        Compute the log likelihood for a network. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
        input_matrix_type: string
	    Type of matrix to be given (covariance or precision)
    
        Returns
        -------
        l : float
            Log likelihood for the generated network. 
        """
    if input_matrix_type == "cov":
        Cov = W + C_null

        iCov, w = _truncated_inverse(Cov)
        # iCov = np.real(np.matmul(np.matmul(v, np.diag(1 / w)), v.T))
        l = (
            -0.5 * np.sum(np.log(w))
            - 0.5 * np.trace(np.matmul(C_samp, iCov))
            - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
        )
    else:
        iCov = W + C_null
        w, v = np.linalg.eig(iCov)

        if np.min(w) < 0:
            v = v[:, w > 0]
            w = w[w > 0]
        l = (
            0.5 * np.sum(np.log(w))
            - 0.5 * np.trace(np.matmul(C_samp, iCov))
            - 0.5 * iCov.shape[0] * np.log(2 * np.pi)
        )

    return np.real(l)


def _remedy_degeneracy(C_samp, rho = 1e-3, scale = True):
    """
    Create an invertible matrix from a degenerate one

    Parameters
    --------------
    scale: scale the matrix to get correlation?
    """
    w, v = np.linalg.eigh(C_samp)
    if np.min(w) < rho:

        w[w<0] = rho

        # Compute the precision matrix from covariance matrix with a ridge regularization.
        lambda_hat = 2 / (np.sqrt(w ** 2) + np.sqrt(w ** 2 + 8 * rho))
        iC = np.matmul(np.matmul(v, np.diag(lambda_hat)), v.T)

        # Compute the correlation matrix from the precision matrix
        _C_samp = np.linalg.inv(iC)
        
        # Homogenize the variance 
        std_ = np.sqrt(np.diag(_C_samp))
        if scale:
            C_samp = _C_samp / np.outer(std_, std_)
        else:
            C_samp = _C_samp

    return C_samp


def _truncated_inverse(X):
    """
    Inverse a matrix based on truncated svd
    """

    u, l, vt = np.linalg.svd(X)

    v = vt.T

    v = v[:, l > 0]
    u = u[:, l > 0]
    l = l[l>0]
    l_cum = np.cumsum(l)/np.sum(l)

    l_store = l.copy()
    l = l[l_cum < 0.9999]
    if len(l) <= 0:
        # there was only one eigenvalue containing all the variance, l looked something like [21.2, e-18, e-18,...]
        l = np.array([l_store[0]])

    iX = np.zeros(X.shape)
    for i in range(len(l)):

        iX += (1.0/l[i]) * np.outer(v[:, i], u[:, i])

    return iX, l



def mahalanobis_distance(x, mu, Sigma_inv):
    """
    Return the Mahalanobis distance between mu and all vector in x

    :param x: A N by p matrix, N number of data points
    
    """


    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d[i] = np.dot(x[i,:] - mu , np.dot(Sigma_inv, x[i,:] - mu ) )

    return d
    
