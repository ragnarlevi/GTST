
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.cm import get_cmap
import pandas as pd
import os



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



def PlotROCGeneral(df, power_measure, comparison, n1 , nr_nodes_1, n2 = None, nr_nodes_2 = None , figsize = (20,10), graph_statistic_list = None, graph_kernel_list = None):
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


def plotVaryingBGDEG(df, param_vary_name, params_fixed, mmd_stat = "MMD_b", color_name = "viridis"):

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
        h, l = ax.get_legend_handles_labels()

        leg = ax.legend(handles=h, labels=label, 
                handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}, bbox_to_anchor=(1, 0.4), fontsize = 14)

        leg.set_title(param_vary_name, prop={'size':20})

        ax.set_xlabel('alpha', fontsize = 20)
        ax.set_ylabel('Power', fontsize = 20)
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