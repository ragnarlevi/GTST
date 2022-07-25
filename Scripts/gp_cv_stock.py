import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import scipy
import importlib
import os, sys
import seaborn as sns
import tqdm
import networkx as nx
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, WhiteKernel

from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold

from multiprocessing import Pool, freeze_support

import pickle


def fit_gp_cv(kernel, X_obs, y_obs, X_pred, scale = True):

    from sklearn.model_selection import RandomizedSearchCV

    # Create CV
    index = np.array(range(y_obs.shape[0]))
    cv = list()
    for i in [0,1,2,3]:
        test = index[i:len(index):4]
        train = index[~np.isin(index,test)]
        cv.append((train, test))

    n_iter = 120

    # rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    distributions = dict(alpha=np.linspace(0.001, 0.3, n_iter))# dict(alpha=uniform(loc = 0, scale = 0.5))

    clf = RandomizedSearchCV(gaussian_process.GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=30, normalize_y= scale),
                                                                         distributions, random_state=0, cv = cv, n_iter  = n_iter)
    search = clf.fit(X_obs, y_obs)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,normalize_y = scale, n_restarts_optimizer=30, alpha =  search.best_params_['alpha']).fit(X_obs, y_obs)

    mean_prediction, std_prediction = gp.predict(X_pred, return_std=True)

    return mean_prediction, std_prediction,  gp, search

def fit_gp_envelope(i):

    esg_data = pd.read_pickle('Yahoo/refined_no_diff.pkl')
    print(f'{i} {esg_data.shape[1]} \n')

    kernel = 1*Matern(length_scale=1, nu = 1.5)

    X = np.array(range(len(esg_data.index)))
    y = np.array(esg_data.iloc[:,i])
    obs_index = np.isfinite(y)

    y_obs = y[obs_index]
    X_obs = np.expand_dims(X[obs_index], axis = 1)
    X = np.expand_dims(X, axis=1)

    X_pred = np.expand_dims(np.array(range(X.shape[0])), axis = 1)
    mean_prediction, std_prediction,  gp, search = fit_gp_cv(kernel, X_obs, y_obs, X_pred)


    info = {'mean_prediction':mean_prediction,
    'std_prediction':std_prediction,
    'gp':gp,
    'search':search
    }


    name = esg_data.columns[i]

    out = {name:info}


    return out







if __name__ == '__main__':

    esg_data = pd.read_pickle('Yahoo/refined_no_diff.pkl')

    with Pool(6) as pool:
        L = pool.map(fit_gp_envelope, list(range(esg_data.shape[1])))

    
    with open(f'data/paper/gp_esg_stock.pkl', 'wb') as f:
            pickle.dump(L, f)
            

    