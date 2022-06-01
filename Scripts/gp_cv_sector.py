import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import scipy
import importlib
import os, sys
import seaborn as sns
import tqdm
from pandas_datareader import data
import networkx as nx
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, WhiteKernel
import GPy
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, freeze_support, Process

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir) 

import pickle

import argparse

parser = argparse.ArgumentParser()

args = parser.parse_args()


def Gpy_cv(X,y, kernel, alphas):

    data = pd.DataFrame()
    for alpha in alphas:


        cv = list()
        index = np.array(range(y.shape[0]))
        Rsq = []
        for i in [0,1,2,3]:
            test = index[i:len(index):4]
            train = index[~np.isin(index,test)]


            Xtrain = X[train]
            ytrain = y[train]
            m = GPy.models.GPRegression(Xtrain, ytrain, kernel)
            m.Gaussian_noise.variance = alpha
            m.Gaussian_noise.variance.fix()
            m.optimize()

            Xtest = X[test]
            ytest = y[test]
            mean_prediction, _ = m.predict(Xtest, full_cov=False)
            Rsq.append(1 - np.sum((ytest - mean_prediction)**2)/np.sum((ytest - np.mean(ytest))**2))

        data = pd.concat((data, pd.DataFrame(
                        {'alpha':alpha,
                        'Rsq':np.mean(Rsq)}, index = [0]

                        )), ignore_index=True) 
    
    # Select best variance and fit best using best variance
    variance = data['alpha'].loc[data['Rsq'] == data['Rsq'].max()].iloc[0]
    if np.where(variance == alphas)[0][0] == len(alphas)-1:
        print(f'last alpha selected ')



    m = GPy.models.GPRegression(X, y, kernel)
    m.Gaussian_noise.variance = variance
    m.Gaussian_noise.variance.fix()
    m.optimize()
    mean_prediction, var_prediction = m.predict(X, full_cov=False)
    residuals = y - mean_prediction

    Rsq = 1 - np.sum((y - mean_prediction)**2)/np.sum((y - np.mean(y))**2)



    return m, kernel, residuals, variance,  data, Rsq


def get_index(tick):
    """
    Function that takes the sp500 index from yahoo
    """
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # ESG historical data (only changes yearly)
    url_esg = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols={tick}&range=10y&interval=1d&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance"
    response = requests.get(url_esg, headers=headers)
    if response.ok:
        sp500 = pd.DataFrame({'date':pd.to_datetime(response.json()['spark']['result'][0]['response'][0]['timestamp'], unit= 's'),
                              'price':response.json()['spark']['result'][0]['response'][0]['indicators']['quote'][0]['close']})
    
    else:
        print("Empty data frame")
        sp500 = pd.DataFrame()



    return sp500





sp500 = get_index('^GSPC')
sp500['date'] = pd.to_datetime(sp500['date']).dt.date
sp500['return'] = 1 + sp500['price'].pct_change()
sp500['log_return'] = np.log(sp500['price']).diff()
sp500 = sp500.iloc[:,:].dropna(axis= 0)


Price = pd.read_pickle('Yahoo/sector_index_price.pkl')
log_return = np.log(Price.pct_change()+1).dropna()


sector_gp = pd.read_pickle('Yahoo/sector_gp.pkl')

price_all = pd.merge(Price, sp500, left_index=True, right_on= 'date')
price_all = pd.merge(price_all, sector_gp, left_on='date', right_index=True)




print("data load finished")


import warnings
warnings.filterwarnings("ignore")

pbar = tqdm.tqdm( total= len(range(0,price_all.shape[0]-125, 5)))





def cv_fun(sector, price_all):
    df = pd.DataFrame()
    residuals_df = pd.DataFrame()
    alphas = np.linspace(0, 2, 50)

    for i in range(0,price_all.shape[0]-125, 5):


        # With ESG FIT
        X = np.array(price_all[['price', sector + '_y']].iloc[i:(i+125)])
        y = np.array(price_all[[sector + '_x']].iloc[i:(i+125)])
        scaler = StandardScaler()
        y = scaler.fit_transform(y)
        X = scaler.fit_transform(X)
        k = GPy.kern.Matern32(input_dim = 2, ARD = True)
        m_ESG, kernel_ESG, resid_ESG, variance_esg,  _, Rsq_ESG = Gpy_cv(X,y, k, alphas)

        # Without ESG fit
        X = np.array(price_all[['price']].iloc[i:(i+125)])
        y = np.array(price_all[[sector + '_x']].iloc[i:(i+125)])
        scaler = StandardScaler()
        y = scaler.fit_transform(y)
        X = scaler.fit_transform(X)
        k = GPy.kern.Matern32(input_dim = 1, ARD = True)
        m, kernel, resid, variance,  _ , Rsq = Gpy_cv(X, y, k, alphas)
        
        # Safe data
        df = pd.concat((df,pd.DataFrame({'sector':sector, 
                        'time':i,
                        'lengthscale_1': np.array(kernel_ESG.lengthscale)[0],
                        'lengthscale_2': np.array(kernel_ESG.lengthscale)[1],
                        'objective_esg':m_ESG.objective_function(),
                        'Rsq_esg':Rsq_ESG,
                        'variance_esg':variance_esg,
                        'lengthscale_m': np.array(kernel.lengthscale)[0],
                        'objective_m':m.objective_function(),
                        'Rsq_m':Rsq,
                        'variance':variance}, index = [0])),ignore_index= True)

        residuals_df = pd.concat((residuals_df,pd.DataFrame({'sector':[sector]*len(resid), 
                        'time':[Price.index[i+125]]*len(resid),
                        'order':list(range(len(resid))),
                        f'{sector}_residuals_esg':resid_ESG[:,0],
                        f'{sector}_residuals':resid[:,0]
                        }, index = list(range(len(resid))))),ignore_index= True)

        pbar.update()

    pbar.close()


    sector_dict = dict()

    sector_dict['info'] = df
    sector_dict['residuals'] = residuals_df


    with open(f'data/gp_cv/{sector}.pkl', 'wb') as fp:
        pickle.dump(sector_dict, fp)


# 'Healthcare', 'Industrials', 'Consumer Cyclical', 'Technology',
#        'Consumer Defensive', 'Utilities', 'Financial Services',
#        'Basic Materials', 'Real Estate', 'Energy', 'Communication Services'

# Done Energy Utilities Healthcare Industrials Consumer Cyclical,  Technology, Consumer Defensive, Financial Services
 
if __name__ ==  '__main__':
    p1 = Process(target=cv_fun, args= ('Basic Materials',price_all))
    p1.start()
    p2 = Process(target=cv_fun, args= ('Real Estate',price_all))
    p2.start()
    p3 = Process(target=cv_fun, args= ('Communication Services',price_all))
    p3.start()
    #p4 = Process(target=cv_fun, args= ('Financial Services',price_all))
    #p4.start()










