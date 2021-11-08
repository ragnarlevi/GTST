# Functions for covariance estimation, regerssion and projection

from logging import warning
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import shape
import scipy
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from scipy.optimize import minimize, brentq
import statsmodels.api as sm
import warnings


def pca_projection(x, axis, cov = None, scale = 'empirical_correlation' ):
    """
    project onto the princicap compoents axis


    :parma cov: covariance matrix
    :param x: numpy array, data N x p. N number of data points, p number of features
    :param axis: np list. The principal component axes we want to project onto
    :param scale: Type of scaling
    """

    

    # PCA
    if scale == 'empirical_correlation':
        cov = np.cov(x.T)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        corr = corrMat(cov)
        _, v = np.linalg.eigh(corr)
        v = np.fliplr(v)
        
        v = v[:, axis].reshape((v.shape[0], len(axis)))
        pca_projection_out = projection(x_scaled, v)*scaler.scale_ + scaler.mean_

    elif scale == 'covariance':
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        _, v = np.linalg.eigh(cov)
        v = np.fliplr(v)

        v = v[:, axis].reshape((v.shape[0], len(axis)))
        pca_projection_out = projection(x - scaler.mean_, v) +scaler.mean_

    elif scale == 'robust_correlation' and not cov is None:
        
        sd = np.sqrt(np.diag(cov))
        mu = np.median(x, axis = 0)
        x_scaled = scale_data(x, mu, sd)

        corr = corrMat(cov)
        _, v = np.linalg.eigh(corr)
        v = np.fliplr(v)

        v = v[:, axis].reshape((v.shape[0], len(axis)))
        pca_projection_out = projection(x_scaled, v)*sd + mu


    return pca_projection_out


def projection(x, v):
    """
    project x onto v
    :param x: A n times p data matrix
    """

    return np.dot(x, v).dot(v.T)


def corrMat(cov):
    """
    Calculate the covariance matrix
    :param cov: Covariance matrix
    """
 
    corr_mat = np.zeros(cov.shape)
 
    for i in range(cov.shape[0]):
 
        for j in range(cov.shape[0]):

            # not here that we are just normalizing the covariance matrix
            corr_mat[i,j] = cov[i,j] / np.sqrt(cov[i,i] * cov[j,j])
 
    return corr_mat


def scale_data(x, mu, sd):

    x_scaled = x.copy()
    for j in range(x.shape[1]):
        x_scaled[:,j] = (x_scaled[:,j] - mu[j])/sd[j]

    return x_scaled

    

def subtract_projection(x, axis, cov = None, scale = 'empirical_correlation'):

    """
    Subtract projection along axis from x

    :parma cov: covariance matrix
    :param x: numpy array, data N x p. N number of data points, p number of features
    :param axis: np list. The principal component axes we want to project onto
    :param scale: Type of scaling
    """
    
    return x - pca_projection(x, axis, cov, scale)



def remove_effect(x, effect_type, market_return = None, cov = None):
    """
    Remove market effect from x:

    :param x: n times p data matrix
    :param effect_type: string: pca, rob_pca, reg, rob_reg
    """

    if effect_type == "pca":
        projected_x = pca_projection(x, axis = np.array([0]), scale = 'empirical_correlation' )
        x_no_market_effect = x - projected_x

    elif effect_type == "rob_pca":
        if cov is None:
           raise ValueError("Please parse the robust covariance matrix")

        projected_x = pca_projection(x, axis = np.array([0]), cov = cov, scale = 'robust_correlation' )
        x_no_market_effect = x - projected_x

    elif effect_type == "reg":

        if market_return is None:
            raise ValueError("Please parse the market return vector")

        x_market_effect = x.copy()

        for j in range(x.shape[1]):
            covariates = pd.DataFrame({'market': market_return})
            covariates['const'] = 1

            model = sm.OLS(x[:,j], covariates)
            results = model.fit()
            x_market_effect[:, j] = model.predict(results.params)

        x_no_market_effect = x - x_market_effect

    elif effect_type == "rob_reg":

        if market_return is None:
            raise ValueError("Please parse the market return vector")

        x_market_effect = x.copy()

        for j in range(x.shape[1]):
            covariates = pd.DataFrame({'market': market_return})
            covariates['const'] = 1

            model = sm.QuantReg(x[:,j], covariates)
            results = model.fit(q = 0.5, max_iter = 5000)
            x_market_effect[:, j] = model.predict(results.params)

        x_no_market_effect = x - x_market_effect
    
    else:
        raise ValueError("effect_type: pca, rob_pca, reg, rob_reg")

    
    return x_no_market_effect



def construct_covariance(x, sector_dict, step = 30, remove_market = True, remove_market_type = 'pca', reconstruction_type = 'empirical', market_return = None):
    """

    :param x: pandas, n times p, log returns of stocks
    :param sector_dict: dictionary where key is sector and values are companies within that sector
    :param step: How many data points for each covariance construction
    :param remove_market: Should the market effect be removed
    :param reconstruction_type: How should the covariance matrix be calculated after the market effect has been removed? empirical or robust
    :param market_return: Vector containing the market return, only used if market effects are removed by regression. Should have same order as x
    """


    # dictionary that will store covariances, and nodes at each snap shot, for each sector
    covariance_dict = dict()
    nodes_dict = dict()
    for k in sector_dict.keys():
        covariance_dict[k] = list()
        nodes_dict[k] = list()

    # time_index that will store the time of each covariance
    time_index = list()

    n = x.shape[0]

    for i in range(0, n, step):
        
        if i > n-step:
            break


        for k in sector_dict.keys():
            #print(k)
            x_sector = x.loc[:,x.columns.isin(sector_dict[k])].iloc[i:(step+i)].copy()

            if not market_return is None:
                market = market_return[i:(step+i)]

            nr_data_points_before = float(x_sector.shape[0])
            # drop rows with na values (so we can keep all nodes)
            x_sector_drop = x_sector.dropna(axis = 0)
            if nr_data_points_before*0.7 > float(x_sector_drop.shape[0]):
                # warnings.warn("rows contaning NA removed. Post matrix has less the 70pct data points. Will instead remove the columns with NA vale")
                x_sector_drop = x_sector.dropna(axis = 1)

            # we need to keep track of missing data points in our market_return as well
            if not market_return is None:
                index_to_keep = np.array(range(market.shape[0]))
                index_to_keep = index_to_keep[x_sector.index.isin(x_sector_drop.index)]
                market = market[index_to_keep]

            x_sector = x_sector_drop.copy()

            nodes_dict[k].append(list(x_sector.columns))
            x_sector_array = np.array(x_sector)

            time_index.append(list(x_sector.index[i:(step+i)]))

            if x_sector.shape[0] != x_sector_array.shape[0]:
                raise ValueError("Some error during casting pandas to array, x_sector pandas and x_sector_array")

            if remove_market:
                if remove_market_type == 'pca':
                    x_no_market = remove_effect(x_sector_array, effect_type = 'pca')
                elif remove_market_type == 'rob_pca':
                    mu, V, dist = S_estimation_cov(x_sector_array, initial='K', maxsteps=5, propmin=0.01, qs=2, maxit=200, tol=1e-4, corr=False)
                    x_no_market = remove_effect(x_sector_array, effect_type = remove_market_type, cov = V)
                elif remove_market_type in ['reg', 'rob_reg']:
                    x_no_market = np.log(remove_effect(np.exp(x_sector_array), effect_type = remove_market_type, market_return= market))
                else:
                    raise ValueError("remove_market_type should be pca, rob_pca, reg, rob_reg ")

                if np.sum(np.isnan(x_no_market)) > 0:
                    warnings.warn("nan value in x_no_market")


                if reconstruction_type == 'empirical':
                    covariance_dict[k].append(np.corrcoef(x_no_market.T))
                elif reconstruction_type == 'robust':
                    mu, V, dist = S_estimation_cov(x_no_market, initial='K', maxsteps=5, propmin=0.01, qs=2, maxit=200, tol=1e-4, corr=False)
                    V_cor = corrMat(V)
                    covariance_dict[k].append(V_cor)
                else:
                    raise ValueError("reconstruction_type should be emprical or robust ")
            else:
                covariance_dict[k].append(np.corrcoef(x_sector_array.T))


    return covariance_dict, nodes_dict, time_index

    

    

    





    
        










##################################################### Rocke code taken from https://github.com/msalibian/RobStatTM/blob/master/R/Multirobu.R and adjusted ###################################

def rho_rocke(t, gamma, q = 2):
    """
    Rocke rho function. 
    """

    rho = ((t-1.0)/(2.0*q*gamma)) * (q + 1 - np.power((t-1.0)/gamma, q )) + 0.5
    rho[t < 1.0 - gamma] = 0.0
    rho[t >= 1.0 + gamma] = 1.0
    return rho



def WRoTru(tt, gamma, q):

    ss = (tt - 1) / gamma 
    w = 1 - np.power(ss,q)
    w *= (q + 1.0) / (2.0 * q * gamma)
    w[ np.abs(ss) > 1 ] = 0
    return w




def rho_rocke_obj(sig, t, gamma, q, delta):

    return np.mean(rho_rocke(t/sig, gamma, q)) - delta


def mahalanobis_distance(x, mu, Sigma_inv):
    """
    Return the Mahalanobis distance between mu and all vector in x

    :param x: A N by p matrix, N number of data points
    
    """


    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):

        d[i] = np.dot(x[i,:] - mu , np.dot(Sigma_inv, x[i,:] - mu ) )

    return d

def svd_trunc_mahalanobis_distance(x, mu, Sigma, pct_cutoff = 0.99):

    """
    Return the Mahalanobis distance between mu and all vector in x, using a truncated svd inversion
    
    """

    u, s, vt = np.linalg.svd(Sigma)

    ut = u.T
    v = vt.T

    k = len(s[ np.cumsum(s) / np.sum(s) < pct_cutoff])

    SS = np.dot(v[:, :k], np.diag(np.reciprocal(s[:k]))).dot(ut[:k, :])

    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):

        d_tmp = 0.0

        # for j in range(k):

        #     #d_tmp += np.dot(x[i,:] - mu, np.dot((1.0/s[j])*np.outer(v[:,j], ut[j,:]), x[i,:] - mu))

        d[i] = np.dot(x[i,:] - mu, np.dot(SS, x[i,:] - mu))

    return d




def consRocke(p, n, initial):
    # The constants in Section 6.10.4 of Maronna et al. (2019)
    
    if initial == "mve": 
        beta = np.array([-5.4358, -0.50303, 0.4214])
    else:
        beta = np.array([-6.1357, -1.0078, 0.81564])

    if p >= 15:
        a = np.array([1, np.log(p), np.log(n)])
        alpha = np.exp( np.sum( beta*a ) )
        gamma = scipy.stats.chi2.ppf(1-alpha, df=p)/p - 1
        gamma = np.min([gamma, 1])
    else:
        gamma = 1
        alpha = 1e-6
    return gamma, alpha



def S_estimation_cov(x, mahalanobis_type = 'svd', initial='K',  maxsteps=5, propmin=2, qs=2, maxit=50, tol=1e-4, corr=False, mahalanobis_cutoff = 0.99):

    """
    S-estimation of covariance and mean
    
    :param x: n times p matrix where n is the number of observations and p is the number of features
    """

    n = x.shape[0]
    p = x.shape[1]

    gamma0, alpha = consRocke(p, n, initial ) # gamma in eq. (6.40)


    mu0 = np.mean(x, axis = 0)
    V0 = np.cov(x.T)
    if mahalanobis_type == 'svd':
        dist = svd_trunc_mahalanobis_distance(x, mu0, V0, mahalanobis_cutoff)
    else:
        dist = mahalanobis_distance(x, mu0, V0)

    delta = 0.5 #(1.0-p/n)/2.0 # max breakdown
    # Compute M-scale in eq. (6.29)
    sig = MScalRocke(x=dist, gamma=gamma0, q=qs, delta=delta) 
    didi =  dist / sig
    dife = np.sort( np.abs( didi - 1) )
    # If the number of observations with positive weights is less
    # than propmin*p, then enlarge gamma to avoid instability
    gg = np.min( dife[ np.array(range(n)) >= (propmin*p) ] )
    gamma = np.max([gg, gamma0])
    sig0 = MScalRocke(x=dist, gamma=gamma, delta=delta, q=qs)
    count = 0
    difpar = np.inf
    difsig = np.inf

    while ( ( (difsig > tol) | (difpar > tol) ) & (count < maxit) ) & (difsig > 0):
        
        count += 1

        w = WRoTru(tt=dist/sig, gamma=gamma, q=qs)


        mu = np.dot(w,x)/ np.sum(w, axis = 0)
        V = np.dot(w*(x-mu).T, x-mu) / n

        if mahalanobis_type == 'svd':
            dist = svd_trunc_mahalanobis_distance(x, mu0, V0, mahalanobis_cutoff)
        else:
            dist = mahalanobis_distance(x, mu, V)

        sig = MScalRocke(x=dist, gamma=gamma0, q=qs, delta=delta) 
        step = 0
        delgrad = 1

        while (sig > sig0) & (step < maxsteps):
            # If needed, perform linear search
            delgrad = delgrad / 2.0
            step +=  1
            mu = delgrad * mu + (1.0 - delgrad)*mu0
            V = delgrad*V + (1.0-delgrad)*V0
            V = V/np.power(np.linalg.det(V), 1.0/p)
            dist = mahalanobis_distance(x, mu, V)
            sig = MScalRocke(x=dist, gamma=gamma0, q=qs, delta=delta) 


        dif1 = np.array(np.dot((mu-mu0).T, np.dot(np.linalg.inv(V0), mu-mu0) ))
        ok = np.linalg.inv(V0)
        dif2 = np.max(np.abs(np.dot(ok, V)-np.identity(p)))
        difpar = np.max([dif1, dif2])
        difsig = 1.0 - sig/sig0
        mu0 = mu.copy()
        V0 = V.copy()
        sig0 = sig

    ff = np.median(dist)/scipy.stats.chi2.ppf(0.5, df=p)

    V = V*ff
    dist = dist/ff

    return mu, V, dist




def MScalRocke(x, gamma, q, delta = 0.5, tol=1e-5):
    """
    
    :param q; tuning parameter, usually q = 2
    """

    n = len(x)
    y = np.array(sorted(np.abs(x)))
    n1 = int(np.floor(n * (1.0-delta) ) ) # if #(x_i = 0)> n(1-delta) there is no solution
    n2 = int(np.ceil(n * (1.0 - delta) / (1.0 - delta/2.0)))
    qq = y[np.array([n1,n2])]
    u = 1.0 + gamma*(delta-1.0)/2.0
    sigin = np.array([qq[0]/(1.0+gamma), qq[1]/u])
    # print(sigin)
    # sigin = np.array([1e-16, 1])
    if qq[0] >= 1.0:
        tolera = tol
    else:
        tolera = tol*qq[0]

    if np.mean(x == 0) > 1.0 - delta:
        sol = 0
    else:
        sol = brentq(rho_rocke_obj, a = sigin[0], b = sigin[1], args = (x, gamma, q, delta), xtol = tolera)

    return sol





