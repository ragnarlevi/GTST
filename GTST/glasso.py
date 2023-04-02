# Wrapper for Sklearn glasso + ebic and nonparanormal transformation


import numpy as np
from scipy.stats import rankdata
from scipy.stats import norm, multivariate_normal
from sklearn.covariance import GraphicalLasso
import networkx as nx



class glasso_wrapper():

    def __init__(self, alpha, beta, nonparanormal:bool = False, scale:bool = False) -> None:
        """
        Parameters
        --------------------------
        alpha: list, float,
            Regularization parameters in a list or a single float
        beta: float:
            EBIC hyperparameter
        nonparanormal: bool
            Should data be nonparanormally transformed
        scale:bool,
            Should data be scaled

        """


        self.alpha = alpha
        self.beta = beta
        self.nonparanormal = nonparanormal
        self.scale = scale

        self.emp_mean =None
        self.emp_var = None
        self.n = None
        self.is_fitted = False

    def nonparanormal_transform(self):
        """
        Perform nonparanormal transform
        """

        if self.emp_mean is None:
            raise ValueError("empirical mean not calculated")
        if self.emp_var is None:
            raise ValueError("empirical variance not calculated")
        if self.n is None:
            raise ValueError("X not yet passed")

    
        X_ranked = rankdata(self.X,axis=0)/self.n
        threshold = 1/(4*(self.n**0.25)*np.sqrt(np.pi*np.log(self.n)))
        return self.emp_mean + np.sqrt(self.emp_var)*norm.ppf(np.minimum(np.maximum(X_ranked,threshold),1-threshold))


    def log_lik(self,mean,cov, X = None):
        """
        mean: numpy 1d array
            mean vector
        cov: numpy 2d array
            Covariance matrix
        X: numpy 2d array
            Data matrix, where each row is an observation
        """

        if X is None:
            X = self.X

        return np.sum(multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True))

    def ebic(self,mean,cov,prec, X = None):
        """
        mean: numpy 1d array
            mean vector
        cov: numpy 2d array
            Covariance matrix
        prec: numpy 2d array
            Precision matrix
        X: numpy 2d array
            Data matrix, where each row is an observation
        """
        
        if X is None:
            X = self.X

        n = X.shape[0]

        prec[np.abs(prec)<1e-3] = 0
        G = nx.from_numpy_array(prec)
        n_edges = G.number_of_edges()

        log_lik_val = self.log_lik(mean,cov, X = self.X)
        return -2*log_lik_val + np.log(n)*n_edges + 4*n_edges*self.beta*np.log(prec.shape[0])



    def fit(self, X, **kwargs):

        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.emp_mean = np.mean(self.X,axis = 0)
        self.emp_var = np.var(X,axis = 0)
        self.best_index = None
        self.ebic_vals = []

        if self.nonparanormal:
            self.X = self.nonparanormal_transform()

        if isinstance(self.alpha, int) or isinstance(self.alpha, float):
            self.alpha = [float(self.alpha)]
            
        # Perform estimation
        best_ebic = np.inf
        for idx, alpha in enumerate(self.alpha):
            glasso = GraphicalLasso(alpha = alpha, **kwargs).fit(self.X)
            ebic_val = self.ebic(mean = self.emp_mean, cov = glasso.covariance_, prec = glasso.precision_)
            self.ebic_vals.append(ebic_val)

            if ebic_val < best_ebic:
                self.best_index = idx
                self.covariance_ = glasso.covariance_.copy()
                self.precision_ = glasso.precision_

                best_ebic = ebic_val

        return self






















