# -*- coding: utf-8 -*-

"""
This script is from https://github.com/skojaku/scola, https://royalsocietypublishing.org/doi/full/10.1098/rspa.2019.0578?af=R

I have taken the code and personalized it for my needs.
"""


# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg, sparse, stats
from scipy.special import gamma, digamma
import os
import tqdm
import sys
from functools import partial
import gradient_descent as gd

from mmdutils import _fast_mat_inv_lapack
from mmdutils import _comp_EBIC
from mmdutils import _comp_loglikelihood
from mmdutils import _remedy_degeneracy, _truncated_inverse, mahalanobis_distance



def _comp_loglikelihood_mvt(W, x, C_null, nu):
    """
        Compute the log likelihood for a network using multivariate student t. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        x : 2D numpy.ndarray, shape (L, N)
            data 
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
        nu: float
            mvt parameter
        input_matrix_type: string
	    Type of matrix to be given (covariance or precision)
    
        Returns
        -------
        l : float
            Log likelihood for the generated network. 
        """

    Cov = W + C_null
    p = float(x.shape[1])
    L = float(x.shape[1])
    iCov, w = _truncated_inverse(Cov)

    d = mahalanobis_distance(x,0,iCov)
    # iCov = np.real(np.matmul(np.matmul(v, np.diag(1 / w)), v.T))
    l = L * np.log(gamma((nu + p)/2.0)) - L * np.log(gamma(nu / 2.0)) - L * (p/2.0) * np.log(nu) - L * (p/2.0) * np.log(p) - L * 0.5 * np.sum(np.log(w)) + L * np.log(nu) - ((nu + p) / 2.0) * np.sum(np.log(nu + nu * d))
    


    return np.real(l)


def _comp_EBIC_mvt(W, nu, x, C_null, L, beta, Knull):
    """
        Compute the extended Bayesian Information Criterion (BIC) for a network fot mvt. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        nu: 
        x : 2D numpy.ndarray, shape (L, N)
            Data
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
        L : int
            Number of samples.
        beta : float
            Parameter for the extended BIC. 
        K_null: int
            Number of parameters of the null correlation matrix.
        Returns
        -------
        EBIC : float
            The extended BIC value for the generated network.
        """
    k = Knull + np.count_nonzero(np.triu(W, 1))  + np.count_nonzero(np.diag(W))
    EBIC = (
        np.log(L) * k
        - 2 * _comp_loglikelihood_mvt(W, x, C_null)
        + 4 * beta * k * np.log(W.shape[0])
    )
    return EBIC

class Scola:

    def __init__(self, likelihood = 'normal'):
        self.approximation = False
        self.input_matrix_type = "cov"
        self.likelihood = likelihood

    def detect(self, C_samp, C_null, lam, Winit = None, nu_init = 3, x = None):
        """
	    Minorisation-maximisation algorithm. 
	        
	    Parameters
	    ----------
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix.
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of the generated network.
	    """

        N = C_samp.shape[0]
        Lambda = 1.0 / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)

        if self.approximation:
            W = self._prox(C_samp - C_null, lam * Lambda)
            return W

        if Winit is not None:
            W = Winit 
        else:
            W = self._prox(C_samp - C_null, lam * Lambda)

        score_prev = -1e300
        count_small_itnum = 0 
        if self.likelihood == 'mvt':
            score = self._comp_penalized_loglikelihood_mvt(W, C_samp, C_null, lam * Lambda, nu_init)
        else:
            score = self._comp_penalized_loglikelihood(W, C_samp, C_null, lam * Lambda)

        while True:
            if self.likelihood == 'mvt':
                _W, itnum = self._maximisation_step_mvt(x, C_samp, C_null, W, lam, nu_init)
                score = self._comp_penalized_loglikelihood_mvt(_W, C_samp, C_null, lam * Lambda, nu_init)
            else:
                _W, itnum = self._maximisation_step(C_samp, C_null, W, lam)
                score = self._comp_penalized_loglikelihood(_W, C_samp, C_null, lam * Lambda)
            

            if score <= score_prev:
                break
            W = _W

            # The changes in W are gentle, stop iteration.
            if itnum <=2:
               count_small_itnum+=1 
               if count_small_itnum>5:
                    break
            else:
               count_small_itnum=0 

            score_prev = score

        return W

    def comp_upper_lam(self, C_samp, C_null):
        """
        Compute the upper bound of the Lasso penalty.
    
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
    
        Returns
        -------
        lam_upper : float
            Upper bound of the Lasso penalty. 
        """

        abC_samp = np.abs(C_samp - C_null)
        # iCov = _fast_mat_inv_lapack(C_null)
        iCov, _ = _truncated_inverse(C_null)
        D = iCov - np.matmul(np.matmul(iCov, C_samp), iCov)
        lam_upper = np.max(np.multiply(np.abs(D), np.power(abC_samp, 2)))

        return lam_upper

    def _comp_penalized_loglikelihood(self, W, C_samp, C_null, Lambda):
        """
	    Compute the penalized log likelihood for a network. 
	    
	    Parameters
	    ----------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of a network.
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix used for constructing the network.
	    Lambda : 2D numpy.ndarray, shape (N, N)
	        Lambda[i,j] is the Lasso penalty for W[i,j]. 
	
	    Returns
	    -------
	    l : float
	        Penalized log likelihood for the generated network. 
	    """
        return (
            _comp_loglikelihood(W, C_samp, C_null, "cov")
            - np.sum(np.multiply(Lambda, np.abs(W))) / 4
        )

    def _comp_penalized_loglikelihood_mvt(self, W, C_samp, C_null, Lambda, nu):
        """
	    Compute the penalized log likelihood for a network using mvt. 
	    
	    Parameters
	    ----------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of a network.
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix used for constructing the network.
	    Lambda : 2D numpy.ndarray, shape (N, N)
	        Lambda[i,j] is the Lasso penalty for W[i,j]. 
	
	    Returns
	    -------
	    l : float
	        Penalized log likelihood for the generated network. 
	    """
        return (
            _comp_loglikelihood_mvt(W, C_samp, C_null, nu)
            - np.sum(np.multiply(Lambda, np.abs(W))) / 4
        )

    def _prox(self, x, lam):
        """
	    Soft thresholding operator.
	    
	    Parameters
	    ----------
	    x : float
	        Variable.
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    y : float
	        Thresholded value of x. 
	    """

        return np.multiply(np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape)))

    def _maximisation_step(self, C_samp, C_null, W_base, lam):
        """
	    Maximisation step of the MM algorithm. 
	    (A subroutine for detect) 
	    ----------
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix.
	    W_base : 2D numpy.ndarray, shape (N, N)
	        W at which the minorisation is performed.  
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of the generated network.
	"""
        N = C_samp.shape[0]
        t = 0
        #eps = 1e-8
        #b1 = 0.9
        #b2 = 0.999
        maxscore = -1e300
        t_best = 0
        #eta = 0.001
        maxLocalSearch = 30
        maxIteration = 300 

        #maxIteration = 1e7
        #maxLocalSearch = 300
        W = W_base
        Lambda = lam / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)
        # inv_C_base = _fast_mat_inv_lapack(C_null + W_base)
        inv_C_base, _ = _truncated_inverse(C_null + W_base)
        _diff_min = 1e300
        score0 = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)


        adam = gd.ADAM()
        adam.eta = 0.01
        adam.decreasing_learning_rate = True
        prev_score = score0
        while (
            (t < maxIteration) & ((t - t_best) <= maxLocalSearch) & (_diff_min > 1e-5)
        ):
            t = t + 1
            # inv_C = _fast_mat_inv_lapack(C_null + W)
            inv_C, _ = _truncated_inverse(C_null + W)
            # gt = inv_C_base - np.matmul(np.matmul(inv_C, C_samp), inv_C)
            gt = inv_C - np.matmul(np.matmul(inv_C, C_samp), inv_C)

            W_prev = W
            np.fill_diagonal(gt, 0.0)
            W = adam.update( W, -gt, Lambda)    
            np.fill_diagonal(W, 0.0)

            _diff = np.sqrt( np.mean(np.power(W - W_prev, 2)) )
       

            if _diff < (_diff_min * 0.95):
                _diff_min = _diff
                t_best = t

            if _diff < 5e-5:
                break

            # If the score isn't improved in the first 50 iterations, then break
            if t % 10 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)
                if (prev_score > score):
                    break
                prev_score = score
            if t % 50 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)
                if (score0 > score):
                    break

        return W, t

    def _maximisation_step_mvt(self, x, C_samp, C_null, W_base, lam, nu_init):
        """
	    Maximisation step of the MM algorithm. 
	    (A subroutine for detect) 
	    ----------
        x : 2D numpy.ndarray, shape (L, N)
	        data
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix.
	    W_base : 2D numpy.ndarray, shape (N, N)
	        W at which the minorisation is performed.  
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of the generated network.
	"""
        N = C_samp.shape[0]
        t = 0
        #eps = 1e-8
        #b1 = 0.9
        #b2 = 0.999
        maxscore = -1e300
        t_best = 0
        #eta = 0.001
        maxLocalSearch = 30
        maxIteration = 300 

        #maxIteration = 1e7
        #maxLocalSearch = 300
        W = W_base
        Lambda = lam / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)
        # inv_C_base = _fast_mat_inv_lapack(C_null + W_base)
        inv_C_base, _ = _truncated_inverse(C_null + W_base)
        _diff_min = 1e300
        nu = nu_init
        score0 = self._comp_penalized_loglikelihood_mvt(W, x, C_null, Lambda, nu)


        adam = gd.ADAM()
        adam.eta = 0.01
        adam.decreasing_learning_rate = True

        adam_nu = gd.ADAM()
        adam_nu.eta = 0.01
        adam_nu.decreasing_learning_rate = True
        prev_score = score0
        while (
            (t < maxIteration) & ((t - t_best) <= maxLocalSearch) & (_diff_min > 1e-5)
        ):
            t = t + 1
            # inv_C = _fast_mat_inv_lapack(C_null + W)
            inv_C, _ = _truncated_inverse(C_null + W)
            d = mahalanobis_distance(x,0, inv_C)

            cc = np.zeros(inv_C.shape)
            for i in range(x.shape[0]):
                cc += d[i]*np.outer(x[i,:], x[i,:])
            
            gt = 0.5 * inv_C - (nu + x.shape[1])/2 * cc
            gt_nu = digamma((nu + x.shape[1]) /2 ) - digamma(nu /2) - (nu + x.shape[1])/2 * np.sum(1/ (nu + d)) + 1/nu

            W_prev = W
            np.fill_diagonal(gt, 0.0)
            W = adam.update( W, -gt, Lambda)    
            nu = adam_nu.update_no_penalty( nu, -gt_nu)    
            np.fill_diagonal(W, 0.0)

            _diff = np.sqrt( np.mean(np.power(W - W_prev, 2)) )
       

            if _diff < (_diff_min * 0.95):
                _diff_min = _diff
                t_best = t

            if _diff < 5e-5:
                break

            # If the score isn't improved in the first 50 iterations, then break
            if t % 10 == 0:
                score = self._comp_penalized_loglikelihood_mvt(W, C_samp, C_null, Lambda, nu)
                if (prev_score > score):
                    break
                prev_score = score
            if t % 50 == 0:
                score = self._comp_penalized_loglikelihood_mvt(W, C_samp, C_null, Lambda, nu)
                if (score0 > score):
                    break

        return W, t

def find_best_W( C_samp, C_null, L, beta = 0.5, disp = True):
    """
    Generate a network from a correlation matrix
    using the Scola algorithm.
    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. *N* is the number of nodes.
    C_null : 2D numpy.ndarray, shape (N, N)
        Null correlation matrix. *N* is the number of nodes.
    L : int
        Number of samples.
    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.
    beta : float, default 0.5 
        Hyperparameter for the extended BIC. When beta = 0, the EBIC is equivalent to the BIC. The higher value yields a sparser network. Range [0,1].
        A value of 0.5 typically provides good results.
    Returns
    -------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of the generated network.
    EBIC : float
        The extended BIC value for the generated network. 
    all_networks : list of dict 
        Results of all generated networks. Each dict object in the list consists of 'W', 'C_null', 'null_model', 'EBIC_min', 'construct_from' and 'W_list'. 'W_list' is a list of dict objects, in which each dict consists of a network (i.e., 'W') and its EBIC value (i.e., 'EBIC') found by the golden section search algorithm.

    """

    if type(C_samp) is not np.ndarray:
        raise TypeError("C_samp must be a numpy.ndarray")

    if (type(L) is not int) and (type(L) is not float):
        raise TypeError("L must be an integer")

    if type(disp) is not bool:
        raise TypeError("disp must be a bool")

    if not ((type(beta) is float) or (type(beta) is int)):
        raise TypeError("beta must be float")

    if (beta < 0) or (1 < beta):
        raise TypeError("beta must be in range [0,1]")

    # Improve the degeneracy of C_samp for the computational stability if it has too small eigenvalues
    C_samp = _remedy_degeneracy(C_samp, rho = 1e-6, scale = True)


    # pbar is used for computing and displaying the progress of computation.
    pbar = tqdm.tqdm(
        disable=(disp is False), total=(13 * 1 * 1)
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))
    )

    estimator = Scola()

    W, C_null, EBIC_min, all_networks = golden_section_search(
        C_samp, L, C_null, 0, estimator, beta, pbar, disp
    )

    #W = np.matmul(np.matmul(R.T, W), R)

    res = [
        {
            "W": W,
            "EBIC_min": EBIC_min,
            "W_list": all_networks
        }
    ]


    pbar.close()

    idx = np.argmin(np.array([r["EBIC_min"] for r in res]))
    return res[idx]["W"], res[idx]["EBIC_min"], res



def golden_section_search(C_samp, L, C_null, K_null, estimator, beta, pbar, disp):

    lam_upper = 0.1 # estimator.comp_upper_lam(C_samp, C_null)
    lam_lower = 0

    W_best, C_null, EBIC, lam_best, all_networks = _golden_section_search(C_samp, L, C_null, K_null, estimator, beta, lam_lower, lam_upper, pbar, False, disp)

    return W_best, C_null, EBIC, all_networks


def _golden_section_search(C_samp, L, C_null, K_null, estimator, beta, lam_lower, lam_upper, pbar, W_interpolate, disp):
    """
    Find the Lasso penalty that minimises the extended BIC using
    the golden-section search method.
    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    L : int
        Number of samples.
    C_null : 2D numpy.ndarray, shape (N, N)
       Null covariance
    beta : float
        Hyperparameter for the extended BIC.
    pbar : tqdm instance
        This instance is used for computing and displaying 
        the progress of computation.
    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.
    Returns
    -------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of the generated network.
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the 
        network.
    EBIC : float
        The extended BIC value for the generated network.
    all_networks : list of dict 
        Results of all generated networks. Each dict object in the list consists of 'W' and 'EBIC'.
    """

    invphi = (np.sqrt(5) - 1.0) / 2.0
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0
    h = lam_upper - lam_lower
    lam_1 = lam_lower + invphi2 * h
    lam_2 = lam_lower + invphi * h
    
    n = int(np.ceil(np.log(0.01) / np.log(invphi)))
    N = C_samp.shape[0]
    W_best = None
    lam_best = 0
    EBIC_min = 0

    ns = pbar.n
    nf = ns + n
    
    all_networks = []
    for k in range(n):
        print(f'{lam_lower} {lam_upper} {lam_1} {lam_2}')
        if k == 0:

            #W_l = C_samp - C_null
            W_l = estimator.detect(C_samp, C_null, lam_lower)
            pbar.update()
            W_u = estimator.detect(C_samp, C_null, lam_upper)
            pbar.update()
            W_1 = estimator.detect(C_samp, C_null, lam_1)
            pbar.update()
            W_2 = estimator.detect(C_samp, C_null, lam_2)
            pbar.update()

            EBIC_l = _comp_EBIC(
                W_l, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_u = _comp_EBIC(
                W_u, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_1 = _comp_EBIC(
                W_1, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_2 = _comp_EBIC(
                W_2, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )

            print([EBIC_l, EBIC_u, EBIC_1, EBIC_2])
            mid = np.argmin([EBIC_l, EBIC_u, EBIC_1, EBIC_2])
            W_best = [W_l, W_u, W_1, W_2][mid]
            lam_best = [lam_lower, lam_upper, lam_1, lam_2][mid]
            EBIC_min = [EBIC_l, EBIC_u, EBIC_1, EBIC_2][mid]

            all_networks += [
                {
                    "W": W_l,
                    "EBIC": EBIC_l,
                },
                {
                    "W": W_u,
                    "EBIC": EBIC_u,
                },
                {
                    "W": W_1,
                    "EBIC": EBIC_1,
                },
                {
                    "W": W_2,
                    "EBIC": EBIC_2,
                }
            ]
            continue

        if (EBIC_1 < EBIC_2) | ((EBIC_1 == EBIC_2) & (np.random.rand() > 0.5)):
            lam_upper = lam_2
            lam_2 = lam_1
            EBIC_u = EBIC_2
            EBIC_2 = EBIC_1
            h = invphi * h
            lam_1 = lam_lower + invphi2 * h

            if W_interpolate == True:
                W_1 = estimator.detect(C_samp, C_null, lam_1, (W_l + W_1)/2)
            else:
                W_1 = estimator.detect(C_samp, C_null, lam_1)


            pbar.update()
            EBIC_1 = _comp_EBIC(
                W_1, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            print(EBIC_1)

            if EBIC_1 < EBIC_min:
                EBIC_min = EBIC_1
                W_best = W_1
                lam_best = lam_1

            all_networks += [
                {
                    "W": W_1,
                    "EBIC": EBIC_1
                }
            ]

        else:
            lam_lower = lam_1
            lam_1 = lam_2
            EBIC_l = EBIC_1
            EBIC_1 = EBIC_2
            h = invphi * h
            lam_2 = lam_lower + invphi * h

            if W_interpolate == True:
                W_2 = estimator.detect(C_samp, C_null, lam_2, (W_2 + W_u)/2)
            else:
                W_2 = estimator.detect(C_samp, C_null, lam_2)

            pbar.update()
            EBIC_2 = _comp_EBIC(
                W_2, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            print(EBIC_2)
            if EBIC_2 < EBIC_min:
                EBIC_min = EBIC_2
                W_best = W_2
                lam_best = lam_2

            all_networks += [
                {
                    "W": W_2,
                    "EBIC": EBIC_2
                }
            ]

        pbar.refresh()

    EBIC = EBIC_min
    return W_best, C_null, EBIC, lam_best, all_networks