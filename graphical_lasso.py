""""
Script for graphical lasso glasso, and tlasso.

"""

import numpy as np
from  sklearn.covariance import graphical_lasso
import tqdm
import gradient_descent as gd
import tqdm


def gaussian_likelihood(S, Theta):

    w, v = np.linalg.eig(Theta)

    if np.min(w) < 0:
        v = v[:, w > 1e-5]
        w = w[w > 1e-5]
    l = (
        0.5 * np.sum(np.log(w))
        - 0.5 * np.trace(np.matmul(S, Theta))
        - 0.5 * Theta.shape[0] * np.log(2 * np.pi)
        )

    return l

def EBIC(N, S, Theta, beta = 0.5):
    k =  np.count_nonzero(np.triu(Theta, 1))  + np.count_nonzero(np.diag(Theta))
    EBIC = (
        np.log(N) * k
        - 2 * N * gaussian_likelihood(S, Theta)
        + 4 * beta * k * np.log(Theta.shape[0])
    )

    return EBIC





class graphical_lasso_wrapper():


    def __init__(self, emp_cov, N, alpha, beta) -> None:
        """
        Parameters
        ----------------------

        emp_cov: Empirical covariance matrix
        N: number of samples
        alpha: regularization parameter
        
        """
        self.emp_cov = emp_cov
        self.alpha = alpha
        self.N = N
        self.p = self.emp_cov.shape[0]
        self.beta = beta

    def fit_CV(self):

        """"
        test many alphas, pick the one with best EBIC
        """
        ebic_vals = [] #np.ones(len(alphas)) * np.inf

        prec_list = [] # np.zeros((len(alphas), self.p, self.p))
        cov_list = [] # np.zeros((len(alphas), self.p, self.p))
        alpha_list = []


        alpha = 0.5
        max_itr = 1000
        best_not_found = True
        nr_iterations = 0
        best_ebic = np.inf
        time_since_last_min = 0
        not_all_sparse = True

        while (best_not_found) and (nr_iterations < max_itr) and (not_all_sparse):
            # print(f'{nr_iterations} {alpha}')
            alpha_list.append(alpha)
            
            try:
                out_cov, out_prec = graphical_lasso(emp_cov=self.emp_cov.copy(), alpha=alpha)
            except FloatingPointError:
                ebic_vals.append(np.inf)
                prec_list.append(np.inf)
                cov_list.append(np.inf)
                alpha += 0.01
                nr_iterations += 1
                continue
            except ValueError:
                ebic_vals.append(np.inf)
                prec_list.append(np.inf)
                cov_list.append(np.inf)
                alpha += 0.01
                nr_iterations += 1
                continue


            ebic_t = EBIC(self.N, self.emp_cov, out_prec, beta = self.beta)
            print(f'{nr_iterations} {alpha} {ebic_t} {self.N * gaussian_likelihood(self.emp_cov, out_prec)}')
            ebic_vals.append(ebic_t)
            
            time_since_last_min += 1
            if ebic_t < best_ebic:
                best_ebic = ebic_t
                time_since_last_min = 0
                best_alpha = alpha

            prec_list.append(out_prec)
            cov_list.append(out_cov)

            if time_since_last_min > 50:
                best_not_found = False

            alpha += 0.01
            nr_iterations += 1
            not_all_sparse = (np.count_nonzero(np.triu(out_prec, 1)) != 0)

        best_idx = np.argmin(ebic_vals)
        if np.isscalar(best_idx):
            best_prec = prec_list[best_idx]
            best_alpha = alpha_list[best_idx]
        else:
            best_prec = prec_list[best_idx[0]]
            best_alpha = alpha_list[best_idx[0]]


        return best_prec, prec_list, cov_list, ebic_vals, best_alpha, alpha_list


    def fit_sklearn(self):

        return graphical_lasso(emp_cov=self.emp_cov, alpha=self.alpha)

    def fit_mm(self, init_Theta = None, P = None, reltol = 1e-5, max_itr = 1000):
        """
        Fit graphical lasso using majorization-minimization

        Parameters
        ---------------
        P: lasso matrix
        alpha: lasso penalty
        
        """

        if init_Theta is None:
            init_Theta = np.identity(self.p)

        if P is None:
            P = np.ones(init_Theta.shape)
            np.fill_diagonal(P,0)

        nr_itr = 0
        has_not_converged = True

        tol = np.inf
        Theta0 = init_Theta

        adam = gd.ADAM()
        adam.eta = 0.1
        adam.decreasing_learning_rate = True

        while has_not_converged and nr_itr < max_itr:
            print(tol)

            Theta0_inv = np.linalg.pinv(Theta0)
            gt = -Theta0_inv + self.emp_cov
            print(Theta0)
            Theta = adam.update_no_penalty(Theta0, -gt, P, self.alpha)

            tol = np.linalg.norm(Theta - Theta0, 'fro') / np.linalg.norm(Theta0, 'fro')
            has_not_converged = (np.linalg.norm(Theta - Theta0, 'fro') / np.linalg.norm(Theta0, 'fro') > reltol)
            Theta0 = Theta.copy()

            nr_itr += 1


        return Theta


    def soft_threshold(self, A, B):


        return np.multiply(np.sign(A), np.maximum(A-B, np.zeros(A.shape)))



class tlasso():

    def __init__(self, x, emp_cov, alpha, nu, mu_init, Theta_init, mu_zero) -> None:
        """
        Parameters
        ----------------------
        x: data array
        emp_cov: Empirical covariance matrix
        alpha: regularization parameter
        nu: student degree of freedom
        mu_init: initialization of mean
        theta_init: initialization of biased precision
        mu_zero: bool - is the mean assumed to be zero?
        """

        self.emp_cov = emp_cov
        self.N = x.shape[0]
        self.alpha = alpha
        self.nu = nu
        self.mu_init = mu_init
        self.Theta_init = Theta_init
        self.mu_zero = mu_zero
        self.x = x
        self.p = x.shape[1]

    def fit(self, reltol = 1e-5, max_itr = 1000, verbose = True):

        
        nr_itr = 0
        has_not_converged = True
        Theta0 = self.Theta_init.copy()
        mu0 = self.mu_init
        tau = np.zeros(self.N)
        tol = np.inf
        while has_not_converged and nr_itr < max_itr:
            print(f'{nr_itr} / {max_itr}, {tol}')

            if verbose:
                print(f' {nr_itr} / {max_itr}')

            if self.mu_zero:
                for t in range(self.N):
                    tau[t] = (self.nu + self.N) / (self.nu + np.dot(self.x[t,:], Theta0).dot(self.x[t,:]))


                # S_hat = np.zeros((self.p, self.p))
                # for i in range(self.N):
                #     S_hat += np.outer(self.x[i,:], self.x[i,:]) * tau[i]

                S_hat = np.array([np.outer(self.x[i,:], self.x[i,:]) * tau[i] for i in range(self.N)]).sum(0)
                # print(S_hat[:5,:5])

                _, Theta_t = graphical_lasso(S_hat, self.alpha)

                tol = np.linalg.norm(Theta_t - Theta0, 'fro') / np.linalg.norm(Theta0, 'fro')
                has_not_converged = (np.linalg.norm(Theta_t - Theta0, 'fro') / np.linalg.norm(Theta0, 'fro') > reltol)
                Theta0 = Theta_t.copy()
            else:
                for t in range(self.N):
                    tau[t] = (self.nu + self.N) / (self.nu + np.dot(self.x[t,:] - mu0, Theta0).dot(self.x[t,:] - mu0))

                sum_tau = np.sum(tau)

                mu_hat = np.array([tau[i]* self.x[i,:] for i in range(self.p)]).sum(0)/sum_tau
                S_hat = np.array([np.outer(self.x[i,:] - mu_hat, self.x[i,:]-mu_hat) * tau[i] for i in range(self.N)]).sum(0)

                _ , Theta_t= graphical_lasso(S_hat, self.alpha)
                tol = np.linalg.norm(Theta_t - Theta0, 'fro') / np.linalg.norm(Theta0, 'fro')
                has_not_converged = (tol > reltol)

                Theta0 = Theta_t.copy()
                mu0 = mu_hat.copy()

            nr_itr += 1


        
        return ((self.nu-2.0)/self.nu) * Theta_t


class my_glasso():

    def __init__(self, x, alpha, mu_init, Theta_init, mu_zero) -> None:
        """
        Parameters
        ----------------------
        x: data array
        emp_cov: Empirical covariance matrix
        alpha: regularization parameter
        nu: student degree of freedom
        mu_init: initialization of mean
        theta_init: initialization of biased precision
        mu_zero: bool - is the mean assumed to be zero?
        """

        self.N = x.shape[0]
        self.alpha = alpha
        self.mu_init = mu_init
        self.Theta_init = Theta_init
        self.mu_zero = mu_zero
        self.x = x
        self.p = x.shape[1]


    def fit(self, step, max_itr, verbose = True):
        nr_itr = 0
        has_not_converged = True
        Theta0 = self.Theta_init.copy()
        mu0 = self.mu_init
        tau = np.zeros(self.N)
        tol = np.inf

        P = np.ones(Theta0.shape)
        np.fill_diagonal(P,0)

        if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total=max_itr)

        while has_not_converged and nr_itr < max_itr:
            
            grad = -np.linalg.pinv(Theta0) + np.einsum('ij,ik->jk', self.x, self.x)/self.N
            Theta = self._prox(Theta0 - step*grad, step*self.alpha*P)
            Theta0 = Theta.copy()

            nr_itr += 1


            if verbose:
                    pbar.update()

        if verbose:
            pbar.close()

        return Theta

    
    
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