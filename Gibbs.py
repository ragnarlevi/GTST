
from matplotlib.pyplot import axis
import lcurve
import numpy as np

def KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, calc_cond = False, regularize = False, reg_param = 0.1):
    """
    Calculate Kalman Filter
    y = Fx + A
    x = Gx + B 
    """
    state = np.zeros((y.shape[0]+1, init_x.shape[0]))
    state_cov = np.zeros((y.shape[0]+1, init_x.shape[0], init_x.shape[0]))

    state[0] = init_x
    state_cov[0] = init_c

    state_one_step = np.zeros((y.shape[0], init_x.shape[0]))
    state_cov_one_step = np.zeros((y.shape[0], init_x.shape[0], init_x.shape[0]))

    R_cond = np.zeros((y.shape[0], 2))

    y_est = np.zeros(y.shape)

    # create a numpy list of co-variates (if a np list is not an input)
    # This allows dynamic co-variates
    if A.ndim <= 1:
        A_new = np.empty((y.shape[0], A.shape[0]))
        A_new[:] = A
    else:
        A_new = A

    # To deal with missing observations 
    tmp_A = A_new.copy()
    for i in range(y.shape[0]):
        tmp_A[i, np.isnan(y[i,:])] = 0 


    for i in range(y.shape[0]):

        state_one_step[i] = np.dot(G, state[i]) + B
        state_cov_one_step[i] = W + np.dot(G, state_cov[i]).dot(G.T)

        # To deal with missing observations
        tmp_F = F.copy()
        tmp_F[np.isnan(y[i,:]), :] = 0

        tmp_y = y[i].copy()
        tmp_y[np.isnan(y[i,:])] = 0

        tmp_V = V.copy()
        tmp_V[np.isnan(y[i,:]),:] = 0 
        tmp_V[:,np.isnan(y[i,:])] = 0
        tmp_V[np.isnan(y[i,:]),np.isnan(y[i,:])] = 1

        R = np.dot(tmp_F, state_cov_one_step[i]).dot(tmp_F.T) + tmp_V
        R[np.isnan(y[i,:]),:] = 0 
        R[:,np.isnan(y[i,:])] = 0
        R[np.isnan(y[i,:]),np.isnan(y[i,:])] = 1


        if calc_cond:
            R_cond[i, 0] = np.linalg.cond(R)
            if R_cond[i,0] > 1000:
                R = R + reg_param*np.identity(tmp_V.shape[0])
                R_cond[i, 1] = np.linalg.cond(R)

        if regularize:
            R = R + reg_param*np.identity(tmp_V.shape[0])

        if R.ndim == 1:
            Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(1/R)
        else:
            Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(np.linalg.inv(R))

        state[i+1] = state_one_step[i] + np.dot(Kalman_gain, tmp_y - tmp_A[i] - np.dot(tmp_F, state_one_step[i]))
        state_cov[i+1] = state_cov_one_step[i] - np.dot(Kalman_gain, tmp_F).dot(state_cov_one_step[i])
        y_est[i] = np.dot(F, state[i+1]) + tmp_A[i]

    print(f'{np.max(R_cond[:, 0])} vs {np.max(R_cond[:, 1])}')
    return state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond


def KalmanSmooth(state, state_one_step, state_cov, state_cov_one_step, G, B, W, regularize = False, reg_param = 0.1):
    """
    Calculate Kalman Smoother
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """

    smooth_state = np.zeros((state.shape[0], state.shape[1]))
    smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    smooth_state[-1] = state[-1]
    smooth_state_cov[-1] = state_cov[-1]

    for i in reversed(range(1, state.shape[0])): 

        R = np.dot(G, state_cov[i]).dot(G.T) + W

        if regularize:
            R = R + reg_param*np.identity(W.shape[0])

        if R.ndim == 1:
            J = np.dot(state_cov[i], G.T).dot(1/R)
        else:
            J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))

        smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state[i] - B - state_one_step[i-1])
        smooth_state_cov[i-1] = state_cov[i-1] - np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)


    return smooth_state, smooth_state_cov



def FFBS(y, G, B, W, F, A, V, init_x, init_c, calc_cond = False, regularize_F = False, regularize_S = False, reg_params = {'reg_f':0.1, 'reg_s':0.1}):
    """
    Forward Filtering Backward Sampling for a Gibbs sampler
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 

    regularize_F: regularize filter?
    regularize_S: regularize smoother?
    """


    state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond = KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, calc_cond, regularize_F, reg_params['reg_f'])


    smooth_state = np.zeros((state.shape[0], state.shape[1]))
    smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    smooth_state_draws = np.zeros((state.shape[0], state.shape[1]))

    smooth_state[-1] = state[-1]
    smooth_state_cov[-1] = state_cov[-1]
    smooth_state_draws[-1] = np.random.multivariate_normal(smooth_state[-1], smooth_state_cov[-1])

    for i in reversed(range(1, state.shape[0])): 

        R = np.dot(G, state_cov[i]).dot(G.T) + W

        if regularize_S:
            R = R + reg_params['reg_s']*np.identity(W.shape[0])

        if R.ndim == 1:
            J = np.dot(state_cov[i], G.T).dot(1/R)
        else:
            J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))

        smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state_draws[i] - B - state_one_step[i-1])
        smooth_state_cov[i-1] = state_cov[i-1] - np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)

        smooth_state_draws[i-1] = np.random.multivariate_normal(smooth_state[i-1], smooth_state_cov[i-1])

    return smooth_state_draws, R_cond


def lc_sector(N, y, group_membership, calc_cond = False, regularize_S = False, regularize_F = False, reg_params = {'reg_f':0.1, 'reg_s':0.1} ):
    """
    Calculate Kalman Smoother
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """

    n_stock = y.shape[1]
    T = y.shape[0]
    n_groups = len(np.unique(group_membership))

    print(f'T {T}, n_stock {n_stock}, n_groups {n_groups}')

    beta_mean = 1.0
    beta_var = 1.0 

    alpha_mean = np.array([70.0] * n_stock)
    alpha_var = np.array([10.0] * n_stock) 

    eta_mean = np.array([0.0] * (1 + n_groups))
    eta_var = np.array([2.0] * (1 + n_groups))

    v_alpha = 10.0 
    v_beta = 0.1

    w_alpha = 10.0
    w_beta = 0.1

    beta_init = np.array([0.1] * n_stock + [0.1] * n_stock )
    alpha_init = np.nanmean(y, axis = 0)
    eta_init = 0.0 * np.ones(1 + n_groups) 
    w_init = 0.001 * np.ones(1 + n_groups) 
    v_init = 0.001*np.ones(n_stock) 

    init_x = np.array([0.0] * (1+n_groups))
    init_c = np.identity((1 + n_groups)) * 10


    B_vec = np.zeros((N+1,1 + n_groups))
    B_vec[0] = eta_init

    R_conds = np.zeros((N,y.shape[0],2))

    w = np.zeros((N+1,1 + n_groups)) * 0.3
    w[0] = w_init

    # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
    beta_vec = np.zeros((N+1,2*n_stock))
    beta_vec[0] = beta_init

    A_vec = np.zeros((N+1,n_stock))
    A_vec[0] = alpha_init

    v = np.zeros((N+1, n_stock))
    v[0] = v_init

    states_store = np.zeros((N, T, 1 + n_groups))

    G = np.identity(1 + n_groups)
    # F = np.zeros((y.shape[1], (nr_groups + 1)))

    # index to help count
    index = np.array(range(n_stock))

    # constraints
    beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])
    # beta_vec[0, n_stock:] = beta_vec[0, n_stock:]/np.sum(beta_vec[0, n_stock:])
    for j in range(n_groups):
        beta_vec[0, n_stock + index[group_membership == j]] = beta_vec[0, n_stock + index[group_membership == j]]/np.sum(beta_vec[0, n_stock + index[group_membership == j]])



    for i in range(1,N+1):
        print( f'{i} of {N} ')

        F = np.zeros((n_stock,  1+ n_groups))
        F[:,0] = beta_vec[i-1, :n_stock]
        for j in range(n_groups):
            F[index[group_membership == j],1 + j] = beta_vec[i-1,n_stock + index[group_membership == j]]

        smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond, regularize_S = regularize_S, regularize_F = regularize_F, reg_params = reg_params)
        R_conds[i-1] = R_cond

        # Constraint
        smooth_state_new =   smooth_state[1:]  - np.mean(smooth_state[1:] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1] = smooth_state_new

        # sample beta_i
        for j in range(y.shape[1]):
            var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var))
            tmp1 = y[:,j]*smooth_state_new[:,0]
            tmp2 = A_vec[i-1, j]*smooth_state_new[:,0] 
            tmp3 = beta_vec[i-1, n_stock + j]*smooth_state_new[:,0]*smooth_state_new[:,1 + group_membership[j]] 

            avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2 -tmp3))/v[i-1,j]) * var
            beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # constraints
        beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])

        # sample beta_g_i
        for j in range(y.shape[1]):
            var = 1.0 / ((np.sum(smooth_state_new[:,1 + group_membership[j]]  ** 2) / v[i-1,j]) + (1.0 / beta_var))
            tmp1 = y[:,j]**smooth_state_new[:,1 + group_membership[j]] 
            tmp2 = A_vec[i-1, j]**smooth_state_new[:,1 + group_membership[j]] 
            tmp3 = beta_vec[i-1, j]*smooth_state_new[:,0]*smooth_state_new[:,1 + group_membership[j]] 

            avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2 -tmp3))/v[i-1,j]) * var
            beta_vec[i,n_stock + j] = np.random.normal(avg, np.sqrt(var))

        # constraints
        # beta_vec[i, n_stock:] = beta_vec[i, n_stock:]/np.sum(beta_vec[i, n_stock:])
        for j in range(n_groups):
            # print(index[group_membership == j])
            beta_vec[i, n_stock + index[group_membership == j]] = beta_vec[i, n_stock + index[group_membership == j]]/np.sum(beta_vec[i, n_stock + index[group_membership == j]])

        # sample alpha
        for j in range(n_stock):
            var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
            avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,n_stock + j]*smooth_state_new[:,1 + group_membership[j]]  )/v[i-1,j]
            avg += alpha_mean[j]/alpha_var[j]
            avg *= var
            A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # sample variance of observation
        for j in range(n_stock):
            alpha = (y.shape[0]/2.0) + v_alpha
            # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
            beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,n_stock + j]*smooth_state_new[:,1 + group_membership[j]]  ) ** 2) + v_beta
            # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
            v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


        # state equations
        for j in range(1 + n_groups):
            var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[j]))
            avg = np.nansum(smooth_state_new[1:, j] - smooth_state_new[:(smooth_state_new.shape[0]-1), j]) / w[i-1,0]
            avg += (eta_mean[j]/eta_var[j])
            avg *= var
            B_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        for j in range(1 + n_groups):
            alpha_w_tmp = y.shape[0]/2.0 + w_alpha
            beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,j] - smooth_state_new[:(smooth_state_new.shape[0]-1),j] - B_vec[i, j]) ** 2) + w_beta
            #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
            # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
            w[i,j] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


    return w, v, beta_vec, A_vec, B_vec, states_store, R_conds




def em_cov_missing_data(y, nr_iterations, Sigma, mu):
    """
    Estimate the covariance of y, where y has missing data

    :param y: data size = (N x k) where N is number of data points and k is number of variables 
    :param y: size (k,) mean of each variable
    """
    
    N = y.shape[0]
    k = y.shape[1]

    index = np.array(range(y.shape[1]))

    loglik = np.zeros(nr_iterations+1)
    loglik[0] = -np.inf

    for i in range(nr_iterations):
        print(i)
        sum_matrix = np.zeros((y.shape[1], y.shape[1]))
        y_tmp = np.zeros((y.shape[0], y.shape[1]))
        for t in range(N):
            index_m = index[np.isnan(y[t,:])]
            index_o = index[~np.isnan(y[t,:])]

            inv_sigma = np.linalg.inv(Sigma[np.ix_(index_o, index_o)])
            
            expected_m = mu[index_m] + np.matmul(Sigma[np.ix_(index_m, index_o)], inv_sigma).dot(y[t,index_o] - mu[index_o])

            y_tmp[t,index_o] = y[t, index_o]
            # print(len(index_m))
            y_tmp[t,index_m] = expected_m

            sum_matrix[np.ix_(index_o, index_o)] = sum_matrix[np.ix_(index_o, index_o)] + np.outer(y[t,index_o] - mu[index_o], y[t,index_o] - mu[index_o])
            sum_matrix[np.ix_(index_o, index_m)] = sum_matrix[np.ix_(index_o, index_m)] + np.outer(y[t,index_o] - mu[index_o], expected_m - mu[index_m])
            sum_matrix[np.ix_(index_m, index_o)] = sum_matrix[np.ix_(index_m, index_o)] + np.outer(expected_m - mu[index_m], y[t,index_o] - mu[index_o])
            sum_matrix[np.ix_(index_m, index_m)] = sum_matrix[np.ix_(index_m, index_m)] + np.outer(expected_m - mu[index_m], expected_m - mu[index_m]) + Sigma[np.ix_(index_m, index_m)] - np.matmul(Sigma[np.ix_(index_m, index_o)], inv_sigma).dot(Sigma[np.ix_(index_o, index_m)])

        # loglik[i+1] = -np.trace(np.dot(np.linalg.inv(Sigma), sum_matrix)) - np.log(np.linalg.det(Sigma)) * (N / 2)
        Sigma = sum_matrix.copy() / N
        # print(Sigma[0, 0])
        mu = np.nanmean(y_tmp, axis=0)


    return Sigma, mu



            
from scipy.stats import invgamma

def lc_sector_test(N, y,x, calc_cond = False):
    """
    Calculate Kalman Smoother
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """

    n_stock = y.shape[1]
    T = y.shape[0]
    print(f'T {T}, n_stock {n_stock}')

    beta_mean = np.array([1.0] * n_stock)
    beta_var = np.array([1.0] * n_stock)

    alpha_mean = np.array([70.0] * n_stock)
    alpha_var = np.array([10.0] * n_stock) 

    eta_mean = np.array([0.0])
    eta_var = np.array([2.0])

    v_alpha = 10.0 
    v_beta = 0.1

    w_alpha = 10.0
    w_beta = 0.1

    beta_init = np.array([0.1] * n_stock )
    alpha_init = np.nanmean(y, axis = 0)
    eta_init = 0.0 * np.ones(1) 
    w_init = 0.0001 * np.ones(1) 
    v_init = 0.001*np.ones(n_stock) 

    init_x = np.array([0.0]).reshape((1,1))
    init_c = np.identity((1)) * 10


    B_vec = np.zeros((N+1,1))
    B_vec[0] = eta_init

    R_conds = np.zeros((N,y.shape[0],2))

    w = np.zeros((N+1,1)) * 0.3
    w[0] = w_init

    # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
    beta_vec = np.zeros((N+1,n_stock))
    beta_vec[0] = beta_init

    A_vec = np.zeros((N+1,n_stock))
    A_vec[0] = alpha_init

    v = np.zeros((N+1, n_stock))
    v[0] = v_init

    states_store = np.zeros((N, T, 1))

    G = np.identity(1)
    # F = np.zeros((y.shape[1], (nr_groups + 1)))

    # constraints
    beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])


    for i in range(1,N+1):
        print( f'{i} of {N} ')

        F = np.zeros((n_stock,  1))
        F[:,0] = beta_vec[i-1,:n_stock]
        # for j in range(F.shape[0]):
        #     F[j,  1] = beta_vec[i-1, n_stock + j]

        smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond)
        R_conds[i-1] = R_cond

        # Constraint
        smooth_state_new =   smooth_state[1:,0]  - np.mean(smooth_state[1:,0] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1,:,0] = smooth_state_new

        smooth_state_new = np.reshape(smooth_state_new, (smooth_state_new.shape[0],1))

        # sample beta_i
        for j in range(y.shape[1]):
            var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
            tmp1 = y[:,j]*smooth_state_new[:,0]
            tmp2 = A_vec[i-1, j]*smooth_state_new[:,0] 

            avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 ))/v[i-1,j]) * var
            beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # constraints
        # print(beta_vec[i, :n_stock].sum())
        beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])
        # print(beta_vec[i, :n_stock].sum())

        # sample alpha
        for j in range(n_stock):
            var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
            avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] )/v[i-1,j]
            avg += alpha_mean[j]/alpha_var[j]
            avg *= var
            A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # sample variance of observation
        for j in range(n_stock):
            alpha = (y.shape[0]/2.0) + v_alpha
            # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
            beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] ) ** 2) + v_beta
            # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
            v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


        # state equations
        var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[0]))
        avg = np.nansum(smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]) / w[i-1,0]
        avg += (eta_mean[0]/eta_var[0])
        avg *= var
        B_vec[i,0] = np.random.normal(avg, np.sqrt(var))

        alpha_w_tmp = y.shape[0]/2.0 + w_alpha
        beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1),0] - B_vec[i, 0]) ** 2) + w_beta
        #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
        # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
        w[i,0] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


    return w, v, beta_vec, A_vec, B_vec, states_store, R_conds


def lc_sector_test_2(N, y, calc_cond = False):
    """
    Calculate Kalman Smoother
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """

    n_stock = y.shape[1]
    T = y.shape[0]
    print(f'T {T}, n_stock {n_stock}')

    beta_mean = np.array([1.0] * n_stock + [1.0] * n_stock)
    beta_var = np.array([1.0] * n_stock + [1.0] * n_stock)

    alpha_mean = np.array([70.0] * n_stock)
    alpha_var = np.array([10.0] * n_stock) 

    eta_mean = np.array([0.0] * 2)
    eta_var = np.array([2.0] * 2)

    v_alpha = 10.0 
    v_beta = 0.1

    w_alpha = 10.0
    w_beta = 0.1

    beta_init = np.array([0.4] * n_stock + [0.6] * n_stock )
    alpha_init = np.nanmean(y, axis = 0)
    eta_init = 0.0 * np.ones(2) 
 

    init_x = np.array([0.0] * 2)
    init_c = np.identity((2)) * 10


    B_vec = np.zeros((N+1,2))
    B_vec[0] = eta_init

    R_conds = np.zeros((N,y.shape[0],2))

    w_init = np.array([0.3, 0.5])#0.0001 * np.ones(2) 
    w = np.zeros((N+1,2)) * 0.3
    w[:] = w_init

    # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
    beta_vec = np.zeros((N+1,2*n_stock))
    beta_vec[:] = beta_init

    A_vec = np.zeros((N+1,n_stock))
    A_vec[:] = np.array(n_stock * [60.0]) #alpha_init

    v_init = np.ones(n_stock)*0.03# 0.001*np.ones(n_stock)
    v = np.zeros((N+1, n_stock))
    v[:] = v_init

    states_store = np.zeros((N, T, 2))

    G = np.identity(2)
    # F = np.zeros((y.shape[1], (nr_groups + 1)))

    # constraints
    beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])
    beta_vec[0, n_stock:] = beta_vec[0, n_stock:]/np.sum(beta_vec[0, n_stock:])


    for i in range(1,N+1):
        print( f'{i} of {N} ')

        F = np.zeros((n_stock,  2))
        F[:,0] = beta_vec[i-1,:n_stock]
        F[:,1] = beta_vec[i-1,n_stock:]

        smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond)
        R_conds[i-1] = R_cond

        # Constraint
        smooth_state_new =   smooth_state[1:,:]  - np.mean(smooth_state[1:,:] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1] = smooth_state_new

        smooth_state_new = np.reshape(smooth_state_new, (smooth_state_new.shape[0],2))

        # # sample beta_i
        # for j in range(y.shape[1]):
        #     var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
        #     tmp1 = y[:,j]*smooth_state_new[:,0]
        #     tmp2 = A_vec[i-1, j]*smooth_state_new[:,0]
        #     tmp3 = beta_vec[i-1,j + n_stock] * smooth_state_new[:,0] * smooth_state_new[:,1] 

        #     avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 -tmp3 ))/v[i-1,j]) * var
        #     beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # # constraints
        # beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])

        # # sample beta_s_i
        # for j in range(y.shape[1]):
        #     var = 1.0 / ((np.sum(smooth_state_new[:,1] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
        #     tmp1 = y[:,j]*smooth_state_new[:,1]
        #     tmp2 = A_vec[i-1, j]*smooth_state_new[:,1]
        #     tmp3 = beta_vec[i,j] * smooth_state_new[:,0] *smooth_state_new[:,1] 

        #     avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 -tmp3 ))/v[i-1,j]) * var
        #     beta_vec[i,j + n_stock] = np.random.normal(avg, np.sqrt(var))

        # # constraints
        # beta_vec[i, n_stock:] = beta_vec[i, n_stock:]/np.sum(beta_vec[i, n_stock:])

        # sample alpha
        # for j in range(n_stock):
        #     var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
        #     avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,j + n_stock]*smooth_state_new[:, 1] )/v[i-1,j]
        #     avg += alpha_mean[j]/alpha_var[j]
        #     avg *= var
        #     A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # # sample variance of observation
        # for j in range(n_stock):
        #     alpha = (y.shape[0]/2.0) + v_alpha
        #     # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
        #     beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,j + n_stock]*smooth_state_new[:, 1] ) ** 2) + v_beta
        #     # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
        #     v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


        # state equations
        for j in range(2):
            var = 1.0 / ((y.shape[0] / w[i-1,j]) + (1.0 / eta_var[j]))
            avg = np.nansum(smooth_state_new[1:, j] - smooth_state_new[:(smooth_state_new.shape[0]-1), j]) / w[i-1,j]
            avg += (eta_mean[j]/eta_var[j])
            avg *= var
            B_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # for j in range(2):
        #     alpha_w_tmp = y.shape[0]/2.0 + w_alpha
        #     beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,j] - smooth_state_new[:(smooth_state_new.shape[0]-1),j] - B_vec[i, j]) ** 2) + w_beta
        #     #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
        #     # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
        #     w[i,j] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


    return w, v, beta_vec, A_vec, B_vec, states_store, R_conds