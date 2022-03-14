
from matplotlib.pyplot import axis
import numpy as np
from scipy.stats import invgamma, norm
from scipy.stats import beta as beta_dist
import tqdm
# from sklearn.covariance import LedoitWolf


def KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, regularize = False, cov_step_ceiling = None):
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

    if y.ndim == 1:
        R_vec = np.zeros(y.shape[0])
    else:
        R_vec = np.zeros((y.shape[0], y.shape[1], y.shape[1]))

    R_inv = np.zeros((y.shape[0], V.shape[0], V.shape[0]))

    y_est = np.zeros(y.shape)
    error = np.zeros(y.shape)
    
    R_cond = np.zeros((y.shape[0], 2, ))

    neglik = 0
    for i in range(y.shape[0]):

        if A.shape[0] == y.shape[0]:
            tmp_A = A[i]
        elif A.shape[0] == 1:
            tmp_A = A
        else: 
            raise ValueError("A must be a single vector or an array of vector with size the same as number of observations")

        state_one_step[i] = np.dot(G, state[i]) + B
        state_cov_one_step[i] = W + np.dot(G, state_cov[i]).dot(G.T)
    
        if cov_step_ceiling is not None:
            state_cov_one_step[i] = np.minimum(state_cov_one_step[i], cov_step_ceiling*np.ones(state_cov_one_step[0].shape[0]))
        
        # Set up if any NA
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


        # if np.isnan(y[i]):
        #     state[i+1] = state_one_step[i]
        #     state_cov[i+1] = state_cov_one_step[i]
        #     y_est[i] = np.dot(F, state_one_step[i]) + tmp_A
        #     R_vec[i] = np.dot(F, state_cov_one_step[i]).dot(F.T) + V
        #     continue


        # R = np.dot(tmp_F, state_cov_one_step[i]).dot(tmp_F.T) + V
        R_vec[i] = R

        if R.ndim == 1:
            R_inv[i] = 1/R
            Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(1/R)
        else:

            R_inv[i] = np.linalg.inv(R)
            Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(R_inv[i])

        state[i+1] = state_one_step[i] + np.dot(Kalman_gain, tmp_y - tmp_A - np.dot(tmp_F, state_one_step[i]))
        state_cov[i+1] = state_cov_one_step[i] - np.dot(Kalman_gain, tmp_F).dot(state_cov_one_step[i])
        y_est[i] = np.dot(F, state_one_step[i]) + tmp_A
            

        error[i] = y[i] - y_est[i]

        
        if R.ndim == 1 or error[i].ndim == 1:
            d2 = error[i]*R_inv[i]*error[i]
        else:
            d2 = np.dot(error[i], R_inv[i]).dot(error[i])
        # print(f'det {d1} inverse {d2}')
        #neglik += 0.5* d1 + 0.5 * d2

    # print(f'{np.max(R_cond[:, 0])} vs {np.max(R_cond[:, 1])}')
    # print(f'negative likelihood {neglik}')
    return state, state_cov, state_one_step, state_cov_one_step, R_vec, R_inv, y_est, error, neglik, R_cond


def KalmanSmooth(state, state_one_step, state_cov, state_cov_one_step, G, B, W):
    """
    Calculate Kalman Smoother
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """

    smooth_state = np.zeros((state.shape[0], state.shape[1]))
    smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    Js = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    Rs = np.zeros((state.shape[0], state.shape[1], state.shape[1]))

    smooth_state[-1] = state[-1]
    smooth_state_cov[-1] = state_cov[-1]

    # print(smooth_state[-1])
    # print(smooth_state_cov[-1])

    for i in reversed(range(1, state.shape[0])): 
    
        R = np.dot(G, state_cov[i]).dot(G.T) + W
        # R = np.min((R, [[20]]))
        # R = (1-0.1)*R + 0.1*2*np.identity(R.shape[0])

        if (R.ndim == 1):
            J = np.dot(state_cov[i], G.T).dot(1/R)
        elif (R.ndim == 0):
            J = np.dot(state_cov[i], G.T)*(1/R)
        else:
            J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))


        Js[i] = J
        Rs[i] = R
        smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state[i] - B - state_one_step[i-1])
        smooth_state_cov[i-1] = state_cov[i-1] + np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)





    return smooth_state, smooth_state_cov


def FFBS(y, G, B, W, F, A, V, init_x, init_c, cov_step_ceiling = None):
    """
    Forward Filtering Backward Sampling for a Gibbs sampler
    y_t = Fx_t + A
    x_t = Gx_{t-1} + B 
    """


    (state, state_cov, state_one_step, state_cov_one_step, R_vec, R_inv, y_est, error, neglik, R_cond) = KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, cov_step_ceiling = cov_step_ceiling)


    smooth_state = np.zeros((state.shape[0], state.shape[1]))
    smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    smooth_state_draws = np.zeros((state.shape[0], state.shape[1]))
    Js = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
    Rs = np.zeros((state.shape[0], state.shape[1], state.shape[1]))

    smooth_state[-1] = state[-1]
    smooth_state_cov[-1] = state_cov[-1]
    smooth_state_draws[-1] = np.random.multivariate_normal(smooth_state[-1], smooth_state_cov[-1])

    # print(smooth_state[-1])
    # print(smooth_state_cov[-1])

    for i in reversed(range(1, state.shape[0])): 
    
        R = np.dot(G, state_cov[i]).dot(G.T) + W
        # R = np.min((R, [[20]]))
        # R = (1-0.1)*R + 0.1*2*np.identity(R.shape[0])

        if (R.ndim == 1):
            J = np.dot(state_cov[i], G.T).dot(1/R)
        elif (R.ndim == 0):
            J = np.dot(state_cov[i], G.T)*(1/R)
        else:
            J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))


        Js[i] = J
        Rs[i] = R
        smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state_draws[i] - B - state_one_step[i-1])
        smooth_state_cov[i-1] = state_cov[i-1] - np.dot(J, G).dot(state_cov[i])  #+ np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)

        # smooth_state_cov[i-1] = np.minimum(smooth_state_cov[i-1], 5*np.identity(smooth_state_cov[i-1].shape[0]))

        # print("as")
        # print(smooth_state[i-1])
        # print(J)
        # print(state_cov[i-1])
        # print(smooth_state_cov[i])
        # print(smooth_state_cov[i-1])
        # print(f'{smooth_state[i-1]} {smooth_state_cov[i-1]}')
        smooth_state_draws[i-1] = np.random.multivariate_normal(smooth_state[i-1], smooth_state_cov[i-1])
    return smooth_state_draws, smooth_state, smooth_state_cov, Js, Rs, R_cond






def local_level(N, y, init_params, mh_width, shock = None, verbose = True, cov_step_ceiling = None):
    """
    Calculate Kalman Smoother
    y_t = a + beta x_t + v
    x_t = x_{t-1} + eta + W

    Parameters
    ------------------------------------
    :param N: Number of Gibbs iterations
    :param y: np.array of data n times p:
    :param shock: tuple, index when the shift occurred and the index of the previous observed value . if None no shocks
    :param init_params: dict with the prior parameters and initial guess 

    """

    n_stock = 1
    T = y.shape[0]
    T_obs = np.sum(~np.isnan(y[:,0]))
    print(T_obs)
    #print(n_stock)

    # Priors
    beta_mean = init_params['beta_mean']
    beta_var = init_params['beta_var']

    alpha_mean = init_params['alpha_mean']
    alpha_var = init_params['alpha_var']

    eta_mean = init_params['eta_mean']
    eta_var = init_params['eta_var']

    v_alpha = init_params['v_alpha']
    v_beta = init_params['v_beta']

    w_alpha = init_params['w_alpha']
    w_beta = init_params['w_beta']

    G_alpha = init_params['G_alpha']
    G_beta = init_params['G_beta']

    # initial gibbs
    beta_init = init_params['beta_init']
    alpha_init = init_params['alpha_init']
    eta_init = init_params['eta_init']
    w_init = init_params['w_init']
    v_init = init_params['v_init']
    G_init = init_params['G_init']

    # init kalman
    init_x = init_params['init_x']
    init_c = init_params['init_c']

    # Define vector to store Gibbs values
    B_vec = np.zeros((N+1,1 ))
    B_vec[0] = eta_init

    w = np.zeros((N+1,1 ))
    w[0] = w_init

    beta_vec = np.zeros((N+1,n_stock))
    beta_vec[0] = beta_init

    if shock is None:
        A_vec = np.zeros((N+1,1))
        A_vec[0] = alpha_init
    else:
        A_vec = np.zeros((N+1,2))
        A_vec[0, 0] = alpha_init[0]
        A_vec[0, 1] = alpha_init[1]
    

    v = np.zeros((N+1, n_stock))
    v[0] = v_init

    w = np.zeros((N+1, 1))
    w[0] = w_init

    states_store = np.zeros((N, T, 1 ))

    G_vec = np.zeros((N+1,1 ))
    G_vec[0] = G_init

    # constraints, For single this will always be 1
    #beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])

    if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total= N)

    for i in range(1,N+1):

        F = np.zeros((1,  1))
        F[:,0] = beta_vec[i-1]

        if shock is not None:
            a = np.zeros(T)
            a[:shock[0]] = A_vec[i-1, 0]
            a[shock[0]:] = A_vec[i-1, 1]
            a = np.expand_dims(a, axis = 1)
            # print(a)
        else:
            a = A_vec[i-1]



        #print(y)
        #print(np.diag(v[i-1]))
        #print(np.diag(w[i-1]))
        #print( B_vec[i-1])
        #print(G_vec[i-1,0]*np.identity(1))
        # print("new")
        # print(G_vec[i-1,0]*np.identity(1))
        # print(B_vec[i-1])
        # print(np.diag(w[i-1]))
        # print(F)
        # print(f'{a[0]} {a[-1]}')
        # print(np.diag(v[i-1]))

        #smooth_state, smooth_state, smooth_state_cov, Js, Rs, R_cond = FFBS(y, G_vec[i-1,0]*np.identity(1), B_vec[i-1], np.diag(w[i-1]), F, a,  np.diag(v[i-1]), init_x, init_c, cov_step_ceiling)
        smooth_state_draws, smooth_state, smooth_state_cov, Js, Rs, R_cond = FFBS(y, G_vec[i-1,0]*np.identity(1), B_vec[i-1], np.diag(w[i-1]) , F, a, np.diag(v[i-1]), init_x, init_c, cov_step_ceiling)
        # print((np.sum((smooth_state_draws[2:, 0] - 0.8*smooth_state_draws[1:1000, 0])**2)*0.5 + 2 )/ (500 + 3))
        # Constraint
        smooth_state_new = smooth_state_draws[1:]# - np.mean(smooth_state[1:] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1] = smooth_state_new
        #print(smooth_state_new[1:,0] - smooth_state_new[:(len(smooth_state_new[:,0])-1)])
        # print(np.sum((smooth_state_new[1:,0] - G_vec[i-1,0]*smooth_state_new[:(len(smooth_state_new[:,0])-1),0])**2 ))
        # print(np.nansum((smooth_state_new[1:,0] - G_vec[i-1,0]*smooth_state_new[:(len(smooth_state_new[:,0])-1),0]  - B_vec[i,0]) ** 2)*0.5/500)

        
        # if shock is not None:
        #     var = 1.0 / ((np.sum(smooth_state_new[~np.isnan(y[:, 0]),0] ** 2) / v[i-1,0]) + (1.0 / beta_var))
        #     tmp11 = y[:shock[0],0]*smooth_state_new[:shock[0],0]
        #     tmp21 = A_vec[i-1, 0]*smooth_state_new[:shock[0],0]
        #     tmp12 = y[shock[0]:,0]*smooth_state_new[shock[0]:,0]
        #     tmp22 = A_vec[i-1, 1]*smooth_state_new[shock[0]:,0] 
        #     avg = ((beta_mean/beta_var) + (np.nansum(tmp11 - tmp21) + np.nansum(tmp12 - tmp22))/v[i-1,0]) * var
        #     beta_vec[i,0] = np.random.normal(avg, np.sqrt(var))
        # else:
        #     var = 1.0 / ((np.sum(smooth_state_new[~np.isnan(y[:, 0]),0] ** 2) / v[i-1,0]) + (1.0 / beta_var))
        #     tmp1 = y[:shock[0],0]*smooth_state_new[:,0]
        #     tmp2 = A_vec[i-1, 0]*smooth_state_new[:,0]
        #     avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2))/v[i-1,0]) * var
        #     beta_vec[i,0] = np.random.normal(avg, np.sqrt(var))
        beta_vec[i,0] = 1



        # sample alpha
        if shock is not None:
            # print("shock")
            nr_obs_before_shock = np.sum(~np.isnan(y[:shock[0], 0]))
            var = 1.0 / ((nr_obs_before_shock / v[i-1,0]) + (1 / alpha_var[0]))
            avg = np.nansum(y[:shock[0], 0] - beta_vec[i,0]*smooth_state_new[:shock[0], 0])/v[i-1,0]
            avg += alpha_mean[0]/alpha_var[0]
            avg *= var
            A_vec[i,0] = np.random.normal(avg, np.sqrt(var))
            
            nr_obs_after_shock = np.sum(~np.isnan(y[shock[0]:, 0]))
            var = 1.0 / ((nr_obs_after_shock / v[i-1,0]) + (1 / alpha_var[0]))
            avg = np.nansum(y[shock[0]:, 0] - beta_vec[i,0]*smooth_state_new[shock[0]:, 0]   )/v[i-1,0]
            avg += alpha_mean[1]/alpha_var[1]
            avg *= var
            A_vec[i,1] = np.random.normal(avg, np.sqrt(var))
        else:
            nr_obs = np.sum(~np.isnan(y[:, 0]))
            var = 1.0 / ((nr_obs / v[i-1,0]) + (1 / alpha_var[0]))
            avg = np.nansum(y[:, 0] - beta_vec[i,0]*smooth_state_new[:, 0]   )/v[i-1,0]
            avg += alpha_mean[0]/alpha_var[0]
            avg *= var
            A_vec[i,0] = np.random.normal(avg, np.sqrt(var))

        # Sample variance
        nr_obs = np.sum(~np.isnan(y[:, 0]))
        alpha = (nr_obs/2.0) + v_alpha
        if shock is not None:
            # print("shock")
            beta = 0.5 * np.nansum((y[:shock[0],0] - A_vec[i,0] - beta_vec[i,0]*smooth_state_new[:shock[0], 0] ) ** 2 ) + 0.5 * np.nansum((y[shock[0]:,0] - A_vec[i,1] - beta_vec[i,0]*smooth_state_new[shock[0]:, 0] ) ** 2 ) + v_beta
        else:
            beta = 0.5 * np.nansum((y[:,0] - A_vec[i,0] - beta_vec[i,0]*smooth_state_new[:, 0] ) ** 2) + v_beta
        v[i,0] = invgamma.rvs(a = alpha, loc = 0, scale = beta)

        # state equation  
        # intercept
        x_no_missing = smooth_state_new[~np.isnan(y[:, 0]),0]
        var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[0]))
        # smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]
        avg = np.nansum(x_no_missing[1:] - G_vec[i-1,0]*x_no_missing[:(len(x_no_missing)-1)]) / w[i-1,0]
        avg += (eta_mean[0]/eta_var[0])
        avg *= var
        B_vec[i,0] = 0# np.random.normal(avg, np.sqrt(var))


        # variance
        #x_no_missing = smooth_state_new[~np.isnan(y[:, 0]),0]
        alpha_w_tmp = 0.5*T + w_alpha
        # print(alpha_w_tmp)
        # print(0.5 * np.nansum((smooth_state_new[1:] - G_vec[i-1,0]*smooth_state_new[:(len(smooth_state_new)-1)]  - B_vec[i,0]) ** 2))
        beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:] - G_vec[i-1,0]*smooth_state_new[:(len(smooth_state_new)-1)]  - B_vec[i,0]) ** 2) + w_beta
        # m = beta_w_tmp/(alpha_w_tmp-1)
        # print(m)
        # print(m**2/(alpha_w_tmp-2))
        w[i,0] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)
        # w[i,0] = w[i-1,0]


        # G_vec[i,0] = G_vec[i-1,0]
        cnt = 0
        # G_vec[i,0] = 1.5 
        while ((np.abs(G_vec[i]) >= 1) and (cnt < 100)) or cnt == 0:
            # Sample G using Metropolis hasting
            proposal = np.random.uniform(G_vec[i-1,0]-mh_width, G_vec[i-1,0]+mh_width)
            # print(proposal)

            proposal_post = -0.5 * np.nansum((x_no_missing[1:] - proposal*x_no_missing[:(len(x_no_missing)-1)]  - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(proposal, a = G_alpha, b = G_beta)
            current_post = -0.5 * np.nansum((x_no_missing[1:] - G_vec[i-1,0]*x_no_missing[:(len(x_no_missing)-1)]  - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(G_vec[i-1,0], a = G_alpha, b = G_beta)

            # print(proposal_post -current_post)
            alpha = np.min((0, proposal_post -current_post))
            # print(proposal_post)
            # print(current_post)
            # print(alpha == -np.inf)
            # print(alpha)
            #print(alpha)

            G_vec[i,0] = np.random.choice([proposal, G_vec[i-1,0]], p = [np.exp(alpha), 1-np.exp(alpha)])

            # if G_vec[i] == G_vec[i-1]:
            #     print("reject")
            # else:
            #     print("accept")

            cnt += 1
            # print(G_vec[i])
            # print(" ")


        assert cnt <= 100, "Accept error"

        if verbose:
                    pbar.update()

    if verbose:
        pbar.close()



    return w, v, beta_vec, A_vec, B_vec, states_store, G_vec




def local_level_pair(N, y, init_params, mh_width, shock = None, verbose = True, cov_step_ceiling = None):
    """
    Calculate Kalman Smoother
    y_t = a + beta x_t + v
    x_t = x_{t-1} + eta + W

    Parameters
    ------------------------------------
    :param N: Number of Gibbs iterations
    :param y: np.array of data n times p:
    :param shock: tuple, index when the shift occurred and the index of the previous observed value . if None no shocks
    :param init_params: dict with the prior parameters and initial guess 

    """

    n_stock = 1
    T = y.shape[0]
    T_obs = np.sum(~np.isnan(y[:,0]))
    print(T_obs)
    #print(n_stock)

    # Priors
    beta_mean = init_params['beta_mean']
    beta_var = init_params['beta_var']

    alpha_mean = init_params['alpha_mean']
    alpha_var = init_params['alpha_var']

    eta_mean = init_params['eta_mean']
    eta_var = init_params['eta_var']

    v_alpha = init_params['v_alpha']
    v_beta = init_params['v_beta']

    w_alpha = init_params['w_alpha']
    w_beta = init_params['w_beta']

    G_alpha = init_params['G_alpha']
    G_beta = init_params['G_beta']

    # initial gibbs
    beta_init = init_params['beta_init']
    alpha_init = init_params['alpha_init']
    eta_init = init_params['eta_init']
    w_init = init_params['w_init']
    v_init = init_params['v_init']
    G_init = init_params['G_init']

    # init kalman
    init_x = init_params['init_x']
    init_c = init_params['init_c']

    # Define vector to store Gibbs values
    B_vec = np.zeros((N+1,2))
    B_vec[0] = eta_init

    w = np.zeros((N+1,2))
    w[0] = w_init

    beta_vec = np.zeros((N+1,2))
    beta_vec[0] = beta_init

    if shock is None:
        A_vec = np.zeros((N+1,2))
        A_vec[0] = alpha_init
    else:
        A_vec = np.zeros((N+1,2, 2))
        A_vec[0, 0] = alpha_init[0]
        A_vec[0, 1] = alpha_init[1]
    

    v = np.zeros((N+1, 2))
    v[0] = v_init

    w = np.zeros((N+1, 2))
    w[0] = w_init

    states_store = np.zeros((N, T, 2))

    G_vec = np.zeros((N+1,2,2))
    G_vec[0] = G_init

    # constraints, For single this will always be 1
    #beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])

    if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total= N)

    for i in range(1,N+1):

        F = np.zeros((2,  2))
        F[0,0] = beta_vec[i-1,0]
        F[1,1] = beta_vec[i-1,1]

        if shock is not None:
            a = np.zeros((T, 2))
            a[:shock[0]] = A_vec[i-1, :, 0]
            a[shock[0]:] = A_vec[i-1, :, 1]
            # print(a)
        else:
            a = A_vec[i-1]
        
        smooth_state_draws, smooth_state, smooth_state_cov, Js, Rs, R_cond = FFBS(y, G_vec[i-1], B_vec[i-1], np.diag(w[i-1]) , F, a, np.diag(v[i-1]), init_x, init_c, cov_step_ceiling)

        # Constraint
        smooth_state_new = smooth_state_draws[1:]# - np.mean(smooth_state[1:] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1] = smooth_state_new

        
        # if shock is not None:
        #     var = 1.0 / ((np.sum(smooth_state_new[~np.isnan(y[:, 0]),0] ** 2) / v[i-1,0]) + (1.0 / beta_var))
        #     tmp11 = y[:shock[0],0]*smooth_state_new[:shock[0],0]
        #     tmp21 = A_vec[i-1, 0]*smooth_state_new[:shock[0],0]
        #     tmp12 = y[shock[0]:,0]*smooth_state_new[shock[0]:,0]
        #     tmp22 = A_vec[i-1, 1]*smooth_state_new[shock[0]:,0] 
        #     avg = ((beta_mean/beta_var) + (np.nansum(tmp11 - tmp21) + np.nansum(tmp12 - tmp22))/v[i-1,0]) * var
        #     beta_vec[i,0] = np.random.normal(avg, np.sqrt(var))
        # else:
        #     var = 1.0 / ((np.sum(smooth_state_new[~np.isnan(y[:, 0]),0] ** 2) / v[i-1,0]) + (1.0 / beta_var))
        #     tmp1 = y[:shock[0],0]*smooth_state_new[:,0]
        #     tmp2 = A_vec[i-1, 0]*smooth_state_new[:,0]
        #     avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2))/v[i-1,0]) * var
        #     beta_vec[i,0] = np.random.normal(avg, np.sqrt(var))
        beta_vec[i,0] = 1
        beta_vec[i,1] = 1



        # sample alpha
        for j in range(2):
            if shock is not None:
                nr_obs_before_shock = np.sum(~np.isnan(y[:shock[j], j]))
                var = 1.0 / ((nr_obs_before_shock / v[i-1,j]) + (1 / alpha_var[j,0]))
                avg = np.nansum(y[:shock[j], j] - beta_vec[i,j]*smooth_state_new[:shock[j], j]   )/v[i-1,j]
                avg += alpha_mean[j,0]/alpha_var[j,0]
                avg *= var
                A_vec[i,j,0] = np.random.normal(avg, np.sqrt(var))
                
                nr_obs_after_shock = np.sum(~np.isnan(y[shock[j]:, j]))
                var = 1.0 / ((nr_obs_after_shock / v[i-1,j]) + (1 / alpha_var[j,1]))
                avg = np.nansum(y[shock[j]:, j] - beta_vec[i,j]*smooth_state_new[shock[j]:, j]   )/v[i-1,j]
                avg += alpha_mean[j,1]/alpha_var[j,1]
                avg *= var
                A_vec[i,j,1] = np.random.normal(avg, np.sqrt(var))
            else:
                nr_obs = np.sum(~np.isnan(y[:, j]))
                var = 1.0 / ((nr_obs / v[i-1,j]) + (1 / alpha_var[j]))
                avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, j]   )/v[i-1,j]
                avg += alpha_mean[j]/alpha_var[j]
                avg *= var
                A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

        # print("alpha")

        # Sample variance
        for j in range(2):
            #print(j)
            #beta = 0.5 * np.nansum((y[:shock[j],j] - A_vec[i,j,0] - beta_vec[i,j]*smooth_state_new[:shock[j], j] ) ** 2 ) + 0.5 * np.nansum((y[shock[j]:,j] - A_vec[i,j,1] - beta_vec[i,j]*smooth_state_new[shock[j]:, j] ) ** 2 ) + v_beta
            #print(beta)
            alpha = (T_obs/2.0) + v_alpha[j]
            if shock is not None:
                beta = 0.5 * np.nansum((y[:shock[j],j] - A_vec[i,j,0] - beta_vec[i,j]*smooth_state_new[:shock[j], j] ) ** 2 ) + 0.5 * np.nansum((y[shock[j]:,j] - A_vec[i,j,1] - beta_vec[i,j]*smooth_state_new[shock[j]:, j] ) ** 2 ) + v_beta[j]
            else:
                beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, j] ) ** 2) + v_beta[j]

            v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)
        
        # print("var done")

        # state equation  
        # intercept
        for j in range(2):
            x_no_missing = smooth_state_new
            var = 1.0 / ((T_obs / w[i-1,j]) + (1.0 / eta_var[j]))
            # smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]
            to = (smooth_state_new.shape[0]-1)
            avg = np.nansum(smooth_state_new[1:, j] - G_vec[i-1,j,0]*smooth_state_new[:to, 0]  - G_vec[i-1,j,1]*smooth_state_new[:to, 1]) / w[i-1,j]
            avg += (eta_mean[j]/eta_var[j])
            avg *= var
            B_vec[i,j] = 0# np.random.normal(avg, np.sqrt(var))


        # variance
        for j in range(2):
            x_no_missing = smooth_state_new
            alpha_w_tmp = T_obs/2.0 + w_alpha[j]
            to = (smooth_state_new.shape[0]-1)
            beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,j] - G_vec[i-1,j,0]*smooth_state_new[:to, j] - G_vec[i-1,j,1]*smooth_state_new[:to,j]  - B_vec[i,0]) ** 2) + w_beta[j]
            w[i,j] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)
        # w[i,0] = w[i-1,0]


        cnt = 0
        roots_abs = np.abs(np.roots([G_vec[i,0,0]*G_vec[i,1,1]-G_vec[i,1,0]*G_vec[i,0,1], -(G_vec[i,0,0] +G_vec[i,1,1]), 1]))
        #print("going G")
        while (np.any(roots_abs <= 1) and (cnt < 100)) or cnt == 0:
            # Sample G using Metropolis hasting
            # 0,0
            proposal = np.random.uniform(G_vec[i-1,0,0]-mh_width, G_vec[i-1,0,0]+mh_width)
            to = (smooth_state_new.shape[0]-1)
            proposal_post = -0.5 * np.nansum((smooth_state_new[1:,0] - proposal*smooth_state_new[:to,0] - G_vec[i-1,0,1]*smooth_state_new[:to,1] - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(proposal, a = G_alpha[0,0], b = G_beta[0,0])
            current_post = -0.5 * np.nansum((smooth_state_new[1:,0] - G_vec[i-1,0,0]*smooth_state_new[:to,0] - G_vec[i-1,0,1]*smooth_state_new[:to,1] - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(G_vec[i-1,0,0], a = G_alpha[0,0], b = G_beta[0,0])
            alpha = np.min((0, proposal_post -current_post))
            G_vec[i,0,0] = np.random.choice([proposal, G_vec[i-1,0, 0]], p = [np.exp(alpha), 1-np.exp(alpha)])

            # 0,1
            proposal = np.random.uniform(G_vec[i-1,0,1]-mh_width, G_vec[i-1,0,1]+mh_width)
            to = (smooth_state_new.shape[0]-1)
            proposal_post = -0.5 * np.nansum((smooth_state_new[1:,0] - G_vec[i,0,0]*smooth_state_new[:to,0] - proposal*smooth_state_new[:to,1] - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(proposal, a = G_alpha[0,1], b = G_beta[0,1])
            current_post = -0.5 * np.nansum((smooth_state_new[1:,0] - G_vec[i,0,0]*smooth_state_new[:to,0] - G_vec[i-1,0,1]*smooth_state_new[:to,1] - B_vec[i,0]) ** 2)/w[i,0] +  beta_dist.logpdf(G_vec[i-1,0,1], a = G_alpha[0,1], b = G_beta[0,1])
            alpha = np.min((0, proposal_post -current_post))
            G_vec[i,0,1] = np.random.choice([proposal, G_vec[i-1,0,1]], p = [np.exp(alpha), 1-np.exp(alpha)])

            # 1,1
            proposal = np.random.uniform(G_vec[i-1,1,1]-mh_width, G_vec[i-1,1,1]+mh_width)
            to = (smooth_state_new.shape[0]-1)
            proposal_post = -0.5 * np.nansum((smooth_state_new[1:,1] - proposal*smooth_state_new[:to,1] - G_vec[i-1,1,0]*smooth_state_new[:to,0] - B_vec[i,0]) ** 2)/w[i,1] +  beta_dist.logpdf(proposal, a = G_alpha[1,1], b = G_beta[1,1])
            current_post = -0.5 * np.nansum((smooth_state_new[1:,1] - G_vec[i-1,1,1]*smooth_state_new[:to,1] - G_vec[i-1,1,0]*smooth_state_new[:to,0] - B_vec[i,0]) ** 2)/w[i,1] +  beta_dist.logpdf(G_vec[i-1,1,1], a = G_alpha[1,1], b = G_beta[1,1])
            alpha = np.min((0, proposal_post -current_post))
            G_vec[i,1,1] = np.random.choice([proposal, G_vec[i-1,1,1]], p = [np.exp(alpha), 1-np.exp(alpha)])

            # 1,0
            proposal = np.random.uniform(G_vec[i-1,1,0]-mh_width, G_vec[i-1,1,0]+mh_width)
            to = (smooth_state_new.shape[0]-1)
            proposal_post = -0.5 * np.nansum((smooth_state_new[1:,1] - G_vec[i,1,1]*smooth_state_new[:to,1] - proposal*smooth_state_new[:to,0] - B_vec[i,1]) ** 2)/w[i,1] +  beta_dist.logpdf(proposal, a = G_alpha[1,0], b = G_beta[1,0])
            current_post = -0.5 * np.nansum((smooth_state_new[1:,1] - G_vec[i,1,1]*smooth_state_new[:to,1] - G_vec[i-1,1,0]*smooth_state_new[:to,0] - B_vec[i,1]) ** 2)/w[i,1] +  beta_dist.logpdf(G_vec[i-1,1,0], a = G_alpha[1,0], b = G_beta[1,0])
            alpha = np.min((0, proposal_post -current_post))
            G_vec[i,1,0] = np.random.choice([proposal, G_vec[i-1,1,0]], p = [np.exp(alpha), 1-np.exp(alpha)])

            cnt += 1
            roots_abs = np.abs(np.roots([G_vec[i,0,0]*G_vec[i,1,1]-G_vec[i,1,0]*G_vec[i,0,1], -(G_vec[i,0,0] +G_vec[i,1,1]), 1]))



        assert cnt <= 100, "Accept error"

        if verbose:
                    pbar.update()

    if verbose:
        pbar.close()



    return w, v, beta_vec, A_vec, B_vec, states_store, G_vec




def local_level_trend(N, y, init_params, verbose = True, cov_step_ceiling = None):
    """
    Calculate Kalman Smoother
    y_t = a + beta x_t + v
    x_t = x_{t-1} + eta + W

    :param N: Number of Gibbs iterations
    :param y: np.array of data n times p
    :param init_params: dict with the prior parameters and initial guess 
    """

    n_stock = 1
    T = y.shape[0]
    T_obs = np.sum(~np.isnan(y[:,0]))

    # Priors
    beta_mean = init_params['beta_mean']
    beta_var = init_params['beta_var']

    alpha_mean = init_params['alpha_mean']
    alpha_var = init_params['alpha_var']

    eta_mean = init_params['eta_mean']
    eta_var = init_params['eta_var']

    v_alpha = init_params['v_alpha']
    v_beta = init_params['v_beta']

    w_alpha = init_params['w_alpha']
    w_beta = init_params['w_beta']


    # initial gibbs
    beta_init = init_params['beta_init']
    alpha_init = init_params['alpha_init']
    eta_init = init_params['eta_init']
    w_init = init_params['w_init']
    v_init = init_params['v_init']

    # init kalman
    init_x = init_params['init_x']
    init_c = init_params['init_c']

    # Define vector to store Gibbs values
    B_vec = np.zeros((N+1,1 ))
    B_vec[0] = eta_init

    w = np.zeros((N+1,1 ))
    w[0] = w_init

    beta_vec = np.ones((N+1,n_stock))
    #beta_vec[0] = beta_init

    A_vec = np.zeros((N+1,n_stock))
    A_vec[0] = alpha_init

    v = np.zeros((N+1, n_stock))
    v[0] = v_init

    w = np.zeros((N+1, 2))
    w[0] = w_init

    states_store = np.zeros((N, T, 2 ))

    G = np.identity(2 )
    G[0,1] = 1

    # constraints, For single this will always be 1
    #beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])

    if verbose:
            pbar = tqdm.tqdm(disable=(verbose is False), total= N)

    for i in range(1,N+1):

        F = np.zeros((n_stock,  2))
        F[:,0] = beta_vec[i-1]

        
        smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, cov_step_ceiling)

        # Constraint
        smooth_state_new = smooth_state[1:]# - np.mean(smooth_state[1:] , axis = 0)
        # print(smooth_state_new.shape)
        states_store[i-1] = smooth_state_new



        # sample alpha
        var = 1.0 / ((T_obs / v[i-1,0]) + (1 / alpha_var[0]))
        avg = np.nansum(y[:, 0] - beta_vec[i,0]*smooth_state_new[:, 0]   )/v[i-1,0]
        avg += alpha_mean[0]/alpha_var[0]

        avg *= var
        A_vec[i,0] = np.random.normal(avg, np.sqrt(var))

        # sample variance of observation

        alpha = (T_obs/2.0) + v_alpha
        beta = 0.5 * np.nansum((y[:,0] - A_vec[i,0] - beta_vec[i,0]*smooth_state_new[:, 0] ) ** 2) + v_beta
        v[i,0] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


        # state equation  
        x_no_missing = smooth_state_new[~np.isnan(y[:, 0]),0]
        # var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[0]))
        # smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]
        # avg = np.nansum(x_no_missing[1:] - x_no_missing[:(len(x_no_missing)-1)]) / w[i-1,0]
        # avg += (eta_mean[0]/eta_var[0])
        # avg *= var
        # B_vec[i,0] = np.random.normal(avg, np.sqrt(var))

        alpha_w_tmp = T_obs/2.0 + w_alpha[0]
        # - B_vec[i, 0]
        x1_no_missing = smooth_state_new[~np.isnan(y[:, 0]),0]
        x2_no_missing = smooth_state_new[~np.isnan(y[:, 0]),1]
        beta_w_tmp = 0.5 * np.nansum((x1_no_missing[1:] - x1_no_missing[:(len(x_no_missing)-1)] - x2_no_missing[:(len(x_no_missing)-1)]  - B_vec[i,0]) ** 2) + w_beta[0]
        w[i,0] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


        alpha_w2_tmp = T_obs/2.0 + w_alpha[1]
        # - B_vec[i, 0]
        x1_no_missing = smooth_state_new[~np.isnan(y[:, 0]),0]
        x2_no_missing = smooth_state_new[~np.isnan(y[:, 0]),1]
        beta_w2_tmp = 0.5 * np.nansum((x2_no_missing[1:] - x2_no_missing[:(len(x_no_missing)-1)]) ** 2) + w_beta[1]
        w[i,1] = invgamma.rvs(a = alpha_w2_tmp, loc = 0, scale = beta_w2_tmp)

        if verbose:
                    pbar.update()

    if verbose:
        pbar.close()



    return w, v, beta_vec, A_vec, B_vec, states_store




def local_level_freq(params, y):
    """
    Function for frequentist Estimation of a local tren level model
    """

    eta = 0.0
    w = np.array([params[0]])
    beta = 1
    a = np.array([params[1]])
    v = np.array([params[2]])

    F = np.array(beta)


    init_x = np.array([0])
    init_c = np.array([1])

    (state, 
    state_cov, 
    state_one_step, 
    state_cov_one_step, 
    R,
    R_inv,
    y_est_single, 
    error_single,
    neglik_single,
    R_cond) = KalmanFilter(y, np.identity(1), eta, np.diag(w),F, a, np.diag(v), init_x, init_c )


    smooth, smooth_cov_ = KalmanSmooth(state, state_one_step, state_cov, state_cov_one_step, np.identity(1), eta, np.diag(w))

    y_est_smooth = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        y_est_smooth[i] = np.dot(F, smooth[i+1]) + a


    # calculate negative log likelihood

    neglik = 0
    for i in range(y.shape[0]):
        if np.isnan(y[i,0]):
            continue

        error = y[i,0]- y_est_smooth[i]
        d1 = np.abs(R_inv[i])
        d2 = np.dot(error, R_inv[i]).dot(error)
        neglik += 0.5* d1 + 0.5 * d2

    print(neglik)
    return neglik










# def KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, calc_cond = False, regularize = False, reg_param = 0.1):
#     """
#     Calculate Kalman Filter
#     y = Fx + A
#     x = Gx + B 
#     """
#     state = np.zeros((y.shape[0]+1, init_x.shape[0]))
#     state_cov = np.zeros((y.shape[0]+1, init_x.shape[0], init_x.shape[0]))

#     state[0] = init_x
#     state_cov[0] = init_c

#     state_one_step = np.zeros((y.shape[0], init_x.shape[0]))
#     state_cov_one_step = np.zeros((y.shape[0], init_x.shape[0], init_x.shape[0]))

#     R_cond = np.zeros((y.shape[0], 2))
#     R_inv = np.zeros((y.shape[0], y.shape[1], y.shape[1]))

#     y_est = np.zeros(y.shape)

#     # create a numpy list of co-variates (if a np list is not an input)
#     # This allows dynamic co-variates
#     if A.ndim <= 1:
#         A_new = np.empty((y.shape[0], A.shape[0]))
#         A_new[:] = A
#     else:
#         A_new = A

#     # To deal with missing observations 
#     tmp_A = A_new.copy()
#     for i in range(y.shape[0]):
#         tmp_A[i, np.isnan(y[i,:])] = 0 

#     neglik = 0
#     for i in range(y.shape[0]):

#         state_one_step[i] = np.dot(G, state[i]) + B
#         state_cov_one_step[i] = W + np.dot(G, state_cov[i]).dot(G.T)

#         # To deal with missing observations
#         tmp_F = F.copy()
#         tmp_F[np.isnan(y[i,:]), :] = 0

#         tmp_y = y[i].copy()
#         tmp_y[np.isnan(y[i,:])] = 0

#         tmp_V = V.copy()
#         tmp_V[np.isnan(y[i,:]),:] = 0 
#         tmp_V[:,np.isnan(y[i,:])] = 0
#         tmp_V[np.isnan(y[i,:]),np.isnan(y[i,:])] = 1

#         R = np.dot(tmp_F, state_cov_one_step[i]).dot(tmp_F.T) + tmp_V
#         R[np.isnan(y[i,:]),:] = 0 
#         R[:,np.isnan(y[i,:])] = 0
#         R[np.isnan(y[i,:]),np.isnan(y[i,:])] = 1

#         # R_cond[i, 0] = np.linalg.cond(R)  # condition number before
#         # mu = np.median(np.diag(R))
#         # print(mu)
#         # mu = 2.0
#         # R = (1-0.2)*R + 0.2*mu*np.identity(R.shape[0])
#         # R_cond[i, 1] = np.linalg.cond(R)  # condition number after

#         # if calc_cond:
#         #     R_cond[i, 0] = np.linalg.cond(R)
#         #     if R_cond[i,0] > 1000:
#         #         R = R + reg_param*np.identity(tmp_V.shape[0])
#         #         R_cond[i, 1] = np.linalg.cond(R)

#         # if regularize:
#         #     R = R + reg_param*np.identity(tmp_V.shape[0])

#         #R_cond[i, 0] = np.linalg.cond(R)
#         # R_cond[i, 1] = np.linalg.cond(R)

#         if R.ndim == 1:
#             R_inv[i] = 1/R
#             Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(1/R)
#         else:
#             R_inv[i] = np.linalg.pinv(R)
#             Kalman_gain = np.dot(state_cov_one_step[i], tmp_F.T).dot(R_inv[i])

#         state[i+1] = state_one_step[i] + np.dot(Kalman_gain, tmp_y - tmp_A[i] - np.dot(tmp_F, state_one_step[i]))
#         state_cov[i+1] = state_cov_one_step[i] - np.dot(Kalman_gain, tmp_F).dot(state_cov_one_step[i])
#         y_est[i] = np.dot(F, state[i+1]) + tmp_A[i]

#         e = y[i] - np.dot(F, state_one_step[i]) - tmp_A[i]


#         u, w, vt = np.linalg.svd(R)
#         w = w[w>0]


#         d1 = np.sum(np.log(w))
#         d2 = np.dot(e, R_inv[i]).dot(e)
#         # print(f'det {d1} inverse {d2}')
#         neglik += 0.5* d1 + 0.5 * d2

#     # print(f'{np.max(R_cond[:, 0])} vs {np.max(R_cond[:, 1])}')
#     # print(f'negative likelihood {neglik}')
#     return state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond, R_inv, neglik


# def KalmanSmooth(state, state_one_step, state_cov, state_cov_one_step, G, B, W, regularize = False, reg_param = 0.1):
#     """
#     Calculate Kalman Smoother
#     y_t = Fx_t + A
#     x_t = Gx_{t-1} + B 
#     """

#     smooth_state = np.zeros((state.shape[0], state.shape[1]))
#     smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
#     smooth_state[-1] = state[-1]
#     smooth_state_cov[-1] = state_cov[-1]

#     for i in reversed(range(1, state.shape[0])): 

        
#         R = np.dot(G, state_cov[i]).dot(G.T) + W
#         # mu = np.mean(np.diag(R))
#         # R = (1-0.2)* R + 0.2*mu*np.identity(R.shape[0])

#         if R.ndim == 1:
#             J = np.dot(state_cov[i], G.T).dot(1/R)
#         else:
#             J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))

#         smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state[i] - B - state_one_step[i-1])
#         smooth_state_cov[i-1] = state_cov[i-1] - np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)


#     return smooth_state, smooth_state_cov



# def FFBS(y, G, B, W, F, A, V, init_x, init_c, calc_cond = False, regularize_F = False, regularize_S = False, reg_params = {'reg_f':0.1, 'reg_s':0.1}):
#     """
#     Forward Filtering Backward Sampling for a Gibbs sampler
#     y_t = Fx_t + A
#     x_t = Gx_{t-1} + B 

#     regularize_F: regularize filter?
#     regularize_S: regularize smoother?
#     """


#     state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond, R_inv, neglik = KalmanFilter(y, G, B, W, F, A, V, init_x, init_c, calc_cond, regularize_F, reg_params['reg_f'])


#     smooth_state = np.zeros((state.shape[0], state.shape[1]))
#     smooth_state_cov  = np.zeros((state.shape[0], state.shape[1], state.shape[1]))
#     smooth_state_draws = np.zeros((state.shape[0], state.shape[1]))

#     smooth_state[-1] = state[-1]
#     smooth_state_cov[-1] = state_cov[-1]
#     smooth_state_draws[-1] = np.random.multivariate_normal(smooth_state[-1], smooth_state_cov[-1])

#     for i in reversed(range(1, state.shape[0])): 

#         R = np.dot(G, state_cov[i]).dot(G.T) + W

#         if R.ndim == 1:
#             J = np.dot(state_cov[i], G.T).dot(1/R)
#         else:
#             J = np.dot(state_cov[i], G.T).dot(np.linalg.inv(R))

#         smooth_state[i-1] = state[i-1] + np.dot(J, smooth_state_draws[i] - B - state_one_step[i-1])
#         smooth_state_cov[i-1] = state_cov[i-1] - np.dot(J, smooth_state_cov[i] - state_cov_one_step[i-1]).dot(J.T)

#         smooth_state_draws[i-1] = np.random.multivariate_normal(smooth_state[i-1], smooth_state_cov[i-1])

#     return smooth_state_draws, R_cond


# def lc_sector(N, y, group_membership, init_params, calc_cond = False, regularize_S = False, regularize_F = False, reg_params = {'reg_f':0.1, 'reg_s':0.1} ):
#     """
#     Calculate Kalman Smoother
#     y_t = Fx_t + A
#     x_t = Gx_{t-1} + B 

#     :param N: Number of samples
#     :param y: np.array of data n times p
#     :param group_membership: a array indicating the group stock i belongs to. The grouping should have the form  0,1,2,3,..,k-1. where k is the number of groups.
#     :param init_params: dict with the prior parameters and initial guess 
#     """

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     n_groups = len(np.unique(group_membership))

#     print(f'T {T}, n_stock {n_stock}, n_groups {n_groups}')

#     # Priors
#     beta_mean = init_params['beta_mean']
#     beta_var = init_params['beta_var']

#     alpha_mean = init_params['alpha_mean']
#     alpha_var = init_params['alpha_var']

#     eta_mean = init_params['eta_mean']
#     eta_var = init_params['eta_var']

#     v_alpha = init_params['v_alpha']
#     v_beta = init_params['v_beta']

#     w_alpha = init_params['w_alpha']
#     w_beta = init_params['w_beta']

#     # initial gibbs
#     beta_init = init_params['beta_init']
#     alpha_init = init_params['alpha_init']
#     eta_init = init_params['eta_init']
#     w_init = init_params['w_init']
#     v_init = init_params['v_init']

#     # init kalman
#     init_x = init_params['init_x']
#     init_c = init_params['init_c']


#     B_vec = np.zeros((N+1,1 + n_groups))
#     B_vec[0] = eta_init

#     R_conds = np.zeros((N,y.shape[0],2))

#     w = np.zeros((N+1,1 + n_groups)) * 0.3
#     w[0] = w_init

#     # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
#     beta_vec = np.zeros((N+1,2*n_stock))
#     beta_vec[0] = beta_init

#     A_vec = np.zeros((N+1,n_stock))
#     A_vec[0] = alpha_init

#     v = np.zeros((N+1, n_stock))
#     v[0] = v_init

#     states_store = np.zeros((N, T, 1 + n_groups))

#     G = np.identity(1 + n_groups)
#     # F = np.zeros((y.shape[1], (nr_groups + 1)))

#     # index to help count
#     index = np.array(range(n_stock))

#     # constraints
#     beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])
#     # beta_vec[0, n_stock:] = beta_vec[0, n_stock:]/np.sum(beta_vec[0, n_stock:])
#     for j in range(n_groups):
#         beta_vec[0, n_stock + index[group_membership == j]] = beta_vec[0, n_stock + index[group_membership == j]]/np.sum(beta_vec[0, n_stock + index[group_membership == j]])



#     for i in range(1,N+1):
#         print( f'{i} of {N} ')

#         F = np.zeros((n_stock,  1+ n_groups))
#         F[:,0] = beta_vec[i-1, :n_stock]
#         for j in range(n_groups):
#             F[index[group_membership == j],1 + j] = beta_vec[i-1,n_stock + index[group_membership == j]]

#         smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond, regularize_S = regularize_S, regularize_F = regularize_F, reg_params = reg_params)
#         R_conds[i-1] = R_cond

#         # Constraint
#         smooth_state_new =   smooth_state[1:]  - np.mean(smooth_state[1:] , axis = 0)
#         # print(smooth_state_new.shape)
#         states_store[i-1] = smooth_state_new

#         # sample beta_i
#         for j in range(y.shape[1]):
#             var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var))
#             tmp1 = y[:,j]*smooth_state_new[:,0]
#             tmp2 = A_vec[i-1, j]*smooth_state_new[:,0] 
#             tmp3 = beta_vec[i-1, n_stock + j]*smooth_state_new[:,0]*smooth_state_new[:,1 + group_membership[j]] 

#             avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2 -tmp3))/v[i-1,j]) * var
#             beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # constraints
#         beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])

#         # sample beta_g_i
#         for j in range(y.shape[1]):
#             var = 1.0 / ((np.sum(smooth_state_new[:,1 + group_membership[j]]  ** 2) / v[i-1,j]) + (1.0 / beta_var))
#             tmp1 = y[:,j]*smooth_state_new[:,1 + group_membership[j]] 
#             tmp2 = A_vec[i-1, j]*smooth_state_new[:,1 + group_membership[j]] 
#             tmp3 = beta_vec[i-1, j]*smooth_state_new[:,0]*smooth_state_new[:,1 + group_membership[j]] 

#             avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2 -tmp3))/v[i-1,j]) * var
#             beta_vec[i,n_stock + j] = np.random.normal(avg, np.sqrt(var))

#         # constraints
#         # beta_vec[i, n_stock:] = beta_vec[i, n_stock:]/np.sum(beta_vec[i, n_stock:])
#         for j in range(n_groups):
#             # print(index[group_membership == j])
#             beta_vec[i, n_stock + index[group_membership == j]] = beta_vec[i, n_stock + index[group_membership == j]]/np.sum(beta_vec[i, n_stock + index[group_membership == j]])

#         # sample alpha
#         for j in range(n_stock):
#             var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
#             avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,n_stock + j]*smooth_state_new[:,1 + group_membership[j]]  )/v[i-1,j]
#             avg += alpha_mean[j]/alpha_var[j]
#             avg *= var
#             A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # sample variance of observation
#         for j in range(n_stock):
#             alpha = (y.shape[0]/2.0) + v_alpha
#             # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
#             beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,n_stock + j]*smooth_state_new[:,1 + group_membership[j]]  ) ** 2) + v_beta
#             # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
#             v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


#         # state equations
#         for j in range(1 + n_groups):
#             var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[j]))
#             avg = np.nansum(smooth_state_new[1:, j] - smooth_state_new[:(smooth_state_new.shape[0]-1), j]) / w[i-1,0]
#             avg += (eta_mean[j]/eta_var[j])
#             avg *= var
#             B_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         for j in range(1 + n_groups):
#             alpha_w_tmp = y.shape[0]/2.0 + w_alpha
#             beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,j] - smooth_state_new[:(smooth_state_new.shape[0]-1),j] - B_vec[i, j]) ** 2) + w_beta
#             #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
#             # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
#             w[i,j] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


#     return w, v, beta_vec, A_vec, B_vec, states_store, R_conds


# def lc_single(N, y, init_params, calc_cond = False, regularize_S = False, regularize_F = False, reg_params = {'reg_f':0.1, 'reg_s':0.1}, verbose = True):
#     """
#     Calculate Kalman Smoother
#     y_t = a + beta x_t + v
#     x_t = x_{t-1} + eta + W

#     :param N: Number of Gibbs iterations
#     :param y: np.array of data n times p
#     :param init_params: dict with the prior parameters and initial guess 
#     """

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     #print(n_stock)

#     # Priors
#     beta_mean = init_params['beta_mean']
#     beta_var = init_params['beta_var']

#     alpha_mean = init_params['alpha_mean']
#     alpha_var = init_params['alpha_var']

#     eta_mean = init_params['eta_mean']
#     eta_var = init_params['eta_var']

#     v_alpha = init_params['v_alpha']
#     v_beta = init_params['v_beta']

#     w_alpha = init_params['w_alpha']
#     w_beta = init_params['w_beta']

#     # initial gibbs
#     beta_init = init_params['beta_init']
#     alpha_init = init_params['alpha_init']
#     eta_init = init_params['eta_init']
#     w_init = init_params['w_init']
#     v_init = init_params['v_init']

#     # init kalman
#     init_x = init_params['init_x']
#     init_c = init_params['init_c']


#     B_vec = np.zeros((N+1,1 ))
#     B_vec[0] = eta_init

#     R_conds = np.zeros((N,y.shape[0],2))

#     w = np.zeros((N+1,1 ))
#     w[0] = w_init

#     # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
#     beta_vec = np.zeros((N+1,n_stock))
#     beta_vec[0] = beta_init

#     A_vec = np.zeros((N+1,n_stock))
#     A_vec[0] = alpha_init

#     v = np.zeros((N+1, n_stock))
#     v[0] = v_init

#     states_store = np.zeros((N, T, 1 ))

#     G = np.identity(1 )
#     # F = np.zeros((y.shape[1], (nr_groups + 1)))

#     # index to help count
#     index = np.array(range(n_stock))

#     # constraints
#     beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])

#     if verbose:
#             pbar = tqdm.tqdm(disable=(verbose is False), total= N)

#     for i in range(1,N+1):
#         #print( f'{i} of {N} ')

#         F = np.zeros((n_stock,  1))
#         F[:,0] = beta_vec[i-1]

#         smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond, regularize_S = regularize_S, regularize_F = regularize_F, reg_params = reg_params)
#         R_conds[i-1] = R_cond

#         # Constraint
#         smooth_state_new = smooth_state[1:] - np.mean(smooth_state[1:] , axis = 0)
#         # print(smooth_state_new.shape)
#         states_store[i-1] = smooth_state_new

#         #sample beta_i
#         for j in range(y.shape[1]):
#             var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var))
#             tmp1 = y[:,j]*smooth_state_new[:,0]
#             tmp2 = A_vec[i-1, j]*smooth_state_new[:,0] 

#             avg = ((beta_mean/beta_var) + (np.nansum(tmp1 - tmp2 ))/v[i-1,j]) * var
#             beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         #beta_vec[i,0] = 1.0
#         # constraints
#         beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])

#         # sample alpha
#         for j in range(n_stock):
#             var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
#             avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0]   )/v[i-1,j]
#             avg += alpha_mean[j]/alpha_var[j]
#             avg *= var
#             A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # sample variance of observation
#         for j in range(n_stock):
#             alpha = (y.shape[0]/2.0) + v_alpha
#             # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
#             beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] ) ** 2) + v_beta
#             # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
#             v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


#         # state equation  
#         var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[0]))
#         avg = np.nansum(smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]) / w[i-1,0]
#         avg += (eta_mean[0]/eta_var[0])
#         avg *= var
#         B_vec[i,0] = np.random.normal(avg, np.sqrt(var))

#         alpha_w_tmp = y.shape[0]/2.0 + w_alpha
#         beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1),0] - B_vec[i, 0]) ** 2) + w_beta
#         w[i,0] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)

#         if verbose:
#                     pbar.update()

#     if verbose:
#         pbar.close()



#     return w, v, beta_vec, A_vec, B_vec, states_store, R_conds


# def group_membering(data, group_column):

#     """
#     :param group_column: column of grouping identification
#     """

#     groups = np.unique(data[group_column])

#     vocab = dict()

#     for idx, name in enumerate(groups):

#         vocab[name] = idx

#     return vocab


# def em_cov_missing_data(y, nr_iterations, Sigma, mu):
#     """
#     Estimate the covariance of y, where y has missing data

#     :param y: data size = (N x k) where N is number of data points and k is number of variables 
#     :param y: size (k,) mean of each variable
#     """
    
#     N = y.shape[0]
#     k = y.shape[1]

#     index = np.array(range(y.shape[1]))

#     loglik = np.zeros(nr_iterations+1)
#     loglik[0] = -np.inf

#     for i in range(nr_iterations):
#         print(i)
#         sum_matrix = np.zeros((y.shape[1], y.shape[1]))
#         y_tmp = np.zeros((y.shape[0], y.shape[1]))
#         for t in range(N):
#             index_m = index[np.isnan(y[t,:])]
#             index_o = index[~np.isnan(y[t,:])]

#             inv_sigma = np.linalg.inv(Sigma[np.ix_(index_o, index_o)])
            
#             expected_m = mu[index_m] + np.matmul(Sigma[np.ix_(index_m, index_o)], inv_sigma).dot(y[t,index_o] - mu[index_o])

#             y_tmp[t,index_o] = y[t, index_o]
#             # print(len(index_m))
#             y_tmp[t,index_m] = expected_m

#             sum_matrix[np.ix_(index_o, index_o)] = sum_matrix[np.ix_(index_o, index_o)] + np.outer(y[t,index_o] - mu[index_o], y[t,index_o] - mu[index_o])
#             sum_matrix[np.ix_(index_o, index_m)] = sum_matrix[np.ix_(index_o, index_m)] + np.outer(y[t,index_o] - mu[index_o], expected_m - mu[index_m])
#             sum_matrix[np.ix_(index_m, index_o)] = sum_matrix[np.ix_(index_m, index_o)] + np.outer(expected_m - mu[index_m], y[t,index_o] - mu[index_o])
#             sum_matrix[np.ix_(index_m, index_m)] = sum_matrix[np.ix_(index_m, index_m)] + np.outer(expected_m - mu[index_m], expected_m - mu[index_m]) + Sigma[np.ix_(index_m, index_m)] - np.matmul(Sigma[np.ix_(index_m, index_o)], inv_sigma).dot(Sigma[np.ix_(index_o, index_m)])

#         # loglik[i+1] = -np.trace(np.dot(np.linalg.inv(Sigma), sum_matrix)) - np.log(np.linalg.det(Sigma)) * (N / 2)
#         Sigma = sum_matrix.copy() / N
#         # print(Sigma[0, 0])
#         mu = np.nanmean(y_tmp, axis=0)


#     return Sigma, mu



# def lc_sector_ml(param, y, group_membership):

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     n_groups = len(np.unique(group_membership))

#     # index to help count
#     index = np.array(range(n_stock))

    
#     beta_vec = param[:(2*n_stock)]
#     # constraints
#     # beta_vec[:n_stock] = beta_vec[:n_stock]/np.sum(beta_vec[:n_stock])

#     # for j in range(n_groups):
#     #     beta_vec[n_stock + index[group_membership == j]] = beta_vec[n_stock + index[group_membership == j]]/np.sum(beta_vec[n_stock + index[group_membership == j]])

#     F = np.zeros((n_stock,  1+ n_groups))
#     F[:,0] = beta_vec[:n_stock]
#     for j in range(n_groups):
#         F[index[group_membership == j],1 + j] = beta_vec[n_stock + index[group_membership == j]]

#     G = np.identity(1 + n_groups)
#     eta = param[(2*n_stock):(2*n_stock + 1 + n_groups)]
#     a = param[(2*n_stock + 1 + n_groups): (2*n_stock + 1 + n_groups + n_stock)]
#     w = param[(2*n_stock + 1 + n_groups + n_stock): (2*n_stock + 1 + n_groups + 1 + n_stock + n_groups)]
#     W = np.diag(w)
#     v = param[(2*n_stock + 1 + n_groups + 1 + n_stock + n_groups):(2*n_stock + 1 + n_groups + 1 + n_stock + n_groups + n_stock)]
#     V = np.diag(v)

#     init_x = np.array([0.0] * (1 + n_groups))
#     init_c = np.identity((1 + n_groups)) * 10

#     state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond, R_inv, neglik = KalmanFilter(y, G, eta, W, F, a, V, init_x, init_c, calc_cond = False, regularize = False, reg_param = 0.1)
#     smooth_state, smooth_state_cov = KalmanSmooth(state, state_one_step, state_cov, state_cov_one_step, G, eta, W, regularize = False, reg_param = 0.1)

#     # smooth_state_new = smooth_state[1:]  - np.mean(smooth_state[1:] , axis = 0)
#     smooth_state_new = smooth_state[1:] 
#     # calculate likelihood

#     neglike = 0
#     for i in range(T):
#         e = y[i,:] - np.dot(F, smooth_state_new[i]) - a
#         Sigma = np.dot(F, smooth_state_cov[i]).dot(F.T) + V
#         Sigma_inv = np.linalg.inv(Sigma)
#         neglike += 0.5* np.log(np.linalg.det(Sigma)) + 0.5 * np.dot(e, Sigma_inv).dot(e)

#     print(eta)
#     print(neglike)
#     return neglike




# def lc_sector_ml_2(x, y, group_membership):

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     n_groups = len(np.unique(group_membership))


#     param = x.copy()




#     # index to help count
#     index = np.array(range(n_stock))

    
#     beta_vec = param[:(2*n_stock)]
#     # constraints
#     # beta_vec[:n_stock] = beta_vec[:n_stock]/np.sum(beta_vec[:n_stock])

#     # for j in range(n_groups):
#     #     beta_vec[n_stock + index[group_membership == j]] = beta_vec[n_stock + index[group_membership == j]]/np.sum(beta_vec[n_stock + index[group_membership == j]])

#     F = np.zeros((n_stock,  1+ n_groups))
#     F[:,0] = beta_vec[:n_stock]
#     for j in range(n_groups):
#         F[index[group_membership == j],1 + j] = beta_vec[n_stock + index[group_membership == j]]

#     G = np.identity(1 + n_groups)
#     eta = param[(2*n_stock):(2*n_stock + 1 + n_groups)]
#     a = param[(2*n_stock + 1 + n_groups): (2*n_stock + 1 + n_groups + n_stock)]
#     w = param[(2*n_stock + 1 + n_groups + n_stock): (2*n_stock + 1 + n_groups + n_stock + 1)]
#     v = param[(2*n_stock + 1 + n_groups + n_stock +1): (2*n_stock + 1 + n_groups + n_stock + 2)]
#     W = np.identity(1+n_groups) * w
#     V =np.identity(n_stock) * v
#     # w = param[(2*n_stock + 1 + n_groups + n_stock): (2*n_stock + 1 + n_groups + 1 + n_stock + n_groups)]
#     # W = np.diag(w)
#     # v = param[(2*n_stock + 1 + n_groups + 1 + n_stock + n_groups):(2*n_stock + 1 + n_groups + 1 + n_stock + n_groups + n_stock)]
#     # V = np.diag(v)


#     init_x = np.array([0.0] * (1 + n_groups))
#     init_c = np.identity((1 + n_groups)) * 10

#     state, state_cov, state_one_step, state_cov_one_step, y_est, R_cond, R_inv, neglike = KalmanFilter(y, G, eta, W, F, a, V, init_x, init_c, calc_cond = False, regularize = False, reg_param = 0.1)


#     print(param[0])
#     print(neglike)
#     return neglike






# def lc_sector_test(N, y,x, calc_cond = False):
#     """
#     Calculate Kalman Smoother
#     y_t = Fx_t + A
#     x_t = Gx_{t-1} + B 
#     """

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     print(f'T {T}, n_stock {n_stock}')

#     beta_mean = np.array([1.0] * n_stock)
#     beta_var = np.array([1.0] * n_stock)

#     alpha_mean = np.array([70.0] * n_stock)
#     alpha_var = np.array([10.0] * n_stock) 

#     eta_mean = np.array([0.0])
#     eta_var = np.array([2.0])

#     v_alpha = 10.0 
#     v_beta = 0.1

#     w_alpha = 10.0
#     w_beta = 0.1

#     beta_init = np.array([0.1] * n_stock )
#     alpha_init = np.nanmean(y, axis = 0)
#     eta_init = 0.0 * np.ones(1) 
#     w_init = 0.0001 * np.ones(1) 
#     v_init = 0.001*np.ones(n_stock) 

#     init_x = np.array([0.0]).reshape((1,1))
#     init_c = np.identity((1)) * 10


#     B_vec = np.zeros((N+1,1))
#     B_vec[0] = eta_init

#     R_conds = np.zeros((N,y.shape[0],2))

#     w = np.zeros((N+1,1)) * 0.3
#     w[0] = w_init

#     # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
#     beta_vec = np.zeros((N+1,n_stock))
#     beta_vec[0] = beta_init

#     A_vec = np.zeros((N+1,n_stock))
#     A_vec[0] = alpha_init

#     v = np.zeros((N+1, n_stock))
#     v[0] = v_init

#     states_store = np.zeros((N, T, 1))

#     G = np.identity(1)
#     # F = np.zeros((y.shape[1], (nr_groups + 1)))

#     # constraints
#     beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])


#     for i in range(1,N+1):
#         print( f'{i} of {N} ')

#         F = np.zeros((n_stock,  1))
#         F[:,0] = beta_vec[i-1,:n_stock]
#         # for j in range(F.shape[0]):
#         #     F[j,  1] = beta_vec[i-1, n_stock + j]

#         smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond)
#         R_conds[i-1] = R_cond

#         # Constraint
#         smooth_state_new =   smooth_state[1:,0]  - np.mean(smooth_state[1:,0] , axis = 0)
#         # print(smooth_state_new.shape)
#         states_store[i-1,:,0] = smooth_state_new

#         smooth_state_new = np.reshape(smooth_state_new, (smooth_state_new.shape[0],1))

#         # sample beta_i
#         for j in range(y.shape[1]):
#             var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
#             tmp1 = y[:,j]*smooth_state_new[:,0]
#             tmp2 = A_vec[i-1, j]*smooth_state_new[:,0] 

#             avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 ))/v[i-1,j]) * var
#             beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # constraints
#         # print(beta_vec[i, :n_stock].sum())
#         beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])
#         # print(beta_vec[i, :n_stock].sum())

#         # sample alpha
#         for j in range(n_stock):
#             var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
#             avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] )/v[i-1,j]
#             avg += alpha_mean[j]/alpha_var[j]
#             avg *= var
#             A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # sample variance of observation
#         for j in range(n_stock):
#             alpha = (y.shape[0]/2.0) + v_alpha
#             # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
#             beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] ) ** 2) + v_beta
#             # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
#             v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


#         # state equations
#         var = 1.0 / ((y.shape[0] / w[i-1,0]) + (1.0 / eta_var[0]))
#         avg = np.nansum(smooth_state_new[1:, 0] - smooth_state_new[:(smooth_state_new.shape[0]-1), 0]) / w[i-1,0]
#         avg += (eta_mean[0]/eta_var[0])
#         avg *= var
#         B_vec[i,0] = np.random.normal(avg, np.sqrt(var))

#         alpha_w_tmp = y.shape[0]/2.0 + w_alpha
#         beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1),0] - B_vec[i, 0]) ** 2) + w_beta
#         #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
#         # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
#         w[i,0] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


#     return w, v, beta_vec, A_vec, B_vec, states_store, R_conds


# def lc_sector_test_2(N, y, calc_cond = False):
#     """
#     Calculate Kalman Smoother
#     y_t = Fx_t + A
#     x_t = Gx_{t-1} + B 
#     """

#     n_stock = y.shape[1]
#     T = y.shape[0]
#     print(f'T {T}, n_stock {n_stock}')

#     beta_mean = np.array([1.0] * n_stock + [1.0] * n_stock)
#     beta_var = np.array([1.0] * n_stock + [1.0] * n_stock)

#     alpha_mean = np.array([70.0] * n_stock)
#     alpha_var = np.array([10.0] * n_stock) 

#     eta_mean = np.array([0.0] * 2)
#     eta_var = np.array([2.0] * 2)

#     v_alpha = 10.0 
#     v_beta = 0.1

#     w_alpha = 10.0
#     w_beta = 0.1

#     beta_init = np.array([0.4] * n_stock + [0.6] * n_stock )
#     alpha_init = np.nanmean(y, axis = 0)
#     eta_init = 0.0 * np.ones(2) 
 

#     init_x = np.array([0.0] * 2)
#     init_c = np.identity((2)) * 10


#     B_vec = np.zeros((N+1,2))
#     B_vec[0] = eta_init

#     R_conds = np.zeros((N,y.shape[0],2))

#     w_init = np.array([0.3, 0.5])#0.0001 * np.ones(2) 
#     w = np.zeros((N+1,2)) * 0.3
#     w[:] = w_init

#     # F_vec = np.zeros((N+1,F.shape[0], F.shape[1]))
#     beta_vec = np.zeros((N+1,2*n_stock))
#     beta_vec[:] = beta_init

#     A_vec = np.zeros((N+1,n_stock))
#     A_vec[:] = np.array(n_stock * [60.0]) #alpha_init

#     v_init = np.ones(n_stock)*0.03# 0.001*np.ones(n_stock)
#     v = np.zeros((N+1, n_stock))
#     v[:] = v_init

#     states_store = np.zeros((N, T, 2))

#     G = np.identity(2)
#     # F = np.zeros((y.shape[1], (nr_groups + 1)))

#     # constraints
#     beta_vec[0, :n_stock] = beta_vec[0, :n_stock]/np.sum(beta_vec[0, :n_stock])
#     beta_vec[0, n_stock:] = beta_vec[0, n_stock:]/np.sum(beta_vec[0, n_stock:])


#     for i in range(1,N+1):
#         print( f'{i} of {N} ')

#         F = np.zeros((n_stock,  2))
#         F[:,0] = beta_vec[i-1,:n_stock]
#         F[:,1] = beta_vec[i-1,n_stock:]

#         smooth_state, R_cond = FFBS(y, G, B_vec[i-1], np.diag(w[i-1]), F, A_vec[i-1],  np.diag(v[i-1]), init_x, init_c, calc_cond)
#         R_conds[i-1] = R_cond

#         # Constraint
#         smooth_state_new =   smooth_state[1:,:]  - np.mean(smooth_state[1:,:] , axis = 0)
#         # print(smooth_state_new.shape)
#         states_store[i-1] = smooth_state_new

#         smooth_state_new = np.reshape(smooth_state_new, (smooth_state_new.shape[0],2))

#         # # sample beta_i
#         # for j in range(y.shape[1]):
#         #     var = 1.0 / ((np.sum(smooth_state_new[:,0] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
#         #     tmp1 = y[:,j]*smooth_state_new[:,0]
#         #     tmp2 = A_vec[i-1, j]*smooth_state_new[:,0]
#         #     tmp3 = beta_vec[i-1,j + n_stock] * smooth_state_new[:,0] * smooth_state_new[:,1] 

#         #     avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 -tmp3 ))/v[i-1,j]) * var
#         #     beta_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # # constraints
#         # beta_vec[i, :n_stock] = beta_vec[i, :n_stock]/np.sum(beta_vec[i, :n_stock])

#         # # sample beta_s_i
#         # for j in range(y.shape[1]):
#         #     var = 1.0 / ((np.sum(smooth_state_new[:,1] ** 2) / v[i-1,j]) + (1.0 / beta_var[j]))
#         #     tmp1 = y[:,j]*smooth_state_new[:,1]
#         #     tmp2 = A_vec[i-1, j]*smooth_state_new[:,1]
#         #     tmp3 = beta_vec[i,j] * smooth_state_new[:,0] *smooth_state_new[:,1] 

#         #     avg = ((beta_mean[j]/beta_var[j]) + (np.nansum(tmp1 - tmp2 -tmp3 ))/v[i-1,j]) * var
#         #     beta_vec[i,j + n_stock] = np.random.normal(avg, np.sqrt(var))

#         # # constraints
#         # beta_vec[i, n_stock:] = beta_vec[i, n_stock:]/np.sum(beta_vec[i, n_stock:])

#         # sample alpha
#         # for j in range(n_stock):
#         #     var = 1.0 / ((y.shape[0] / v[i-1,j]) + (1 / alpha_var[j]))
#         #     avg = np.nansum(y[:, j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,j + n_stock]*smooth_state_new[:, 1] )/v[i-1,j]
#         #     avg += alpha_mean[j]/alpha_var[j]
#         #     avg *= var
#         #     A_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # # sample variance of observation
#         # for j in range(n_stock):
#         #     alpha = (y.shape[0]/2.0) + v_alpha
#         #     # beta = np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[1:, 0] - beta_vec[i,y.shape[1] + group_membership[j]]*smooth_state_new[1:, group_membership[j] + 1] ) ** 2) + v_beta
#         #     beta = 0.5 * np.nansum((y[:,j] - A_vec[i,j] - beta_vec[i,j]*smooth_state_new[:, 0] - beta_vec[i,j + n_stock]*smooth_state_new[:, 1] ) ** 2) + v_beta
#         #     # v[i,j] = 1 / np.random.gamma(shape = alpha, scale = beta)
#         #     v[i,j] = invgamma.rvs(a = alpha, loc = 0, scale = beta)


#         # state equations
#         for j in range(2):
#             var = 1.0 / ((y.shape[0] / w[i-1,j]) + (1.0 / eta_var[j]))
#             avg = np.nansum(smooth_state_new[1:, j] - smooth_state_new[:(smooth_state_new.shape[0]-1), j]) / w[i-1,j]
#             avg += (eta_mean[j]/eta_var[j])
#             avg *= var
#             B_vec[i,j] = np.random.normal(avg, np.sqrt(var))

#         # for j in range(2):
#         #     alpha_w_tmp = y.shape[0]/2.0 + w_alpha
#         #     beta_w_tmp = 0.5 * np.nansum((smooth_state_new[1:,j] - smooth_state_new[:(smooth_state_new.shape[0]-1),j] - B_vec[i, j]) ** 2) + w_beta
#         #     #print(0.5 * np.nansum((smooth_state_new[1:,0] - smooth_state_new[:(smooth_state_new.shape[0]-1)] - B_vec[i, 0]) ** 2))
#         #     # w[i,0] = 1 / np.random.gamma(shape = alpha, scale = beta)
#         #     w[i,j] = invgamma.rvs(a = alpha_w_tmp, loc = 0, scale = beta_w_tmp)


#     return w, v, beta_vec, A_vec, B_vec, states_store, R_conds



# def KalmanFilterSingle(y, G, B, W, F, A, V, init_x, init_c, regularize = False):
#     """
#     Calculate Kalman Filter when y is univariate
#     y = Fx + A
#     x = Gx + B 
#     """
#     state = np.zeros((y.shape[0]+1, 1))
#     state_cov = np.zeros((y.shape[0]+1, 1))

#     state[0] = init_x
#     state_cov[0] = init_c

#     state_one_step = np.zeros((y.shape[0], 1))
#     state_cov_one_step = np.zeros((y.shape[0], 1))

#     R_vec = np.zeros(y.shape[0])
#     R_inv = np.zeros((y.shape[0], 1))

#     y_est = np.zeros(y.shape)
#     error = np.zeros(y.shape)

#     neglik = 0
#     for i in range(y.shape[0]):



#         state_one_step[i] = G*state[i] + B
#         state_cov_one_step[i] = W + G*state_cov[i]*G

#         #state_cov_one_step[i] = np.array([[np.min((state_cov_one_step[i], [[5]]))]])

#         if np.isnan(y[i]):
#             state[i+1] = state_one_step[i]
#             state_cov[i+1] = state_cov_one_step[i]
#             y_est[i] = F*state_one_step[i] + tmp_A
#             R_vec[i] = F*state_cov_one_step[i]*F.T + V
#             continue

#         R = F*state_cov_one_step[i]*F + V

#         R_vec[i] = R

#         R_inv[i] = 1/R
#         Kalman_gain = (state_cov_one_step[i]*F)/R

#         state[i+1] = state_one_step[i] + Kalman_gain*(y[i] - tmp_A - F*state_one_step[i])
#         state_cov[i+1] = state_cov_one_step[i] -  Kalman_gain*F*state_cov_one_step[i]
#         y_est[i] = F*state_one_step[i] + tmp_A

#         error[i] = y[i] - y_est[i]

#     return state, state_cov, state_one_step, state_cov_one_step, R_vec, R_inv, y_est, error, neglik, R_cond

