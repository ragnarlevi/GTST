# Run Gibbs sample

from codecs import getwriter
import numpy as np
import pandas as pd
import datetime
import scipy
import importlib
import os, sys
import Gibbs
import tqdm
import concurrent.futures
import argparse
from collections import defaultdict
import os
import pickle


# Define help function

def beta_param(x, y, h = -1, s = 2):


    a = (h-x)*(h**2 + h*(s-2*x) - s*x + x**2 + y)/(s*y)
    b = -(h+s-x)*(h**2 + h*(s-2*x) -s*x + x**2 + y)/(s*y)

    return a, b

def inv_gamma_param(x,y):

    a = (x**2)/y + 2
    b = (x**3)/y + x

    return a, b


def find_var(covariance, G_pair):

    return np.dot(np.identity(4) - np.kron(G_pair, G_pair), np.matrix.flatten(covariance))

def find_jump(y):
    """
    Returns
    -------------------------
    jump_index - index when jump occurs
    jump_index_prev - number of observation (nan not included) before jump
    biggest_jump_direction - size of the jump and direction (minus is jump down)


    
    """

    biggest_jump = 0
    biggest_jump_direction = 0
    prev_value = y[0,0]  # previous non nan value
    jump_index = 0
    jump_index_prev = 0

    observations = [0]
    for i in range(1,y.shape[0]):

        if np.isnan(y[i,0]):
            continue

        observations.append(i)


        this_jump = np.abs(y[i,0] - prev_value)
        this_jump_direction = y[i,0] - prev_value
        if this_jump > biggest_jump:
            biggest_jump = this_jump
            biggest_jump_direction = this_jump_direction
            jump_index_prev = observations[-2]
            jump_index = i

        prev_value = y[i,0]

    return jump_index, jump_index_prev, biggest_jump_direction



if __name__ == '__main__':

    # Arguments to script
    parser = argparse.ArgumentParser()
    # Number of Iterations specifics
    parser.add_argument('-esg1', '--esg1',metavar='', type=str, help='Name of ticker 1')
    parser.add_argument('-esg2', '--esg2',metavar='', type=str, help='Name of ticker 2')
    parser.add_argument('-c', '--Ncores', type=int,metavar='', help='Number of cores/chains')
    parser.add_argument('-N', '--Niterations', type=int,metavar='', help='Number of iterations')
    parser.add_argument('-gwidth', '--gwidth', type=float,metavar='', help='Width of random walk for MH G matrix')
    parser.add_argument('-p', '--path', type=str,metavar='', help='Path to save')
    parser.add_argument('-id', '--id', type=str,metavar='', help='Run identity')


    args = parser.parse_args()

    esg1 = args.esg1
    esg2 = args.esg2
    c = args.Ncores
    N = args.Niterations
    gwidth = args.gwidth
    path = args.path
    id = args.id


    # Load Data 
    esg_pivot_shifted_refined_diff = pd.read_pickle('Yahoo/refined.pkl')

    # Extract pair
    y_obs = np.array(esg_pivot_shifted_refined_diff[[esg1, esg2]])

    # Calculate covariance
    y1mask = np.isfinite(y_obs[:,0])
    y2mask = np.isfinite(y_obs[:,1])
    ymask = y1mask & y2mask
    covariance = np.cov(y_obs[ymask,0], y_obs[ymask,1])

    # Inital parameters and prior specification
    init_params = dict()


    # Set prior parameters

    # each v follows an inverse gamma
    init_params['v_alpha'] = np.array([4, 4])
    init_params['v_beta'] = np.array([4,4])
    # w is wishart
    init_params['w_alpha'] = 4
    init_params['w_beta'] = covariance*4
                                    

    # G prior
    ag, bg = beta_param(0.5, 0.1)

    init_params['G_alpha'] = np.array([[ag,4],
                                    [4,ag]])
    init_params['G_beta'] = np.array([[bg,4], 
                                    [4,bg]])

    # initial gibbs parameters
    init_params['w_init'] =  covariance
    init_params['v_init'] = np.ones(2) 
    init_params['G_init'] = np.array([[0.5,0.0], 
                                        [0.0, 0.5]])

    # init kalman
    init_params['init_x'] = np.array([0, 0] * (1))
    init_params['init_c'] = np.identity(2) * 1

    importlib.reload(sys.modules['Gibbs'])
    importlib.reload(sys.modules['tqdm'])

    chains = defaultdict(dict)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(Gibbs.local_level_pair_variance_wishart, n, y_obs, init_params, gwidth) for n in [N] * c]

        # For loop that takes the output of each process and concatenates them together
        cnt = 0
        for f in concurrent.futures.as_completed(results):
            for k,v in f.result().items():
                print(k)
                chains[cnt][k] = v
                # chains[cnt]['v'] = k[1]
                # chains[cnt]['states'] = k[2]
                # chains[cnt]['G'] = k[3]
                # chains[cnt]['params'] = init_params
                # chains[cnt]['esg1'] = esg1
                # chains[cnt]['esg2'] = esg2

            cnt+=1
            

    # save
    newpath = path + "\\"+ esg1+"_"+esg2
    print(newpath)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(newpath + "\\"+ id+ ".pkl", 'wb') as f:
        pickle.dump(chains, f)














