import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import unitary_group
import random
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.stats import unitary_group
import cvxpy as cp
import time
import matplotlib.pyplot as plt
import pickle
import os
import random

from qzsg_funcs import *

############################
# Parameters to Initialize #
############################

n_alice = 1 # game size / 2
n_experiments = 10
verbose = True
rand_seed = 42
num_iters = {1: 50000, 2: 50000, 3:50000} # total iteration number for each game size

##########
# Script #
##########

# Random seed
random.seed(rand_seed)

# we will assume alice and bob have the same number of qubits
n_bob = n_alice
n_qubits = n_alice+n_bob


num_iters_mmwu = num_iters[n_alice] 
num_iters_ommwu = num_iters_mmwu

# select iterations to be saved
iters_save = np.logspace(np.log(1), np.log(num_iters[n_alice]), 20)
iters_save  = list(dict.fromkeys([int(x) for x in iters_save if x < num_iters[n_alice]]))
iters_save.append(num_iters[n_alice]-1)
if verbose:
    print("Saved iteration numbers:", iters_save)

step_size = step_sizes[n_alice]

# experiment file path 
file_path = 'experiment2_'+str(n_alice)+'_qubits_'+str(n_experiments)+'_experiments.pkl'

for experiment in range(n_experiments):
    povm = generate_random_full_rank_povm(n_qubits)
    payoff_obs = generate_random_full_rank_payoff_obs(povm) 

    mmwu_dual_gaps = []
    mmwu_iter_times = []
    
    ommwu_dual_gaps = []
    ommwu_iter_times = []

    for iter_num in iters_save:
        mmwu_alice, mmwu_bob, mmwu_iter_time = MMWU(n_alice, n_bob, num_iters_mmwu, payoff_obs, 1/np.sqrt(iter_num), [iter_num], decay_step=False)
        mmwu_dual_gaps.append(duality_gap(payoff_obs, [mmwu_alice[-1], mmwu_bob[-1]], n_alice, n_bob))
        mmwu_iter_times.append(mmwu_iter_time[-1])

        ommwu_alice, ommwu_bob, ommwu_iter_time = OMMWU(n_alice, n_bob, num_iters_ommwu, payoff_obs, 1/np.sqrt(iter_num), [iter_num])
        ommwu_dual_gaps.append(duality_gap(payoff_obs, [ommwu_alice[-1], ommwu_bob[-1]], n_alice, n_bob))
        ommwu_iter_times.append(ommwu_iter_time[-1])

    mmwu_dual_gaps = [duality_gap(payoff_obs, [mmwu_alice[i], mmwu_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]
    mmwu_sd_dual_gaps = [duality_gap(payoff_obs, [mmwu_sd_alice[i], mmwu_sd_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]
    ommwu_dual_gaps = [duality_gap(payoff_obs, [ommwu_alice[i], ommwu_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]

    experiment_details = (payoff_obs, iters_save, mmwu_dual_gaps, mmwu_iter_times, ommwu_dual_gaps, ommwu_iter_times)
    append_output(file_path, experiment_details)

    if verbose:
        print(" ---- EXPERIMENT "+str(experiment)+" COMPLETED ---- ")