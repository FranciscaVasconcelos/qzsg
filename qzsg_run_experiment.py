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
n_experiments = 50
verbose = True
rand_seed = 42
step_sizes = {1: 1/3, 2: 1/5, 3:1/18} # step-size for each game size
num_iters = {1: 50000, 2: 50000, 3:50000} # total iteration number for each game size

##########
# Script #
##########

# Random seed
random.seed(rand_seed)

# we will assume alice and bob have the same number of qubits\n",
n_bob = n_alice
n_qubits = n_alice+n_bob


num_iters_mmwu = num_iters[n_alice] 
num_iters_ommwu = num_iters_mmwu

# select iterations to be saved
iters_save = np.logspace(np.log(1), np.log(num_iters[n_alice]), 300)
iters_save  = list(dict.fromkeys([int(x) for x in iters_save if x < num_iters[n_alice]]))
iters_save.append(num_iters[n_alice]-1)
if verbose:
    print("Saved iteration numbers:", iters_save)

step_size = step_sizes[n_alice]

# experiment file path 
file_path = 'experiment_'+str(n_alice)+'_qubits_'+str(n_experiments)+'_experiments.pkl'

for experiment in range(n_experiments):
    povm = generate_random_full_rank_povm(n_qubits)
    payoff_obs = generate_random_full_rank_payoff_obs(povm) 

    if verbose:
        print("********* MMWU "+str(experiment)+" ************")
    mmwu_alice, mmwu_bob, mmwu_iter_times = MMWU(n_alice, n_bob, num_iters_mmwu, payoff_obs, step_size, iters_save, decay_step=False)
    
    if verbose:
        print("********* MMWU-SD "+str(experiment)+" ************")
    mmwu_sd_alice, mmwu_sd_bob, mmwu_sd_iter_times = MMWU(n_alice, n_bob, num_iters_mmwu, payoff_obs, step_size, iters_save, decay_step=True)
    
    if verbose:
        print("********* OMMWU "+str(experiment)+" ************")
    ommwu_alice, ommwu_bob, ommwu_iter_times = OMMWU(n_alice, n_bob, num_iters_ommwu, payoff_obs, step_size, iters_save)
    
    if verbose:
        print("Alice eigenvals:", np.linalg.eigvals(ommwu_alice[-1]))
        print("Bob eigenvals:", np.linalg.eigvals(ommwu_bob[-1]))

    mmwu_dual_gaps = [duality_gap(payoff_obs, [mmwu_alice[i], mmwu_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]
    mmwu_sd_dual_gaps = [duality_gap(payoff_obs, [mmwu_sd_alice[i], mmwu_sd_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]
    ommwu_dual_gaps = [duality_gap(payoff_obs, [ommwu_alice[i], ommwu_bob[i]], n_alice, n_bob) for i in range(len(iters_save))]

    experiment_details = (payoff_obs, iters_save, mmwu_dual_gaps, mmwu_iter_times, mmwu_sd_dual_gaps, mmwu_sd_iter_times, ommwu_dual_gaps, ommwu_iter_times)
    append_output(file_path, experiment_details)

    if verbose:
        print(" ---- EXPERIMENT "+str(experiment)+" COMPLETED ---- ")