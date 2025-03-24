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

file_path = 'experiment2_1_qubits_50_experiments.pkl'
experiments = read_all_outputs(file_path)

payoff_obs_exp_vals = []
mmwu_dual_gap_exp_vals = []
mmwu_iter_times_exp_vals = []
ommwu_dual_gap_exp_vals = [] 
ommwu_iter_time_exp_vals = []

for exp_num, exp_outcome in enumerate(experiments):
    payoff_obs, iters_save, mmwu_dual_gaps, mmwu_iter_times, ommwu_dual_gaps, ommwu_iter_times = exp_outcome

    iters_save = list(iters_save)
    payoff_obs_exp_vals.append(payoff_obs)
    mmwu_dual_gap_exp_vals.append(np.abs(mmwu_dual_gaps))
    mmwu_iter_times_exp_vals.append(mmwu_iter_times)
    ommwu_dual_gap_exp_vals.append(np.abs(ommwu_dual_gaps))
    ommwu_iter_time_exp_vals.append(ommwu_iter_times)

#####################################
# Plot Mean and Confidence Interval #
#####################################

mmwu_dual_gap_2d = np.vstack(mmwu_dual_gap_exp_vals) 
mmwu_sd_dual_gap_2d = np.vstack(mmwu_sd_dual_gap_exp_vals) 
ommwu_dual_gap_2d = np.vstack(ommwu_dual_gap_exp_vals) 

# Plot MMWU mean and confidence interval
mean_mmwu_dual_gap = np.mean(mmwu_dual_gap_2d, axis=0)
std_mmwu_dual_gap = np.std(mmwu_dual_gap_2d, axis=0)
n_mmwu = mmwu_dual_gap_2d.shape[0]
mmwu_confidence_interval = 1.96 * std_mmwu_dual_gap / np.sqrt(n_mmwu)  # 95% confidence interval

plt.figure(figsize=(10, 6))
plt.plot(iters_save, mean_mmwu_dual_gap,"--o", color='tab:orange',label='MMWU Mean')
plt.fill_between(iters_save, mean_mmwu_dual_gap - mmwu_confidence_interval, mean_mmwu_dual_gap + mmwu_confidence_interval, color='tab:orange', alpha=0.2, label='MMWU 95% Confidence Interval')

# Plot OMMWU mean and confidence interval
mean_ommwu_dual_gap = np.mean(ommwu_dual_gap_2d, axis=0)
std_ommwu_dual_gap = np.std(ommwu_dual_gap_2d, axis=0)
n_ommwu = ommwu_dual_gap_2d.shape[0]
ommwu_confidence_interval = 1.96 * std_ommwu_dual_gap / np.sqrt(n_ommwu)  # 95% confidence interval

plt.plot(iters_save, mean_ommwu_dual_gap,"--o", color='tab:blue',label='OMMWU Mean')
plt.fill_between(iters_save, mean_ommwu_dual_gap - ommwu_confidence_interval, mean_ommwu_dual_gap + ommwu_confidence_interval, color='tab:blue', alpha=0.2, label='OMMWU 95% Confidence Interval')

plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('|Dual Gap|')
plt.legend()
plt.show()

############################
# Plot Experiment Overlays #
############################

plt.figure(figsize=(10, 6))
for result in mmwu_dual_gap_exp_vals:
    plt.loglog(iters_save, result, color='tab:orange', alpha=0.1)  # Plot each experiment with low opacity

for result in ommwu_dual_gap_exp_vals:
    plt.loglog(iters_save, result, color='tab:blue', alpha=0.1)  # Plot each experiment with low opacity

plt.xlabel('Iteration Number')
plt.ylabel('|Dual Gap|')
plt.title('Overlay of All Experiments')
plt.show()

#################
# Plot Runtimes #
#################

mmwu_runtime_2d = np.vstack(mmwu_iter_times_exp_vals) 
ommwu_runtime_2d = np.vstack(ommwu_iter_time_exp_vals) 
iters_save_time = iters_save.copy()
iters_save_time.insert(0,0)

# Plot MMWU mean and confidence interval
mean_mmwu_runtime = np.mean(mmwu_runtime_2d, axis=0)
mean_mmwu_runtime = np.insert(mean_mmwu_runtime,0,0)
std_mmwu_runtime = np.std(mmwu_runtime_2d, axis=0)
std_mmwu_runtime = np.insert(std_mmwu_runtime,0,0)
n_mmwu = mmwu_runtime_2d.shape[0]
mmwu_conf_int_runtime = 1.96 * std_mmwu_runtime / np.sqrt(n_mmwu)  # 95% confidence interval

print(iters_save_time)
print(mean_mmwu_runtime)

plt.figure(figsize=(10, 6))
plt.plot(iters_save_time, mean_mmwu_runtime,"--o", color='tab:orange',label='MMWU Mean')
plt.fill_between(iters_save_time, mean_mmwu_runtime - mmwu_conf_int_runtime, mean_mmwu_runtime + mmwu_conf_int_runtime, color='tab:orange', alpha=0.2, label='MMWU 95% Confidence Interval')

# Plot OMMWU mean and confidence interval
mean_ommwu_runtime = np.mean(ommwu_runtime_2d, axis=0)
mean_ommwu_runtime = np.insert(mean_ommwu_runtime,0,0)
std_ommwu_runtime = np.std(ommwu_runtime_2d, axis=0)
std_ommwu_runtime = np.insert(std_ommwu_runtime,0,0)
n_ommwu = ommwu_runtime_2d.shape[0]
ommwu_conf_int_runtime = 1.96 * std_ommwu_runtime / np.sqrt(n_ommwu)  # 95% confidence interval



plt.plot(iters_save_time, mean_ommwu_runtime,"--o", color='tab:blue',label='OMMWU Mean')
plt.fill_between(iters_save_time, mean_ommwu_runtime - ommwu_conf_int_runtime, mean_ommwu_runtime + ommwu_conf_int_runtime, color='tab:blue', alpha=0.2, label='OMMWU 95% Confidence Interval')

plt.xlabel('Iteration Number')
plt.ylabel('Runtime')
plt.legend()
plt.show()