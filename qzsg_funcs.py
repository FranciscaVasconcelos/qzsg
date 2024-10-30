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

def identity_state(n_qubits):
    """Generate an identity matrix with the same dimensions as a system density matrix."""
    dim = 2 ** n_qubits  # Dimension of the density matrix
    return np.eye(dim)

def zero_state(n_qubits):
    """Generate a zero matrix with the same dimensions as a system density matrix."""
    dim = 2 ** n_qubits  # Dimension of the density matrix
    return np.zeros((dim,dim),dtype=np.complex_)

def maximally_mixed_state(n_qubits):
    """Generate a maximally mixed state for a set of qubits."""
    dim = 2 ** n_qubits  # Dimension of the density matrix
    identity_matrix = identity_state(n_qubits)  # Identity matrix of size dim x dim
    return identity_matrix / dim

def joint_start_state(n_alice, n_bob):
    """Generate an initial joint state of maximally mixed states."""
    alice_state = maximally_mixed_state(n_alice)
    bob_state = maximally_mixed_state(n_bob)
    return [alice_state, bob_state]


def generate_random_density_matrix(n_qubits):
    """Generate a random density matrix of given dimension."""
    dim = 2**n_qubits
    A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    A = A @ A.conj().T  # Make it Hermitian positive semidefinite
    A = A / np.trace(A)  # Normalize to make it a density matrix
    return A

def partial_trace(rho, n_qubits, subsystem):
    """
    Compute the partial trace of a density matrix `rho`.

    Args:
    - rho (np.ndarray): The density matrix to trace out.
    - n_qubits (list): A list specifying the number of qubits in each of the subsystems.
    - subsystem (int): The subsystem to trace out (0 for the first subsystem, 1 for the second, etc.).

    Returns:
    - np.ndarray: The reduced density matrix.
    """
    # calculate the dimensions of the subsystems
    dims = [2**n_qubits[0], 2**n_qubits[1]]

    # Calculate the total dimension of the system
    total_dim = np.prod(dims)

    # Ensure the input density matrix is of the correct size
    if rho.shape != (total_dim, total_dim):
        raise ValueError("The shape of the density matrix does not match the dimensions provided.")

    # Reshape the density matrix into a multi-dimensional array
    reshaped_rho = rho.reshape(dims + dims)

    # Perform the partial trace over the specified subsystem
    # The axes to sum over are those corresponding to the traced-out subsystem
    remaining_axes = [i for i in range(len(dims)) if i != subsystem]
    traced_out_rho = np.trace(reshaped_rho, axis1=subsystem, axis2=subsystem + len(dims))

    # Reshape the traced-out density matrix back to a 2D array
    new_dims = [dims[i] for i in remaining_axes]
    reduced_rho = traced_out_rho.reshape(np.prod(new_dims), np.prod(new_dims))

    return reduced_rho

def logit_map(Y):
    """Compute the logit map of a matrix."""
    Y = np.asarray(Y, dtype=np.complex128)
    
    # Find the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(Y)
    
    
    max_eval = np.max(evals)

    # Diagonal matrix of eigenvalues
    D = np.diag(evals)
    
    # Diagonal matrix minus largest eigenval
    D_prime = D-max_eval*np.eye(D.shape[0], D.shape[1])

    # Verify the diagonalization
    #Y_diagonal = P @ D @ np.linalg.inv(P)
    
    # Numerically stable computation of logit map
    exp_Y = evecs @ np.diag(np.exp(np.diag(D_prime))) @ np.linalg.inv(evecs)
    
    trace_exp_Y = np.trace(exp_Y)
    
    return exp_Y / trace_exp_Y

def efficient_matrix_log(Y):
    """Compute the matrix logarithm."""
    Y = np.asarray(Y, dtype=np.complex128)
    
    # Find the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(Y)
    
    
    if np.all(evals != 0): # If none of the eigenvalues are zero

        # Diagonal matrix of eigenvalues
        D = np.diag(evals)

        # Numerically stable computation of logit map
        log_Y = evecs @ np.diag(np.log(np.diag(D))) @ np.linalg.inv(evecs)
        
    else:
        log_Y = logm(Y)
    
    return log_Y

def generate_uniform_full_rank_equilibrium(num_qubits):
    """Generate a full-rank equilibrium density matrix with uniform eigenvalues summing to 1."""
    
    d= 2**num_qubits
    
    # Uniform eigenvalues for a maximally mixed state
    eigenvalues = np.ones(d) / d  # Each eigenvalue is 1/d

    # Generate a random unitary matrix to define the basis for the equilibrium state
    U = unitary_group.rvs(d)
    
    # Construct the full-rank density matrix
    equilibrium_state = U @ np.diag(eigenvalues) @ U.conj().T
    return equilibrium_state

##################
# Generate POVMs #
##################

def generate_random_povm(n_qubits):
    """Generate a random set of POVM matrices."""
    povm_elements = []
    num_elements = (2**n_qubits)**2
    for _ in range(num_elements):
        density_matrix = generate_random_density_matrix(n_qubits)
        povm_elements.append(density_matrix)

    povm_sum = sum(povm_elements)
    povm_elements = [sqrtm(np.linalg.inv(povm_sum)) @ e @ sqrtm(np.linalg.inv(povm_sum)) for e in povm_elements]
    return povm_elements

def full_rank_hermitian(d, epsilon=0.2):
    """Generate a random full-rank Hermitian matrix with non-zero eigenvalues."""
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)  # Random complex matrix
    A = A + A.conj().T  # Make Hermitian
    eigenvalues = np.abs(np.random.rand(d)) + epsilon  # Non-zero eigenvalues
    U = unitary_group.rvs(d)
    return U @ np.diag(eigenvalues) @ U.conj().T

def generate_random_full_rank_povm(n_qubits):
    d = 2**n_qubits
    num_elements = (2**n_qubits)**2
    """Generate a random set of POVM matrices."""
    povms = []
    for _ in range(num_elements - 1):
        povm_element = full_rank_hermitian(d)  # Full-rank POVM element
        povms.append(povm_element)
    
    # Ensure completeness: sum of POVM elements = identity matrix
    last_element = np.eye(d) - sum(povms)
    if np.linalg.matrix_rank(last_element) < d:
        U = unitary_group.rvs(d)
        last_element = U @ np.diag(np.abs(np.random.rand(d)) + 0.2) @ U.conj().T
        last_element = np.eye(d) - sum(povms)
    povms.append(last_element)
    return povms

def generate_povm_from_equilibrium(equilibrium):
    
    num_elements = (2**n_qubits)**2
    
    """Generate POVM elements such that the probabilities under the equilibrium are uniform."""
    d = equilibrium.shape[0]
    povms = []
    
    # Generate random full-rank elements, normalized to be positive semi-definite
    for _ in range(num_elements - 1):
        # Create random Hermitian matrix
        M = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        M = M + M.conj().T  # Make Hermitian
        M = M @ M.conj().T  # Make positive semi-definite
        M = M / np.trace(M)  # Normalize

        povms.append(M)

    # Ensure completeness: final POVM element should satisfy sum(POVMs) = identity
    sum_povms = sum(povms)
    last_element = np.eye(d) - sum_povms

    # Adjust if last element is not positive semi-definite (fix numerical errors)
    eigvals = np.linalg.eigvalsh(last_element)
    if np.any(eigvals < 0):
        correction = np.abs(np.min(eigvals)) + 0.01
        last_element += correction * np.eye(d)

    povms.append(last_element)
    return povms

###############################
# Generate Payoff Observables #
###############################

def generate_random_payoff_obs(povm):
    """Generate a random payoff observable given a POVM."""
    num_elements = len(povm)
    payoffs = np.random.uniform(0, 1, size=num_elements)
    U = sum([payoffs[i]*povm[i] for i in range(num_elements)])
    return U

def generate_random_full_rank_payoff_obs(povms):
    """Construct a payoff observable leveraging the POVM structure."""
    d = povms[0].shape[0]
    # Initialize with zeros
    payoff_observable = np.zeros((d, d), dtype=complex)
    
    # Leverage POVM elements to construct the payoff observable
    for i, M_i in enumerate(povms):
        # Weight the POVM elements to create structure in the payoff
        weight = (-1)**i * (i + 1) / len(povms)  # Assign alternating weights as an example
        payoff_observable += weight * M_i

    # Make sure the payoff observable is Hermitian
    payoff_observable = (payoff_observable + payoff_observable.conj().T) / 2
    return payoff_observable

def generate_random_classical_payoff_obs(n_qubits):
    size= 2**n_qubits
    
    # Generate random values for the diagonal
    diagonal_values = np.random.random(size)
    
    # Normalize the diagonal values so that their sum (trace) is 1
    normalized_values = diagonal_values / diagonal_values.sum()
    
    # Create a diagonal matrix using the normalized values
    diagonal_matrix = np.diag(normalized_values)
    
    return diagonal_matrix

def construct_payoff_from_equilibrium(equilibrium, povms):
    """Construct a payoff observable ensuring the equilibrium is optimal."""
    d = equilibrium.shape[0]
    
    # Design a payoff observable to maximize the expected value for the equilibrium
    # Start with a random Hermitian matrix
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    A = A + A.conj().T  # Make Hermitian
    
    # Project it onto the equilibrium to align with the equilibrium strategy
    payoff_observable = A @ equilibrium @ A.conj().T
    return (payoff_observable + payoff_observable.conj().T) / 2  # Ensure Hermitian

######################
# Calculate Feedback #
######################

def alice_feedback(n_alice, n_bob, joint_state, payoff_obs):
    """Alice's Feedback."""
    n_qubits = [n_alice, n_bob] # dimensions of partial trace subsystems
    subsystem = 1 # want to trace out Bob's subsystem

    bob_state = joint_state[1]
    identity_alice = identity_state(n_alice)
    tensor_state = np.kron(identity_alice, bob_state)

    rho = np.dot(payoff_obs.conj().T, tensor_state)

    return partial_trace(rho, n_qubits, subsystem)

def bob_feedback(n_alice, n_bob, joint_state, payoff_obs):
    """Bob's Feedback."""
    n_qubits = [n_alice, n_bob] # dimensions of partial trace subsystems
    subsystem = 0 # want to trace out Alice's subsystem

    alice_state = joint_state[0]
    identity_bob = identity_state(n_bob)
    tensor_state = np.kron(alice_state, identity_bob)

    rho = np.dot(payoff_obs.conj().T, tensor_state)

    return -partial_trace(rho, n_qubits, subsystem)


###########################
# MMWU & OMMWU Algorithms #
###########################

def MMWU(n_alice, n_bob, num_iters, payoff_obs, step_size, iters_save, decay_step=False):
    """MMWU Algorithm with Decaying Step Size."""

    alice_state = maximally_mixed_state(n_alice)  # current state of alice 
    bob_state = maximally_mixed_state(n_bob)  # current state of bob
    
    alice_sum_of_states = zero_state(n_alice)  # to track sum of states for average
    bob_sum_of_states = zero_state(n_bob)  # to track sum of states for average
    
    alice_states_avgs = []  # to save average states
    bob_states_avgs = []  # to save average states

    alice_cum_feedback = zero_state(n_alice)
    bob_cum_feedback = zero_state(n_bob)

    iter_times = [0]  # record times each iteration finishes
    start_time = time.time()  # start time of algo

    for n in range(num_iters):
        # Decaying step size: step_size / sqrt(n + 1)
        if decay_step:
            ss = step_size / np.sqrt(n + 1)
        else:
            ss = step_size

        joint_state = [alice_state, bob_state]
        alice_cum_feedback += alice_feedback(n_alice, n_bob, joint_state, payoff_obs)
        bob_cum_feedback += bob_feedback(n_alice, n_bob, joint_state, payoff_obs)
        
        # Update with decayed step size
        alice_update = logit_map(ss * alice_cum_feedback)
        bob_update = logit_map(ss * bob_cum_feedback)
        
        alice_sum_of_states += alice_update
        bob_sum_of_states += bob_update
        
        if n in iters_save:
            alice_states_avgs.append(alice_sum_of_states / (n + 1))
            bob_states_avgs.append(bob_sum_of_states / (n + 1))
            iter_times.append(time.time() - start_time)
        
        alice_state = alice_update
        bob_state = bob_update

    return (alice_states_avgs, bob_states_avgs, iter_times)



def OMMWU(n_alice, n_bob, num_iters, payoff_obs, step_size, iters_save, step_decay=0):
    """OMMWU Algorithm."""
    
    alice_state = maximally_mixed_state(n_alice) # current state
    bob_state = maximally_mixed_state(n_bob) # current state

    alice_mom = maximally_mixed_state(n_alice) # current momentum
    bob_mom = maximally_mixed_state(n_bob) # current momentum
    
    alice_sum_of_states = zero_state(n_alice) # to track sum of states for average
    bob_sum_of_states = zero_state(n_bob) # to track sum of states for average

    alice_states_avgs = []
    bob_states_avgs = []

    iter_times = [0] #record times each iteration finishes

    start_time = time.time() # start time of algo

    for n in range(num_iters):
        
        log_start = time.time()
        #calculate matrix logarithms
        log_alice_mom = efficient_matrix_log(alice_mom)
        log_bob_mom = efficient_matrix_log(bob_mom)
        
        log_end = time.time()
        
        #print('LOG TIME:', log_end-log_start)

        # calculate updates for \alpha and \beta
        joint_state = [alice_state, bob_state]
        a_feedback = alice_feedback(n_alice, n_bob, joint_state, payoff_obs)
        b_feedback = bob_feedback(n_alice, n_bob, joint_state, payoff_obs)
        
        feedback_time = time.time()
        
        #print('FEEDBACK TIME:', feedback_time-log_end)

        # update \alpha and \beta
        alice_update = logit_map(log_alice_mom + step_size*np.exp(-step_decay*n)*a_feedback)
        alice_sum_of_states += alice_update

        bob_update = logit_map(log_bob_mom + step_size*np.exp(-step_decay*n)*b_feedback)
        bob_sum_of_states += bob_update
        
        logit_time = time.time()
        
        #print('LOGIT TIME:', logit_time-feedback_time)
        
        if n in iters_save:
            alice_states_avgs.append(alice_sum_of_states/(n+1))
            bob_states_avgs.append(bob_sum_of_states/(n+1))  

        # calculate updates for \hat{\alpha} and \hat{\beta}
        joint_state_updated = [alice_update, bob_update]
        a_mom_feedback = alice_feedback(n_alice, n_bob, joint_state_updated, payoff_obs)
        b_mom_feedback = bob_feedback(n_alice, n_bob, joint_state_updated, payoff_obs)

        # update \hat{\alpha} and \hat{\beta}
        alice_mom_update = logit_map(log_alice_mom + step_size*a_mom_feedback)
        bob_mom_update = logit_map(log_bob_mom + step_size*b_mom_feedback)
        
        if n in iters_save:
            iter_times.append(time.time()-start_time)
        
        alice_state = alice_update
        bob_state = bob_update
        alice_mom = alice_mom_update
        bob_mom = bob_mom_update


    return (alice_states_avgs, bob_states_avgs, iter_times)

#########################
# Calculate Duality Gap #
#########################

def maximize_trace_with_complex_density_matrix_constraint(A):
    """Find the density matrix B that maximizes Tr[AB] for a given A."""
    n = A.shape[0]  # Assuming A is a square matrix
    B = cp.Variable((n, n), complex=True)  # Define B as a complex matrix

    # Define the objective function
    objective = cp.Maximize(cp.real(cp.trace(A @ B)))  # Maximize the real part of the trace

    # Define the constraints
    constraints = [B >> 0,  # B is positive semidefinite
                   cp.trace(B) == 1,  # Trace of B is 1
                   B == cp.conj(B.T)]  # B is Hermitian (B = B†)

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # The optimal value of B
    B_opt = B.value

    return problem.value

def duality_gap(payoff_obs, calulated_joint_state, n_alice, n_bob):
    """Calculate the duality gap."""
    a_feedback = alice_feedback(n_alice, n_bob, calulated_joint_state, payoff_obs)
    b_feedback = bob_feedback(n_alice, n_bob, calulated_joint_state, payoff_obs)

    u_max = maximize_trace_with_complex_density_matrix_constraint(a_feedback)
    u_min = maximize_trace_with_complex_density_matrix_constraint(b_feedback)

    return u_max+u_min # I think this should actually be u_max-u_min?

# save experiment outcome
def append_output(file_path, outcome):
    with open(file_path, 'ab') as f:
        pickle.dump(outcome, f)

# open experiment outcome
def read_all_outputs(file_path):
    outputs = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                outputs.append(pickle.load(f))
            except EOFError:
                break
    return outputs