## A quadratic speedup in finding Nash equilibria of quantum zero-sum games [https://arxiv.org/abs/2311.10859]
This repository contains code for the experiments presented in the paper titled "A quadratic speedup in finding Nash equilibria of quantum zero-sum games" (2023), by Francisca Vasconcelos, Emmanouil-Vasileios Vlatakis-Gkaragkounis, Panayotis Mertikopoulos, Georgios Piliouras, and Michael I. Jordan.

All the code necessary to rerun/modify the experiments is contained in the following three files:
-  `qzsg_funcs.py`: This file contains all the core functions required to specify a quantum zero-sum game and run MMWU or OMMWU.
- `qzsg_run_experiment.py`: This script is used to run repeated experiments for a quantum zero-sum game of a specified size (with randomly generated full-rank games). The experimental outcomes are saved to a pickle file.
- `qzsg_display_experiment.py`: This script is used to make different plots displaying the experimental outcomes from the saved pickle file.

The experimental data used to create the plots in the main paper are contained in the following files:
- `experiment_1_qubits_50_experiments_FINAL.pkl`: data from the 2-qubit game (in which Alice and Bob each play with 1 qubit)
- `experiment_2_qubits_50_experiments_FINAL.pkl`: data from the 4-qubit game (in which Alice and Bob each play with 2 qubits)
- `experiment_3_qubits_50_experiments_FINAL.pkl`: data from the 6-qubit game (in which Alice and Bob each play with 3 qubits)

To recreate the plots for the 2-, 4-, and 6-qubit games, simply plug the name of the desired file into the `qzsg_display_experiment.py` script.
