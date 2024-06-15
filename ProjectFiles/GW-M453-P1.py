"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods III 
T. Fogarty, Spring 2024
Project 1
"""

import numpy as np
import matplotlib.pyplot as plt
from LinearAlgebra import power_method
from PDE import FTCSE, BTCSI, CrankNicholson
from Utilities import plot_pde_convergence, calculate_dt, plot_pde_solution, plot_pde_error




A = [
    [2, 1, -1, 3],
    [1, 7, 0, -1],
    [-1, 0, 4, 2],
    [3, -1, -2, 1]
]

A_np = np.array(A)

# Define the spatial domain and time interval
a, b = 0, 1
d = 0.1  # Time interval
nx_values = [10, 20, 40, 80, 160]  # Different spatial resolutions for convergence analysis
alpha = 1  # Thermal diffusivity coefficient

def initial_condition(x):
    return np.sin(np.pi * x) * (1 + 2 * np.cos(np.pi * x))

def boundary_condition_0(t):
    return 0

def boundary_condition_1(t):
    return 0

def exact_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t) + np.sin(2 * np.pi * x) * np.exp(-4 * np.pi**2 * t)

solvers = [FTCSE, BTCSI, CrankNicholson]
solver_names = ["FTCSE", "BTCSI", "Crank-Nicholson"]


eigenvalue, eigenvector = power_method(A)
np_eigenvalue, np_eigenvector = np.linalg.eig(A_np)

print(f"Power Method Eigenvalue: {eigenvalue}")
print(f"Power Method Eigenvector: {eigenvector}")

print(f"Built-in Eigenvalues: {np_eigenvalue}")
print(f"Built-in Eigenvectors: {np_eigenvector}")

plt.close("all")
for solver, solver_name in zip(solvers, solver_names):
    plot_pde_solution(solver, solver_name, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, 40)
    plot_pde_error(solver, solver_name, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, 40, exact_solution)
    