"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods II 
T. Fogarty, Winter 2024
Project 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from M45X import data
from M45X import midpoint, romberg
from M45X.NumericalDifferentiation import *
from M45X import eulers_method, runge_kutta_4, adams_bashforth_2
from M45X import pi
from M45X import e

# Define the functions
def y1(x):
    return np.exp(x**2)

def y2(x):
    return np.sin(x)

def y3(x):
    return np.log(x**2 + x)

def y4(t,y):
    return -3 * y * np.sin(t)

# Exact derivatives for comparison
def dy1(x):
    return 2*x*np.exp(x**2)

def dy2(x):
    return np.cos(x)

def dy3(x):
    return (2*x + 1) / (x**2 + x)

# Exact second derivatives for comparison
def d2y1(x):
    return 4*x**2*np.exp(x**2) + 2*np.exp(x**2)

def d2y2(x):
    return -np.sin(x)

def d2y3(x):
    return (2 - 6*x - 3*x**2) / (x**2 + x)**2

y1data = data(y1, -1.5, 1.5, .001)
y2data = data(y2, pi/2, 7 * pi / 2, .001)
y3data = data(y3, 1, (5 * e), .001)

dy1data = data(dy1, -1.5, 1.5, .001)
dy2data = data(dy2, pi/2, 7 * pi / 2, .001)
dy3data = data(dy2, 1, (5 * e), .001)


def dPlot(f, data, dy, h, f_label):
    x_values = [point[0] for point in data]  # Extract x values from the data set
    
    # Calculate finite differences for the x values from the dataset
    finite_diffs = [finite_difference(data, i, 1, 2) for i in range(len(data))]

    # Use x_values for Richardson extrapolation to maintain consistency
    richardson_diffs = [richardson_extrapolation(f, x, h, 1, 2) for x in x_values]

    # Calculate analytical derivative for the consistent x_values
    analytical_diffs = [dy(x) for x in x_values]

    plt.figure(figsize=(12, 8))    
    plt.plot(x_values, analytical_diffs, 'g-', label='Analytical Derivative', linewidth=2)
    plt.plot(x_values[:-1], finite_diffs[:-1], 'r-.', label='Finite Difference (Data)', linewidth=1.5)
    plt.plot(x_values, richardson_diffs, 'b:', label='Richardson Extrapolation (Function)', linewidth=1.5)
    plt.title(f'Derivative Approximations for {f_label}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def d2Plot(f, data, d2y, h, f_label):
    x_values = [point[0] for point in data]  # Extract x values from the data set

    # First derivative using finite differences
    d1 = [finite_difference(data, i, 1, 2) for i in range(len(data))]

    # Second derivative using finite differences on the first derivative
    d2 = [finite_difference(list(zip(x_values, d1)), i, 1, 2) for i in range(len(data))]

    y_values = [y for _, y in data]
    
    # Direct second derivative
    direct_diffs = [d2y(x) for x in x_values]

    # Richardson extrapolation for second derivatives
    richardson_diffs = [richardson_extrapolation(f, x, h, 2, 2) for x in x_values]



    plt.figure(figsize=(12, 8))
    plt.plot(x_values, direct_diffs, 'g-', label='Direct Second Derivative', linewidth=2)
    plt.plot(x_values, richardson_diffs, 'b:', label='Richardson Extrapolation (o=2, a=2)', linewidth=1.5)
    plt.plot(x_values[:-2], d2[:-2], 'r-.', label='Finite Difference Iterative (Data)', linewidth=1)
    plt.title(f'Second Derivative Approximations for {f_label}')
    plt.xlabel('X')
    plt.ylabel('d²Y/dX²')
    plt.legend()
    plt.grid(True)
    plt.show()  
    
    
def plot_convergence(f, df, d2f, f_label, x0):
    hs = np.geomspace(1e-4, 1e-1, 50)  # Geometrically decreasing step sizes
    errors_df = {"Finite Difference": [], "Richardson": [], "Analytical": []}
    errors_d2f = {"Finite Difference": [], "Richardson": [], "Analytical": []}

    for h in hs:
        # Finite difference and Richardson extrapolation for first derivative
        df_fd = finite_difference(data(f, x0-5*h, x0+5*h, h), 5, 1, 2) 
        df_richardson = richardson_extrapolation(f, x0, h, 1, 2)
        df_analytical = df(x0)

        # Compute errors for first derivative
        errors_df["Finite Difference"].append(np.abs(df_fd - df_analytical))
        errors_df["Richardson"].append(np.abs(df_richardson - df_analytical))
        errors_df["Analytical"].append(0)  # Analytical error is always 0

        # Repeat for second derivative
        d2f_fd = finite_difference(data(f, x0-5*h, x0+5*h, h), 5, 2, 2)  # Adjust for second derivative
        d2f_richardson = richardson_extrapolation(f, x0, h, 2, 2)
        d2f_analytical = d2f(x0)

        # Compute errors for second derivative
        errors_d2f["Finite Difference"].append(np.abs(d2f_fd - d2f_analytical))
        errors_d2f["Richardson"].append(np.abs(d2f_richardson - d2f_analytical))
        errors_d2f["Analytical"].append(0)

    # Plotting
    plt.figure(figsize=(12, 6))

    # First Derivative Convergence Plot
    plt.subplot(1, 2, 1)
    for method, errors in errors_df.items():
        plt.loglog(hs, errors, label=method)
    plt.title(f'First Derivative Convergence for {f_label}')
    plt.xlabel('Step Size h')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    # Second Derivative Convergence Plot
    plt.subplot(1, 2, 2)
    for method, errors in errors_d2f.items():
        plt.loglog(hs, errors, label=method)
    plt.title(f'Second Derivative Convergence for {f_label}')
    plt.xlabel('Step Size h')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    
def generate_report():
    plt.close('all')  
    dPlot(y1, y1data, dy1, 0.001, 'y1 = e^(x^2)')
    d2Plot(y1, y1data, d2y1, 0.001, 'y1 = e^(x^2)')

    dPlot(y2, y2data, dy2, 0.001, 'y2 = sin(x)')
    d2Plot(y2, y2data, d2y2, 0.001, 'y2 = sin(x)')

    dPlot(y3, y3data, dy3, 0.001, 'y3 = log(x^2 + x)')
    d2Plot(y3, y3data, d2y3, 0.001, 'y3 = log(x^2 + x)')
    
    plot_convergence(y1, dy1, d2y1, 'y1 = e^(x^2)', x0=1.0)
    plot_convergence(y2, dy2, d2y2, 'y2 = sin(x)', x0=np.pi/4)
    plot_convergence(y3, dy3, d2y3, 'y3 = log(x^2 + x)', x0=2)
    
    intervals = {
    y1: (-1.5, 1.5),
    y2: (np.pi/2, 7 * np.pi / 2),
    y3: (1, 5 * np.e),
    }
    h = 4E-5

# Function to calculate integrals using different methods
    def calculate_integrals():
        for func, (l, r) in intervals.items():
            midpoint_result = midpoint(func, h, l, r)
            depth = 5
            romberg_result = romberg(func, l, r, depth)
                
        # Quad from SciPy for comparison
            quad_result, _ = quad(func, l, r)
                
            print(f"Results for {func.__name__}:")
            print(f"  Midpoint: {midpoint_result}")
            print(f"  Romberg (depth={depth}): {romberg_result}")
            print(f"  SciPy Quad: {quad_result}\n")

    calculate_integrals()
    # Initial conditions
    y0 = 1
    t0 = 0
    t_end = 5
    h = 0.01  # Step size
    
    # Numerical solutions
    ts_euler, ys_euler = eulers_method(y4, h, t0, t_end, y0)
    ts_rk4, ys_rk4 = runge_kutta_4(y4, h, t0, t_end, y0)
    ts_ab2, ys_ab2 = adams_bashforth_2(y4, h, t0, t_end, y0)
    
    # Exact solution for comparison
    ts_exact = np.linspace(t0, t_end, 500)
    ys_exact = np.exp(-ts_exact**2)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(ts_euler, ys_euler, label='Euler\'s Method', linestyle='--', marker='o', markersize=4)
    plt.plot(ts_rk4, ys_rk4, label='Runge-Kutta 4', linestyle='--', marker='s', markersize=4)
    plt.plot(ts_ab2, ys_ab2, label='Adams-Bashforth 2', linestyle='--', marker='^', markersize=4)
    plt.plot(ts_exact, ys_exact, label='Exact Solution', linewidth=2)

    plt.title('Comparison of Numerical Methods for Solving ODEs')
    plt.xlabel('Time (t)')
    plt.ylabel('Solution (y)')
    plt.legend()
    plt.grid(True)
    plt.show()

generate_report()