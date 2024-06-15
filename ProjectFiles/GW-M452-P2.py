"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods II 
T. Fogarty, Winter 2024
Project 2
"""

import numpy as np
from ODE import runge_kutta_2_midpoint, taylors_method_O3, runge_kutta_4, adams_bashforth_4
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from SpecialNumbers import pi


# ODE 1a definitions
def f_1a(t, y):
    return y + 2*t*np.exp(2*t)
def df_1a(t, y):  # y''
    return 2*np.exp(2*t) + 4*t*np.exp(2*t)
def ddf_1a(t, y):  # y'''
    return 8*np.exp(2*t) + 8*t*np.exp(2*t)


# ODE 1b definitions
def f_1b(t, y):
    return 2*t - 3*y + 1
def df_1b(t, y):  # y''
    return 2  
def ddf_1b(t, y):  # y'''
    return 0

# ODE 2 definitions
def f_2(t,y):
    return -np.exp(t) * y
def df_2(t, y):  # y''
    return -np.exp(t) * y
def ddf_2(t, y): # y'''
    return -np.exp(t) * y


#ODE 3 definitions
def f_3(t, y):
    return -np.cos(t) * y
def df_3(t, y):  # y''
    return np.sin(t) * y
def ddf_3(t, y):  # y'''
    return np.cos(t) * y

def plot_t_rk2_sp(f, df, ddf, interval, y0, h, title):
    """
    Plots solutions of an ODE using Taylor's method, RK2 midpoint method, and solve_ivp, and includes the provided title in the plot.
    
    Parameters:
    - f: The ODE function f(t, y).
    - df: The first derivative of f with respect to t.
    - ddf: The second derivative of f with respect to t.
    - interval: Tuple of (l, r) defining the interval over which to solve the ODE.
    - y0: Initial value y(t0).
    - h: Step size for numerical methods.
    - title: String to be used as the plot title.
    """
    l, r = interval
    t0 = interval[0]

    # Solve using Taylor's method
    ts_taylor, ys_taylor = taylors_method_O3(f, df, ddf, h, l, r, y0)
    
    # Solve using RK2 midpoint method
    ts_rk2, ys_rk2 = runge_kutta_2_midpoint(f, h, l, r, y0, t0)
    
    # Solve using RK4
    ts_rk4, ys_rk4 = runge_kutta_4(f, h, l, r, y0)
    
    # Solve using AB4 (ensure you have generated initial values for ys first)
    ts_ab4, ys_ab4 = adams_bashforth_4(f, h, l, r, y0)    
    
    # Solve using solve_ivp for reference
    sol = solve_ivp(f, [l, r], [y0], t_eval=np.linspace(l, r, int((r-l)/h)+1))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sol.t, sol.y[0], '-', label='SciPy Reference', color='green')
    plt.plot(ts_taylor, ys_taylor, label='Taylor Method', color='blue')
    plt.plot(ts_rk2, ys_rk2, '--', label='RK2 Midpoint Method', color='red')
    plt.plot(ts_rk4, ys_rk4, ':', label='RK4 Method', color='purple')
    plt.plot(ts_ab4, ys_ab4, '-.', label='AB4 Method', color='orange')
    plt.title("Soltuion to ODE: " + title)  # Using the title parameter here
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.show()
    
def convergence_analysis(method, f, df, ddf, interval, y0):
    """
    Analyzes the convergence of a given ODE solving method.

    Parameters:
    - method: A function that solves an ODE using a specific numerical method.
              It must accept the parameters (f, h, l, r, y0).
    - f: The function representing the ODE dy/dt = f(t, y).
    - df: The first derivative of f with respect to t.
    - ddf: The second derivative of f with respect to t.
    - interval: A tuple (start, end) defining the interval over which to solve the ODE.
    - y0: The initial condition y(t0).

    Returns:
    - A tuple (hs, errors) where hs is a list of step sizes and errors is a list of errors
      at those step sizes compared to solve_ivp.
    - An estimation of the rate of convergence as a string "O(h^n)".
    """
    hs = np.geomspace(1e-5, 1e-1, 100)  # Example step sizes, adjust as necessary
    errors = []

    for h in hs:
        # Solve the ODE with the specified method
        ts, ys = method(f, h, interval[0], interval[1], y0)
        
        # Use solve_ivp with a fine mesh for a reference solution
        sol = solve_ivp(f, interval, [y0], t_eval=np.linspace(interval[0], interval[1], 1000))
        
        # Interpolate the method's solution to the reference solution's time points
        y_interp = np.interp(sol.t, ts, ys)
        
        # Calculate the maximum error
        error = np.max(np.abs(sol.y[0] - y_interp))
        errors.append(error)
    
    # Estimate convergence rate
    rates = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(np.array(hs[:-1]) / np.array(hs[1:]))
    avg_rate = np.mean(rates)
    
    return (hs, errors), f"Error = O(h^{avg_rate:.2f})"


def plot_convergence(f, df, ddf, interval, y0, title):
    methods = {
        "Taylor's Method O3": lambda f, h, l, r, y0: taylors_method_O3(f, df, ddf, h, l, r, y0),
        "RK2 Midpoint": runge_kutta_2_midpoint,
        "RK4": runge_kutta_4,
        "AB4": adams_bashforth_4
    }
    
    plt.figure(figsize=(12, 6))
    
    for name, method in methods.items():
        # Assume convergence_analysis returns a tuple (hs, errors)
        (hs, errors), convergence_rate = convergence_analysis(method, f, df, ddf, interval, y0)
        plt.loglog(hs, errors, label=f"{name} - {convergence_rate}")
    
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, which="both", ls='--')
    plt.show()

    
def generate_report():
    plt.close('all')
    plot_t_rk2_sp(f_1a, df_1a, ddf_1a, (0,1), 3, 1E-3, r'$dy = y + 2te^{2t}$')
    plot_t_rk2_sp(f_1b, df_1b, ddf_1b, (1,2), 5, 1E-3, r'$dy = 2t - 3t + 1$')
    plot_t_rk2_sp(f_2, df_2, ddf_2, (0,1), 3, 1E-3, r'$dy = -e^{t}y$')
    plot_t_rk2_sp(f_3, df_3, ddf_3, (0,pi/3), 2, 1E-3, r'$dy = -cos(t)y$')
    plot_convergence(f_1a, df_1a, ddf_1a, (0,1), 3, r'$dy = y + 2te^{2t}$')
    plot_convergence(f_1b, df_1b, ddf_1b, (1,2), 5, r'$dy = 2t - 3y + 1$')
    plot_convergence(f_2, df_2, ddf_2, (0,1), 3, r'$dy = -e^{t}y$')
    plot_convergence(f_3, df_3, ddf_3, (0,pi/3), 2, r'$dy = -cos(t)y$')
    
    
generate_report()
    
    