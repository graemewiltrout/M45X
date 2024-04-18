"""
Numerical Methods Package: Utilities
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ODE import taylors_method_O3
from ODE import runge_kutta_2_midpoint
from ODE import runge_kutta_2_heun
from ODE import runge_kutta_4
from ODE import adams_bashforth_2
from ODE import adams_bashforth_4
from ODE import heuns_method
from ODE import abm4
from ODE import forward_euler
from ODE import backwards_euler


def data(f, l, r, h):
    num_steps = int((r - l) / h)
    points = [(l + i * h, f(l + i * h)) for i in range(num_steps + 1)]
    if l + num_steps * h < r:  # Explicitly include the last point
        points.append((r, f(r)))
    return points

method_selectors = {
    "TMO3": taylors_method_O3,
    "RK2M": runge_kutta_2_midpoint,
    "RK2H": runge_kutta_2_heun,
    "RK4": runge_kutta_4,
    "AB2": adams_bashforth_2,
    "AB4": adams_bashforth_4,
    "HM": heuns_method,
    "ABM4": abm4,
    "FE": forward_euler,
    "BE": backwards_euler
}

def ode_convergence_analysis(method, f, interval, y0, df=None, ddf=None):
    """
    Analyzes the convergence of a given ODE solving method.

    Parameters:
    - method: A function that solves an ODE using a specific numerical method.
              It must accept parameters appropriate to the method.
    - f: The function representing the ODE dy/dt = f(t, y).
    - df: The first derivative of f, required by some methods.
    - ddf: The second derivative of f, required by some methods.
    - interval: A tuple (start, end) defining the interval over which to solve the ODE.
    - y0: The initial condition y(t0).

    Returns:
    - A tuple (hs, errors) where hs is a list of step sizes and errors is a list of errors
      at those step sizes compared to solve_ivp.
    - An estimation of the rate of convergence as a string "O(h^n)".
    """
    hs = np.geomspace(1e-5, 1e-1, 100)
    errors = []

    for h in hs:
        if method.__name__ == "taylors_method_O3":
            ts, ys = method(f, df, ddf, h, interval[0], interval[1], y0)
        else:
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

def plot_ode_convergence(methods_list, f, interval, y0, title, df=None, ddf=None):
    plt.figure(figsize=(12, 6))
    
    for method_abbr in methods_list:
        method = method_selectors.get(method_abbr)
        if not method:
            print(f"Method {method_abbr} is not recognized.")
            continue

        if method_abbr == "TMO3" and (df is None or ddf is None):
            print("df and ddf must be provided for TMO3.")
            continue

        if method_abbr in ["TMO3"]:
            (hs, errors), convergence_rate = ode_convergence_analysis(method, f, interval, y0, df, ddf)
        else:
            (hs, errors), convergence_rate = ode_convergence_analysis(method, f, interval, y0)

        plt.loglog(hs, errors, label=f"{method_abbr} - {convergence_rate}")
    
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, which="both", ls='--')
    plt.show()

def system_ode(t, Y, f1, f2):
    """
    Combines two functions f1, f2 into a single system function for use with solve_ivp.
    """
    y1, y2 = Y
    return [f1(t, y1, y2), f2(t, y1, y2)]

def plot_odesys_convergence(methods_list, f1, f2, interval, y1_0, y2_0, title):
    hs = np.geomspace(1e-5, 1e-1, 100)
    plt.figure(figsize=(10, 6))
    
    for method_abbr in methods_list:
        errors = []
        for h in hs:
            ts, y1s, y2s = method_selectors[method_abbr](f1, f2, interval[0], interval[1], h, y1_0, y2_0)
            
            # Reference solution using solve_ivp
            sol = solve_ivp(lambda t, Y: system_ode(t, Y, f1, f2), interval, [y1_0, y2_0], t_eval=np.linspace(interval[0], interval[1], 1000))
            
            # Interpolation of numerical solution to reference time points
            y1_interp = np.interp(sol.t, ts, y1s)
            y2_interp = np.interp(sol.t, ts, y2s)
            
            # Error computation
            error_y1 = np.max(np.abs(sol.y[0] - y1_interp))
            error_y2 = np.max(np.abs(sol.y[1] - y2_interp))
            errors.append(max(error_y1, error_y2))
        
        plt.loglog(hs, errors, label=f"{method_abbr}")

    plt.title(f"Convergence Analysis for Systems: {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Max Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()