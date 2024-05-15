"""
Numerical Methods Package: Utilities
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

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
from ODE import eulers_method


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
    "BE": backwards_euler,
    "E": eulers_method
}

def compute_error(method, f, df, ddf, h, interval, y0):
    if method.__name__ == "taylors_method_O3":
        ts, ys = method(f, df, ddf, h, interval[0], interval[1], y0)
    else:
        ts, ys = method(f, h, interval[0], interval[1], y0)

    # Use solve_ivp with a fine mesh for a reference solution
    sol = solve_ivp(f, interval, [y0], t_eval=np.linspace(interval[0], interval[1], 10000))

    # Interpolate the method's solution to the reference solution's time points
    y_interp = np.interp(sol.t, ts, ys)

    # Calculate the maximum error
    error = np.max(np.abs(sol.y[0] - y_interp))
    return error


def ode_convergence_analysis(method, f, interval, y0, exact_solution=None, df=None, ddf=None):
    """
    Analyzes the convergence of a given ODE solving method.

    Parameters:
    - method: A function that solves an ODE using a specific numerical method.
              It must accept parameters appropriate to the method.
    - f: The function representing the ODE dy/dt = f(t, y).
    - interval: A tuple (start, end) defining the interval over which to solve the ODE.
    - y0: The initial condition y(t0).
    - exact_solution: A function representing the exact solution y(t).
    - df: The first derivative of f, required by some methods.
    - ddf: The second derivative of f, required by some methods.

    Returns:
    - A tuple (hs, errors) where hs is a list of step sizes and errors is a list of errors
      at those step sizes compared to the exact solution.
    - An estimation of the rate of convergence as a string "O(h^n)".
    """
    hs = np.geomspace(1e-3, 1e-1, 50) #Makes an array of 10 logarithmically evenly spaced points from the smallest to the largest h
    errors = [] #Initialize a list to store the calculated errors
    convergence_order = 0.0
    
    for h in hs: #Iterates through each step size in hs
        if method.__name__ == "taylors_method_O3": #Checks if O3T since it calls different than the rest
            ts, ys = method(f, df, ddf, h, interval[0], interval[1], y0) #Calls O3T with the passed in parameters
        else: #Otherwise calls according to ODE Solver format
            ts, ys = method(f, h, interval[0], interval[1], y0) #Calls method through the interval and fills the ts and ys with sol
        
        y_final = ys[-1] #Grabs the last sol estimate
        exact_y_final = exact_solution(ts[-1]) #if exact_solution else np.interp(interval[1], ts, ys)
        #print(y_final)
        #print(exact_y_final)
        #Makes a var and fills it with the exact solution at the last time step
        
        # Calculate the error at the final timestep
        error = np.abs(exact_y_final - y_final)
        errors.append(error) #Puts that error in a table of errors corresponding to hs.
    log_hs = np.log(hs)
    log_errors = np.log(errors)
    convergence_order, _ = np.polyfit(log_hs, log_errors, 1)
    
    
    return (hs, errors), convergence_order

def plot_ode_convergence(methods_list, f, interval, y0, title, exact_solution=None, df=None, ddf=None):
    plt.figure(figsize=(12, 8)) #Initializes a plot
    
    for method_abbr in methods_list: #Checks each method included in the call
        method = method_selectors.get(method_abbr) #Calles the analysis function with the passed in methods
        if not method: #Makes sure the method exists in the list
            print(f"Method {method_abbr} is not recognized.") #Prints error message if its not there
            continue

        if method_abbr == "TMO3" and (df is None or ddf is None): #If the method is O3T it checks for derivatives
            print("df and ddf must be provided for TMO3.") #Tells you if its missing derivatives
            continue

        if method_abbr in ["TMO3"]: #Calls convergence analysis for Taylor if its Taylor
            (hs, errors), convergence_order = ode_convergence_analysis(method, f, interval, y0, exact_solution, df, ddf)
        else: #Calls it normally for all others
            (hs, errors), convergence_order = ode_convergence_analysis(method, f, interval, y0, exact_solution)
        plt.scatter(hs, errors)
        plt.loglog(hs, errors, label=f"{method_abbr} - $O(h^{{{convergence_order:.2f}}})$") #Makes LogLog plot of hs and errors
    
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.gca().invert_xaxis()  # Reverse the x-axis direction
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