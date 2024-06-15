"""
Numerical Methods Package: Utilities
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

from PDE import FTCSE
from PDE import BTCSI
from PDE import CrankNicholson
from PDE import CFD_5pt
from PDE import solve_vibrating_string


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
    
def compute_pde_error(numerical_solution, exact_solution, x, t):
    """
    Computes the error between the numerical solution and the exact solution.

    Parameters:
    - numerical_solution: 2D numpy array
        The numerical solution matrix.
    - exact_solution: function
        The exact solution function u(x, t).
    - x: 1D numpy array
        The spatial grid points.
    - t: 1D numpy array
        The time grid points.

    Returns:
    - error_matrix: 2D numpy array
        The matrix of errors between the numerical and exact solutions.
    """
    exact_values = np.array([[exact_solution(xi, ti) for xi in x] for ti in t])
    error_matrix = np.abs(numerical_solution - exact_values)
    return error_matrix
    
def pde_convergence_analysis(solver, a, b, d, f, g, h, alpha, nx_values, dt_values, exact_solution):
    errors = []
    hs = []
    for nx, dt in zip(nx_values, dt_values):
        nt = int(d / dt)
        if nt == 0:
            raise ValueError(f"Number of time steps nt calculated as zero for dt = {dt}, which is too large.")
        u, x, t = solver(a, b, d, f, g, h, nx, nt, alpha)
        error = compute_pde_error(u, exact_solution, x, t)
        errors.append(error)
        hs.append((b - a) / nx)
    return hs, errors

def calculate_dt(dx, alpha):
    return (0.5 * dx**2) / (1.01 *alpha)


def plot_pde_convergence(solvers, solver_names, a, b, d, f, g, h, alpha, nx_values, exact_solution, title):
    plt.figure(figsize=(12, 8))
    for solver, name in zip(solvers, solver_names):
        dt_values = [calculate_dt((b - a) / nx, alpha) for nx in nx_values]
        hs, errors = pde_convergence_analysis(solver, a, b, d, f, g, h, alpha, nx_values, dt_values, exact_solution)
        plt.loglog(hs, errors, label=f"{name}")
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Spatial Step Size (dx)')
    plt.ylabel('Error')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True, which="both", ls='--')
    plt.show()
'''    
def plot_pde_solution(solver, solver_name, a, b, d, f, g, h, alpha, nx):
    dx = (b - a) / nx
    dt = calculate_dt(dx, alpha)
    nt = int(d / dt)

    u, x, t = solver(a, b, d, f, g, h, nx, nt, alpha)
    
    # Plot the numerical solution
    plt.figure(figsize=(12, 8))
    plt.imshow(u, extent=[a, b, 0, d], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Solution u(x,t)')
    plt.title(f'Solution of PDE using {solver_name}')
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Time t')
    plt.show()
'''
def plot_pde_solution(solver, solver_name, a, b, d, f, g, h, alpha, nx):
    dx = (b - a) / nx
    dt = calculate_dt(dx, alpha)
    nt = int(d / dt)

    u, x, t = solver(a, b, d, f, g, h, nx, nt, alpha)
    
    # Plot the numerical solution in 2D
    plt.figure(figsize=(12, 8))
    plt.imshow(u, extent=[a, b, 0, d], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Solution u(x,t)')
    plt.title(f'Solution of PDE using {solver_name}')
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Time t')
    plt.show()
    
    # Prepare data for 3D plotting
    X, T = np.meshgrid(x, t)
    U = u
    
    # Plot the numerical solution in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, U, cmap='viridis')
    fig.colorbar(surf, ax=ax, label='Solution u(x,t)')
    
    ax.set_title(f'Solution of PDE using {solver_name}')
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Solution u(x,t)')
    
    plt.show()
    
def plot_pde_error(solver, solver_name, a, b, d, f, g, h, alpha, nx, exact_solution):
    dx = (b - a) / nx
    dt = calculate_dt(dx, alpha)
    nt = int(d / dt)

    u, x, t = solver(a, b, d, f, g, h, nx, nt, alpha)
    error_matrix = compute_pde_error(u, exact_solution, x, t)
    
    # Plot the error in 2D
    plt.figure(figsize=(12, 8))
    plt.imshow(error_matrix, extent=[a, b, 0, d], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Error |u(x,t) - exact|')
    plt.title(f'Error of PDE Solution using {solver_name}')
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Time t')
    plt.show()
    
    # Prepare data for 3D plotting
    X, T = np.meshgrid(x, t)
    E = error_matrix
    
    # Plot the error in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, E, cmap='viridis')
    fig.colorbar(surf, ax=ax, label='Error |u(x,t) - exact|')
    
    ax.set_title(f'Error of PDE Solution using {solver_name}')
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Error |u(x,t) - exact|')
    
    plt.show()
    
def plot_PDEBVP_solution(u, x, y, title):
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, u.T, cmap='plasma')
    fig.colorbar(surf, ax=ax, label='Solution u(x,y)')
    ax.set_title(title)
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Spatial coordinate y')
    ax.set_zlabel('Solution u(x,y)')
    plt.show()

def compute_PDEBVP_error(numerical_solution, exact_solution, x, y):
    exact_values = np.array([[exact_solution(xi, yi) for yi in y] for xi in x])
    error_matrix = np.abs(numerical_solution - exact_values)
    return error_matrix

def plot_PDEBVP_error(error_matrix, x, y, title):
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, error_matrix.T, cmap='viridis')
    fig.colorbar(surf, ax=ax, label='Error |u(x,y) - exact|')
    ax.set_title(title)
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Spatial coordinate y')
    ax.set_zlabel('Error |u(x,y) - exact|')
    plt.show()
    
def plot_vibrating_string_solution(u, x, t, title):
    fig = plt.figure(figsize=(12, 8))
    for n in range(0, len(t), len(t)//10):
        plt.plot(x, u[n, :], label=f't={t[n]:.2f}')
    plt.title(title)
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Solution u(x,t)')
    plt.legend()
    plt.show()

def compute_vibrating_string_error(numerical_solution, exact_solution, x, t):
    exact_values = np.array([[exact_solution(xi, ti) for xi in x] for ti in t])
    error_matrix = np.abs(numerical_solution - exact_values)
    return error_matrix   

def plot_vibrating_string_error(error_matrix, x, t, title):
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, error_matrix, cmap='viridis')
    fig.colorbar(surf, ax=ax, label='Error |u(x,t) - exact|')
    ax.set_title(title)
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Error |u(x,t) - exact|')
    plt.show()   

def animate_vibrating_string(u, x, t, title, save_path=None):
    fig, ax = plt.subplots()
    line, = ax.plot(x, u[0, :], color='k')
    ax.set_xlim(0, np.max(x))
    ax.set_ylim(np.min(u), np.max(u))
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Solution u(x,t)')
    ax.set_title(title)

    def update(frame):
        # Debugging statement to check if the update function is called
        print(f"Updating frame {frame}, time {t[frame]:.2f}")
        line.set_ydata(u[frame, :])
        ax.set_title(f'{title} at t={t[frame]:.2f}')
        return line,

    anim = FuncAnimation(fig, update, frames=len(t), interval=10, blit=False)
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=30)
    plt.show()
