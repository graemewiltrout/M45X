import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
from Utilities import method_selectors

odes = ["TMO3", "RK2M", "RK2H", "RK4", "E"]

def odef(t, y):
    return -2 * y + np.exp(-t)

def odedf(t, y):
    return 4 * y - 3 * np.exp(-t)

def odeddf(t, y):
    return -8 * y + 7 * np.exp(-t)

def odedddf(t, y):
    return 16 * y - 15 * np.exp(-t)

def odesol(t):
    return np.exp(-t)

def simplify_pi_label(x, base_pi=4):
    # Simplify the fraction and express as a fraction of π
    fraction = Fraction(x / np.pi).limit_denominator(base_pi)
    numerator = fraction.numerator
    denominator = fraction.denominator
    
    if numerator == 0:
        return "0"
    elif denominator == 1:
        return f"${numerator}\\pi$"
    elif numerator == 1:
        return f"$\\frac{{\\pi}}{{{denominator}}}$"
    else:
        return f"$\\frac{{{numerator}\\pi}}{{{denominator}}}$"

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
    hs = np.geomspace(1e-3, 1e-1, 100) #Makes an array of 10 logarithmically evenly spaced points from the smallest to the largest h
    errors = [] #Initialize a list to store the calculated errors

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
    
    return (hs, errors)

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
            (hs, errors) = ode_convergence_analysis(method, f, interval, y0, exact_solution, df, ddf)
        else: #Calls it normally for all others
            (hs, errors) = ode_convergence_analysis(method, f, interval, y0, exact_solution)

        plt.loglog(hs, errors, label=f"{method_abbr}") #Makes LogLog plot of hs and errors
    
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.gca().invert_xaxis()  # Reverse the x-axis direction
    plt.scatter(hs, errors)
    plt.legend()
    plt.grid(True, which="both", ls='--')
    plt.show()
    
    """
def ode_convergence_analysis(method, f, interval, y0, df=None, ddf=None):
    
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
    
    hs = np.geomspace(100, 1e-5, 1000)
    errors = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_error, method, f, df, ddf, h, interval, y0) for h in hs]
        for future in concurrent.futures.as_completed(futures):
            errors.append(future.result())

    # Estimate convergence rate
    rates = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(np.array(hs[:-1]) / np.array(hs[1:]))
    avg_rate = np.mean(rates)
    
    return (hs, errors), f"O(h^{avg_rate:.2f})"

def plot_ode_convergence(methods_list, f, interval, y0, title, df=None, ddf=None):
    plt.figure(figsize=(12, 8))
    
    for method_abbr in methods_list:
        method = method_selectors.get(method_abbr)
        if not method:
            print(f"Method {method_abbr} is not recognized.")
            continue

        if method_abbr == "TMO3" and (df is None or ddf is None):
            print("df and ddf must be provided for TMO3.")
            continue

        if method_abbr == "TMO3":
            (hs, errors), convergence_rate = ode_convergence_analysis(method, f, interval, y0, df, ddf)
        else:
            (hs, errors), convergence_rate = ode_convergence_analysis(method, f, interval, y0)

        plt.loglog(hs, errors, label=f"{method_abbr} - {convergence_rate}")
    
    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.legend()
    plt.gca().invert_xaxis()  # Reverse the x-axis direction
    plt.grid(True, which="both", ls='--')
    plt.show()

"""

plt.close("all")
plot_ode_convergence(odes, odef, (0,5), 1, '-2y + e^{-t}', odesol, odedf, odeddf)
