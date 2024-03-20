"""
Numerical Methods Package: Ordinary Differential Equations
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from MatOps import gaussian_elimination

def eulers_method(f, h, l, r, y0, t0=0):
    """
    Parameters
    ----------
    f : The function f(t,y)
    h : The step size
    l : Left bound
    r : Right bound
    y0: Initial value of Y
    t0: Initial value of t (default is 0)


    Returns
    ts: The t values as a list.
    ys: The y values as a list.
    """
    
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        y_next = y + h * f(t,y)
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
    return ts, ys

def forward_euler(f1, f2, l, r, h, y1_0, y2_0):
    """
    Solves a system of ODEs using the forward Euler method with general functions f1 and f2.
    
    Parameters:
    - f1, f2: The functions defining the ODEs (y1' = f1(t, y1, y2), y2' = f2(t, y1, y2)).
    - l, r: The left (start) and right (end) bounds of the interval.
    - h: The step size.
    - y1_0, y2_0: Initial conditions for y1 and y2.
    
    Returns:
    - ts: An array of time values.
    - y1s, y2s: Arrays of solution values for y1 and y2 at each time step.
    """
    N = int((r - l) / h)  # Number of steps
    ts = np.linspace(l, r, N+1)  # Time values
    y1s = np.zeros(N+1)  # Solution values for y1
    y2s = np.zeros(N+1)  # Solution values for y2
    y1s[0], y2s[0] = y1_0, y2_0  # Initial conditions
    
    for i in range(N):
        y1s[i+1] = y1s[i] + h * f1(ts[i], y1s[i], y2s[i])
        y2s[i+1] = y2s[i] + h * f2(ts[i], y1s[i], y2s[i])
        
    return ts, y1s, y2s

def backwards_euler(f1, f2, l, r, h, y1_0, y2_0):
    """
    Solves a system of ODEs using the backward Euler method with general functions f1 and f2.
    
    Parameters:
    - f1, f2: The functions defining the ODEs (y1' = f1(t, y1, y2), y2' = f2(t, y1, y2)).
    - l, r: The left (start) and right (end) bounds of the interval.
    - h: The step size.
    - y1_0, y2_0: Initial conditions for y1 and y2.
    
    Returns:
    - ts: An array of time values.
    - y1s, y2s: Arrays of solution values for y1 and y2 at each time step.
    """
    N = int((r - l) / h)  # Number of steps
    ts = np.linspace(l, r, N+1)  # Time values
    y1s = np.zeros(N+1)  # Solution values for y1
    y2s = np.zeros(N+1)  # Solution values for y2
    y1s[0], y2s[0] = y1_0, y2_0  # Initial conditions
    
    def backward_euler_step(t, y1, y2):
        """
        Solves the backward Euler step for y1 and y2.
        """
        def equations(vars):
            Y1, Y2 = vars
            return [Y1 - y1 - h * f1(t + h, Y1, Y2), Y2 - y2 - h * f2(t + h, Y1, Y2)]
        
        Y1_next, Y2_next = fsolve(equations, (y1, y2))
        return Y1_next, Y2_next
    
    for i in range(N):
        y1s[i+1], y2s[i+1] = backward_euler_step(ts[i], y1s[i], y2s[i])
        
    return ts, y1s, y2s


def runge_kutta_2_midpoint(f, h, l, r, y0, t0=0):
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        
        y_next = y + h * k2
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def runge_kutta_2_heun(f, h, l, r, y0, t0=0):
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        
        y_next = y + h * (k1 + k2) / 2
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def runge_kutta_4(f, h, l, r, y0):
    ts = [l]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t + h
        
        if t_next > r:
            break
            
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def adams_bashforth_2(f, h, l, r, y0):
    # Generate initial value using RK2
    _, ys_initial = runge_kutta_2_midpoint(f, h, l, l+h, y0)
    
    # Initialize ts and ys with the first two values
    ts = [l, l+h]
    ys = [y0, ys_initial[-1]]
    
    # AB2 formula application
    while ts[-1] < r:
        if ts[-1] + h > r:
            break
        t = ts[-1] + h
        y_next = ys[-1] + h * (1.5 * f(ts[-1], ys[-1]) - 0.5 * f(ts[-2], ys[-2]))
        ts.append(t)
        ys.append(y_next)
        
    return ts, ys

def adams_bashforth_4(f, h, l, r, y0):
    # Generate initial values using RK4
    ts_initial, ys_initial = runge_kutta_4(f, h, l, l+4*h, y0)
    
    # Initialize ts and ys with the first four values
    ts = ts_initial
    ys = ys_initial
    
    # Ensure enough initial values for AB4
    if len(ts) < 4:
        raise ValueError("Interval too short for initial RK4 steps.")
    
    # AB4 formula application
    while ts[-1] < r:
        if ts[-1] + h > r:
            break
        t = ts[-1] + h
        y_next = ys[-1] + h/24 * (55*f(ts[-1], ys[-1]) - 59*f(ts[-2], ys[-2]) + 37*f(ts[-3], ys[-3]) - 9*f(ts[-4], ys[-4]))
        ts.append(t)
        ys.append(y_next)
        
    return ts, ys

def taylors_method_O3(f, df, ddf, h, l, r, y0):
    ts = [l]
    ys = [y0]
    while ts[-1] + h <= r:
        t = ts[-1]
        y = ys[-1]
        # Calculating next y value using up to the third derivative
        y_next = y + h * f(t, y) + (h**2 / 2) * df(t, y) + (h**3 / 6) * ddf(t, y)
        ts.append(t + h)
        ys.append(y_next)
    return ts, ys

def heuns_method(f, a, b, y0, h):
    """
    Solves an ODE using Heun's method, which is a type of predictor-corrector
    method that uses Euler's method for the prediction and the trapezoidal
    rule for the correction.

    Parameters:
    f : callable
        The derivative function f(t, y).
    a, b : float
        The interval of integration [a, b].
    y0 : float
        The initial value y(a).
    h : float
        The step size.

    Returns:
    ts : list
        The list of t values.
    ys : list
        The list of approximated y values.
    """
    ts = [a]  
    ys = [y0]

    while ts[-1] < b:
        t_prev, y_prev = ts[-1], ys[-1]
        # Predictor step using Euler's method
        y_pred = y_prev + h * f(t_prev, y_prev)
        t_next = t_prev + h
        # Corrector step using the trapezoidal rule
        y_corr = y_prev + h * 0.5 * (f(t_prev, y_prev) + f(t_next, y_pred))

        ts.append(t_next)
        ys.append(y_corr)

    return ts, ys


def abm4(f, h, l, r, y0):
    """
    AB4 Prediction w/ AM4 Correction
    """

    ts, ys = adams_bashforth_4(f, h, l, r, y0)[:2]

    if len(ts) < 4:
        raise ValueError("Initial interval too short for AB4 steps.")

    while ts[-1] < r:
        t_next = ts[-1] + h
        if t_next > r:
            break

        # AM4-like Corrector step
        y_pred = ys[-1] + h/24 * (55*f(ts[-1], ys[-1]) - 59*f(ts[-2], ys[-2]) + 37*f(ts[-3], ys[-3]) - 9*f(ts[-4], ys[-4]))
        y_corr = ys[-1] + h/24 * (9*f(t_next, y_pred) + 19*f(ts[-1], ys[-1]) - 5*f(ts[-2], ys[-2]) + f(ts[-3], ys[-3]))
        
        ts.append(t_next)
        ys.append(y_corr)
    
    return ts, ys

def solve_bvp_centered_finite_diff(p, q, r, a, b, A, B, N):
    """
    Solves a second-order boundary value problem using centered finite differences
    and Gaussian elimination for solving the resulting linear system.
    
    Parameters:
    p, q, r: Functions defining the BVP y'' + p(x)*y' + q(x)*y = r(x).
    a, b: The spatial domain [a, b].
    A, B: Boundary values at a and b, respectively.
    N: Number of intervals (N+1 points including the boundaries).
    
    Returns:
    x: The spatial points.
    y: The solution at the spatial points.
    """
    h = (b - a) / N  # Step size
    x = [a + i*h for i in range(N+1)]  # Spatial points
    
    # Initialize the augmented matrix for the linear system
    aug_matrix = [[0 for _ in range(N+2)] for _ in range(N+1)]
    
    # Apply boundary conditions
    aug_matrix[0][0] = 1
    aug_matrix[0][-1] = A
    aug_matrix[N][N] = 1
    aug_matrix[N][-1] = B
    
    # Fill the augmented matrix for the interior points
    for i in range(1, N):
        xi = x[i]
        aug_matrix[i][i-1] = 1/h**2 - p(xi)/(2*h)  # Coefficient for y_{i-1}
        aug_matrix[i][i] = -2/h**2 + q(xi)         # Coefficient for y_i
        aug_matrix[i][i+1] = 1/h**2 + p(xi)/(2*h)  # Coefficient for y_{i+1}
        aug_matrix[i][-1] = r(xi)                  # r(x_i)
    
    # Solve the linear system using Gaussian elimination
    aug_matrix_np = np.array(aug_matrix)
    solution = gaussian_elimination(N+1, aug_matrix_np)
    
    return x, solution

def non_linear_shooting_method(f, a, b, A, B, N, tol=1e-5, max_iter=100):
    """
    Solve a second-order BVP using the non-linear shooting method.

    Parameters:
    - f: The function defining the differential equation y'' = f(x, y, y').
    - a, b: The interval endpoints.
    - A, B: The boundary values at a and b, respectively.
    - N: The number of intervals (N+1 points including endpoints).
    - tol: Tolerance for the root-finding procedure.
    - max_iter: Maximum number of iterations for the root-finding procedure.

    Returns:
    - x: Array of x values.
    - y: Array of y values.
    """
    # Define the system of first-order ODEs for use in solve_ivp
    def system(x, Y):
        y, dy = Y
        return [dy, f(x, y, dy)]

    def shoot(guess):
        """
        Solve the IVP for a given guess of y'(a) and return y(b).
        """
        sol = solve_ivp(system, [a, b], [A, guess], t_eval=np.linspace(a, b, N+1))
        return sol.y[0, -1]  # Return y(b)

    # Initial guesses for y'(a)
    guess_1 = 0.0
    guess_2 = 1.0

    # Use the secant method to adjust the guesses
    for _ in range(max_iter):
        f1 = shoot(guess_1) - B
        f2 = shoot(guess_2) - B
        
        if abs(f2 - f1) < tol:
            break

        # Secant method update
        guess_new = guess_2 - f2 * (guess_2 - guess_1) / (f2 - f1)
        
        if abs(guess_new - guess_2) < tol:
            break
        
        guess_1, guess_2 = guess_2, guess_new

    # Once the correct guess is found, solve the BVP with the final guess
    final_guess = guess_2
    x = np.linspace(a, b, N+1)
    sol = solve_ivp(system, [a, b], [A, final_guess], t_eval=x)

    return x, sol.y[0]
