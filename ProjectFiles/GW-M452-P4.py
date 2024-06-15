"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods II 
T. Fogarty, Winter 2024
Project 4
"""

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt
from ODE import adams_bashforth_4 as AB4
from ODE import solve_bvp_centered_finite_diff as BCFD
from ODE import non_linear_shooting_method as nls

def arccot(x):
    return np.pi/2 - np.arctan(x)

def f1(t, v):
    g = 32
    k = 2.2
    return g - k * v

def f1e(t):
    g = 32
    k = 2.2
    return (g / k) * (1 - np.exp(-k * t))

def f2(t,v):
    g = 32
    k =  0.125
    m = 5
    return g - (k / m) * v**2

def f2e(t):
    g = 32
    k = 0.125
    m = 5
    return np.sqrt(g * m / k) * np.tanh(np.sqrt(g * k / m) * t)

def bvpe1(x):
    return -0.3 * np.cos(x) - 0.1 * np.sin(x)

def p1(x):
    return 2 * np.ones_like(x)

def q1(x):
    return np.ones_like(x)

def r1(x):
    return np.cos(x)

def bvpe2(x):
    return -2*x - 0.5*x*np.log(x) + np.log(x) + 2

def p2(x):
    return -1/x

def q2(x):
    return 1/x**2

def r2(x):
    return np.log(x)/x**2

def f4a(x, y, dy):
    return x * dy**2

def f4ae(x):
    return arccot(x/2)

def f4b(x, y, dy):
    return -np.sin(x) * np.exp(np.sin(x)) + y * np.cos(x)**2

def f4be(x):
    return np.exp(np.sin(x))

m0 = 10000  # Initial mass
k = 1/3000  # Thrust constant
g = 9.81    # Gravitational acceleration (m/s^2)
U = 1500 

def lander_system(t, Y):
    y, v = Y  # Y = [y, v] where y is position, v is velocity
    m = m0 - k*U*t  # Mass as a function of time
    dvdt = (U/m) - g  # The second-order ODE transformed to first-order
    return [v, dvdt]  # Return derivatives of y and v



def pltivp(f, interval, h, y0, exact=None, title=None):
    scipysol = solve_ivp(f, interval, [y0], dense_output=True)
    ab4sol = AB4(f, h, interval[0], interval[1], y0)
    
    t_points = np.linspace(interval[0], interval[1], 300)
    v_scipy = scipysol.sol(t_points)[0]
    
    plt.figure(figsize=(12,6))
    plt.plot(t_points, v_scipy, label='SciPy Solution', linestyle='-', color='blue')
    plt.plot(ab4sol[0], ab4sol[1], label='AB4 Solution', linestyle='--', marker='o', color='red', markersize=1)
    
    if exact is not None:
        v_exact = [exact(t) for t in t_points]
        plt.plot(t_points, v_exact, label='Exact Solution', linestyle=':') 
    
    if title is not None:
        plt.title('Comparison of solutions to: ' + title)
    else: 
        plt.title('Comparison of IVP Solutions')
    
    plt.xlabel('Time')
    plt.ylabel('Solution Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def pltbvp(p, q, r, a, b, A, B, N, exact=None, title=None):
    # Solve the BVP using BCFD method
    x_bcf, y_bcf = BCFD(p, q, r, a, b, A, B, N)
    
    # Define the differential equations for solve_bvp
    def fun(x, y):
        return np.vstack((y[1], p(x)*y[1] + q(x)*y[0] + r(x)))

    def bc(ya, yb):
        return np.array([ya[0] - A, yb[0] - B])

    # Solve the BVP using solve_bvp
    x_solve_bvp = np.linspace(a, b, N)
    y_guess = np.zeros((2, x_solve_bvp.size))
    sol = solve_bvp(fun, bc, x_solve_bvp, y_guess)
    
    plt.figure(figsize=(12,6))
    
    # Graph both solutions
    plt.plot(x_bcf, y_bcf, 'r-', label='BCFD Solution')
    plt.plot(sol.x, sol.y[0], 'b--', label='SciPy Solution')
    
    # Plot exact solution if provided
    if exact is not None:
        x_exact = np.linspace(a, b, 10*N)  # More points for smoother curve
        y_exact = exact(x_exact)
        plt.plot(x_exact, y_exact, 'g-', label='Exact Solution', linewidth=2)
    
    plot_title = 'Solution for: ' + title if title else 'BVP Solution Comparison'
    plt.title(plot_title)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    
def pltnls(f, a, b, A, B, N, exact=None, title=None, plot_start=None, plot_end=None):
    """
    Plots the solution from the non-linear shooting method against SciPy's solve_bvp
    and an optional exact solution over a custom plotting range.
    
    Parameters:
    - f: The function defining the differential equation y'' = f(x, y, y').
    - a, b: Interval endpoints for the boundary value problem.
    - A, B: Boundary values at a and b, respectively.
    - N: Number of intervals (N+1 points including endpoints) for the boundary value problem.
    - exact: An optional function for the exact solution. Takes an array of x values.
    - title: An optional title for the plot.
    - plot_start, plot_end: Optional start and end points for the plotting range. 
      If not specified, uses a, b as the plotting range.
    """
    if plot_start is None or plot_end is None:
        plot_start, plot_end = a, b

    plot_points = np.linspace(plot_start, plot_end, 10*N)

    # Non-linear shooting method solution
    x_nlsm, y_nlsm = nls(f, a, b, A, B, N)
    
    # Prepare for solve_bvp
    def fun(x, y):
        return np.vstack((y[1], f(x, y[0], y[1])))
    def bc(ya, yb):
        return np.array([ya[0] - A, yb[0] - B])
    x_lin = np.linspace(a, b, N+1)
    y_guess = np.zeros((2, x_lin.size))
    sol_bvp = solve_bvp(fun, bc, x_lin, y_guess)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_nlsm, y_nlsm, 'r-', label='Non-linear Shooting Method')
    plt.plot(sol_bvp.x, sol_bvp.y[0], 'b--', label='SciPy Solution')
    if exact is not None:
        y_exact = exact(plot_points)
        plt.plot(plot_points, y_exact, 'g-', label='Exact Solution', linewidth=2)
    plt.title(title if title else 'Comparison of BVP Solutions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.xlim(plot_start, plot_end)
    plt.show()
    
def p(x):
        return
def q(x):
    return
def r(x):
    return
    
    
def repgen():
    h = 1E-5
    N=50
    
    plt.close("all")
    pltivp(f1, (0,5), h, 0, f1e, r"$\frac{dv}{dt} = g - kv$")
    pltivp(f2, (0, 8), h, 0, f2e, r"$\frac{dv}{dt} = g - \frac{k}{m}v^2$")
    pltbvp(p1, q1, r1, 0, np.pi/2, -0.3, -0.1, N, bvpe1, title="3.a: y'' = y + 2y' + cos(x)")
    pltbvp(p2, q2, r2, 1, 2, 0, -2, N, bvpe2, "3.b: x^2y'' - xy' + y = ln(x)")
    pltnls(f4a, 0, 2, np.pi/2, np.pi/4, 50, f4ae, r"Solution for: $y'' = x(y')^2$, $y(0) = \frac{\pi}{2}$, $y(2) = \frac{\pi}{4}$")
    pltnls(f4b, 0, np.pi, 1, 1, 50, f4be, r"$y'' - y\cos^2(x) = -\sin(x)e^{\sin(x)}$", plot_start=1, plot_end=2)


    
repgen()