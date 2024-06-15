"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods II 
T. Fogarty, Winter 2024
Project 3
"""

from ODE import heuns_method, abm4, forward_euler, backwards_euler
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from Utilities import plot_ode_convergence
from Utilities import plot_odesys_convergence

odes = ['HM', 'ABM4']
odesys = ['FE', 'BE']

def f1(t, y):
    return (1 - t) * y

def f2(t, y):
    return t * y**2

def f3a(t, y):
    return np.cos(t) - np.sin(t) - y

def f3b(t, y):
    return 2 + (y - (2*t) + 3)**(1/2)

def f4_1(t, y1, y2):
    return 3*y1 - 37*y2

def f4_2(t, y1, y2):
    return 5*y1 - 39*y2

def plot(f, l, r, y0, h, title):
    tspc, yspc = heuns_method(f, l, r, y0, h)
    tsabm, ys1bm = abm4(f, h, l, r, y0)
    sol = solve_ivp(f, (l, r), [y0], dense_output=True)
    tref = np.linspace(l, r, 300)
    yref = sol.sol(tref)
    
    plt.figure(figsize=(12, 6))
    plt.plot(tref, yref.T, '-', label='SciPy Reference', color='red')  # Note the transpose on yref
    plt.plot(tspc, yspc, label='Trapezoid Predictor-Corrector', color='blue', linestyle='dashdot')
    plt.plot(tsabm, ys1bm, label='ABM4', color='purple', linestyle='dotted', linewidth='4')
    plt.title("Solution to ODE: " + title)
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.show()

def eplot(f1, f2, l, r, h, y1_0, y2_0):
    # Define a wrapper function for use with solve_ivp
    def ode_system(t, y):
        return [f1(t, y[0], y[1]), f2(t, y[0], y[1])]
    
    # Solve with forward Euler
    ts_fe, y1s_fe, y2s_fe = forward_euler(f1, f2, l, r, h, y1_0, y2_0)
    
    # Solve with backward Euler
    ts_be, y1s_be, y2s_be = backwards_euler(f1, f2, l, r, h, y1_0, y2_0)
    
    # Solve with built-in solver
    sol = solve_ivp(ode_system, [l, r], [y1_0, y2_0], method='RK45', t_eval=np.linspace(l, r, int((r-l)/h)+1))
    
    # Define a slice to select every 8th point for plotting
    slicedat = slice(None, None, 8)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # f1 Solutions
    plt.plot(ts_fe, y1s_fe, 'r-', label='Forward Euler y1', linewidth=2)
    plt.plot(ts_be, y1s_be, 'b-.', label='Backward Euler y1')
    plt.plot(sol.t[slicedat], sol.y[0][slicedat], 'ko:', label='Built-in Solver y1', markersize=5)
    
    # f2 Solutions
    plt.plot(ts_fe, y2s_fe, 'r-', label='Forward Euler y2', linewidth=2)
    plt.plot(ts_be, y2s_be, 'b-.', label='Backward Euler y2')
    plt.plot(sol.t[slicedat], sol.y[1][slicedat], 'ko:', label='Built-in Solver y2', markersize=5)
    
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.legend()
    plt.title('Comparison of Euler Methods with Built-in Solver')
    plt.show()

def genrep():
    h = 1E-5
    plt.close('all')
    plot(f1, 0, 3, 3, h, r'$dy = (1-t)y$')
    plot(f2, 0, 2, 2/5, h, r'$dy = ty^{2}$')
    plot(f3a, 0, 10, 2, h, r'$dy = cos(t) - sine(t) - y$')
    plot(f3b, 0, 1.5, 1, h, r'$dy = 2 + (y - 2t + 3)^{\frac{1}{2}}$') 
    eplot(f4_1, f4_2, 0, 5, 0.01, 16, -16)
    plot_ode_convergence(odes, f1, (0,3), 3, r'$dy = (1-t)y$')
    plot_ode_convergence(odes, f2, (0, 2), 2/5, r'$dy = ty^{2}$')
    plot_ode_convergence(odes, f3a, (0, 10), 2, r'$dy = cos(t) - sine(t) - y$')
    plot_ode_convergence(odes, f3b, (0, 1.5), 1, r'$dy = 2 + (y - 2t + 3)^{\frac{1}{2}}$')
    plot_odesys_convergence(odesys, f4_1, f4_2, (0,5), 16, -16, "BE and FE")
    

genrep()