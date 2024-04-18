"""
@author: Graeme Wiltrout
@advisor: Tiernan Fogarty, Spring 2024
Undergraduate Applications of Taylor Series
@school: Oregon Tech
"""

import numpy as np
import matplotlib.pyplot as plt
from NumericalDifferentiation import finite_difference
from NumericalIntegration import trapezoid, simpson
from ODE import runge_kutta_2_heun, runge_kutta_2_midpoint, runge_kutta_4, taylors_method_O3, eulers_method
from Utilities import data, plot_ode_convergence
from fractions import Fraction

def f(x):
    return 2 * np.sin(5*x) + 0.5 * x

def df(x):
    return 10 * np.cos(5*x) + 0.5

def d2f(x):
    return -50 * np.sin(5*x)

odes = ["TMO3", "RK2M", "RK2H", "RK4"]

odef = lambda t, y: -2 * y + np.exp(-t)
odedf = lambda t, y: np.exp(-t) - 2 * (-2 * y + np.exp(-t))
odeddf = lambda t, y: -np.exp(-t) - 2 * (np.exp(-t) - 2 * (-2 * y + np.exp(-t)))
odesol = lambda t: np.exp(-t)

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

def plot_finite_difference(exact_f, df, ddf, l, r, h, ftitle, htitle):
    data_points = data(exact_f, l, r, h)
    x_vals = [point[0] for point in data_points]
    y_vals = [point[1] for point in data_points]
    
    
    a = 2

    df_vals = [finite_difference(data_points, i, 1, a) for i in range(len(data_points))]
    ddf_vals = [finite_difference(data_points, i, 2, a) for i in range(len(data_points))]

    x_continuous = np.linspace(l, r, num=int((r - l) / (h / 10)) + 1)
    y_continuous = exact_f(x_continuous)
    df_continuous = df(x_continuous)
    ddf_continuous = ddf(x_continuous)

    # Configure x-axis as simplified fractions of pi
    x_ticks = np.arange(l, r + h, np.pi/4)
    x_labels = [simplify_pi_label(x) for x in x_ticks]

    # Define a function to create each plot
    def create_plot(x, y, y_cont=None, label1="Finite Difference Approximation", label2="Exact function", title=""):
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, color='red', label=label1)
        if y_cont is not None:
            plt.plot(x_continuous, y_cont, 'b-', label=label2)
        plt.xticks(x_ticks, x_labels)
        plt.title(f'{title}\n h = ${htitle}$ \n a = {a}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Generate each plot
    create_plot(x_vals, y_vals, title='Totally Completely Random Data Points')
    create_plot(x_vals, y_vals, y_continuous, title=f'{ftitle} and Data Points')
    create_plot(x_vals, df_vals, df_continuous, label2="Exact df/dx", title=f'Comparison of First Derivative for {ftitle}\n' + r'Exact: $10\cos(5x) + 0.5$')
    create_plot(x_vals, ddf_vals, ddf_continuous, label2="Exact ddf/dx", title=f'Comparison of Second Derivative for {ftitle}\n' + r'Exact: $-50\sin(5x)$')
    
    
def plot_integration(f, l, r, n, ftitle, fexact):
    # Generate points for plotting the function
    x_plot = np.linspace(l, r, 1000)
    y_plot = f(x_plot)

    # Data points for numerical integration
    x_data = np.linspace(l, r, n+1)
    y_data = f(x_data)

    # Configure x-axis as simplified fractions of pi
    x_ticks = np.linspace(l, r, num=8)  # Adjust num for more or fewer ticks
    x_labels = [simplify_pi_label(x) for x in x_ticks]
    
    h = 1/n

    # Trapezoid Method
    trap_integral = trapezoid(f, h, l, r)

    # Simpson Method
    simp_integral = simpson(f, l, r, n)

    # Plotting Trapezoid Method
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_plot, 'r-', label='Function $f(x)$')
    plt.plot(x_data, y_data, 'bo', label='Data points')
    for i in range(n):
        plt.fill([x_data[i], x_data[i], x_data[i+1], x_data[i+1]],
                 [0, y_data[i], y_data[i+1], 0], 'b', alpha=0.3, edgecolor='b')
    plt.xticks(x_ticks, x_labels)
    plt.title(f'Numerical Integration using Trapezoid Method for {ftitle}\n Integral Approximation: {trap_integral:.8f}\n Exact Integral: {fexact:.8f}\n {n} intervals')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Plotting Simpson's Method
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_plot, 'r-', label='Function $f(x)$')
    plt.plot(x_data, y_data, 'bo', label='Data points')
    for i in range(n//2):
        xi = x_data[2*i:2*i+3]
        yi = y_data[2*i:2*i+3]
        x_fit = np.linspace(xi[0], xi[-1], 100)
        y_fit = np.polyval(np.polyfit(xi, yi, 2), x_fit)
        plt.fill_between(x_fit, 0, y_fit, color='g', alpha=0.3)
    plt.xticks(x_ticks, x_labels)
    plt.title(f'Numerical Integration using Simpson\'s Method for {ftitle}\n Integral Approximation: {simp_integral:.8f}\n  Exact Integral: {fexact:.8f}\n{n} interval')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Return the integral values
    return {'Trapezoid Method': trap_integral, 'Simpson Method': simp_integral}

def plot_ode(f, df, ddf, y0, t0, tF, h, sol, ftitle, fexact):
    # Generating points for the exact solution using the sol function
    t_exact = np.linspace(t0, tF, 1000)
    y_exact = sol(t_exact)

    # Solving the ODE using various methods
    t_euler, y_euler = eulers_method(f, h, t0, tF, y0)
    t_heun, y_heun = runge_kutta_2_heun(f, h, t0, tF, y0)
    t_midpoint, y_midpoint = runge_kutta_2_midpoint(f, h, t0, tF, y0)
    t_rk4, y_rk4 = runge_kutta_4(f, h, t0, tF, y0)
    t_taylor, y_taylor = taylors_method_O3(f, df, ddf, h, t0, tF, y0)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(t_exact, y_exact, 'k-', label='Exact Solution')
    plt.plot(t_euler, y_euler, 'o--', label="Euler's Method", markersize=4)
    plt.plot(t_heun, y_heun, 's--', label="Runge-Kutta 2nd Order (Heun)", markersize=4)
    plt.plot(t_midpoint, y_midpoint, 'd--', label="Runge-Kutta 2nd Order (Midpoint)", markersize=4)
    plt.plot(t_rk4, y_rk4, '^--', label='Runge-Kutta 4th Order', markersize=4)
    plt.plot(t_taylor, y_taylor, 'x--', label="Taylor's Method 3rd Order", markersize=4)

    plt.title(f'Comparison of Numerical Methods for Solving ODEs:\n' f'$\\frac{{dy}}{{dt}} = {ftitle}$, ' f'$y(t) = {fexact}$ \n' f'$h = {h}$')
    plt.xlabel('Time, t')
    plt.ylabel('Solution, y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_ode2(f, df, ddf, y0, t0, tF, h, sol, ftitle, fexact):
    # Generating points for the exact solution using the sol function
    t_exact = np.linspace(t0, tF, 1000)
    y_exact = sol(t_exact)

    # Solving the ODE using various methods
    t_euler, y_euler = eulers_method(f, h, t0, tF, y0)
    t_heun, y_heun = runge_kutta_2_heun(f, h, t0, tF, y0)
    t_midpoint, y_midpoint = runge_kutta_2_midpoint(f, h, t0, tF, y0)
    t_rk4, y_rk4 = runge_kutta_4(f, h, t0, tF, y0)
    t_taylor, y_taylor = taylors_method_O3(f, df, ddf, h, t0, tF, y0)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(t_exact, y_exact, 'k-', label='Exact Solution')
    plt.plot(t_euler, y_euler, 'o--', label="Euler's Method", markersize=4)
    plt.plot(t_taylor, y_taylor, 'x--', label="Taylor's Method 3rd Order", markersize=4)

    plt.title(f'Comparison of Numerical Methods for Solving ODEs:\n' f'$\\frac{{dy}}{{dt}} = {ftitle}$, ' f'$y(t) = {fexact}$ \n' f'$h = {h}$')
    plt.xlabel('Time, t')
    plt.ylabel('Solution, y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
plt.close('all')    
plot_finite_difference(f, df, d2f, 0, 2*np.pi, np.pi/16, '2sin(5x) + 0.5x', '\\frac{\\pi}{16}')
plot_finite_difference(f, df, d2f, 0, 2*np.pi, np.pi/64, '2sin(5x) + 0.5x', '\\frac{\\pi}{64}')
plot_integration(f, 0, (2 * np.pi), 16, '2sin(5x) + 0.5x', 9.86960440)
plot_integration(f, 0, (2 * np.pi), 32, '2sin(5x) + 0.5x', 9.86960440)
plot_integration(f, 0, (2 * np.pi), 128, '2sin(5x) + 0.5x', 9.86960440)
plot_ode2(odef, odedf, odeddf, 1, 0, 5, 0.5, odesol, '-2y + e^{-t}', 'e^{-t}')
plot_ode2(odef, odedf, odeddf, 1, 0, 5, 0.01, odesol, '-2y + e^{-t}', 'e^{-t}')
plot_ode(odef, odedf, odeddf, 1, 0, 5, 0.5, odesol, '-2y + e^{-t}', 'e^{-t}')
plot_ode(odef, odedf, odeddf, 1, 0, 5, 0.01, odesol, '-2y + e^{-t}', 'e^{-t}')
plot_ode_convergence(odes, odef, (0,5), 1, '-2y + e^{-t}', odedf, odeddf)
