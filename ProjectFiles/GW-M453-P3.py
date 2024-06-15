"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods III 
T. Fogarty, Spring 2024
Project 3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from PDE import solve_vibrating_string
from PDE import solve_wave_2d

from Utilities import plot_vibrating_string_solution
from Utilities import plot_PDEBVP_solution

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
        # Save the animation as a GIF
        anim.save(save_path, writer=PillowWriter(fps=30))

    plt.show()
    return anim

def animate_2d_wave(u, x, y, t, title, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    
    ax.set_xlim(0, np.max(x))
    ax.set_ylim(0, np.max(y))
    ax.set_zlim(np.min(u), np.max(u))
    ax.set_xlabel('Spatial coordinate x')
    ax.set_ylabel('Spatial coordinate y')
    ax.set_zlabel('Solution u(x,y,t)')
    ax.set_title(title)
    
    # Initialize the plot with the first frame
    surf = ax.plot_surface(X, Y, u[0, :, :], cmap='viridis')
    
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y, u[frame, :, :], cmap='viridis')
        ax.set_xlim(0, np.max(x))
        ax.set_ylim(0, np.max(y))
        ax.set_zlim(np.min(u), np.max(u))
        ax.set_xlabel('Spatial coordinate x')
        ax.set_ylabel('Spatial coordinate y')
        ax.set_zlabel('Solution u(x,y,t)')
        ax.set_title(f'{title} at t={t[frame]:.2f}')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(t), interval=.1, blit=False)
    
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=30))
    
    plt.show()
    return anim


# Problem #1
f1 = lambda x: np.exp(-4 * x**2)
F1 = lambda x: 0

u, x, t = solve_vibrating_string(f1, F1, 8, 8, 2, 0.1, 0.01)
plot_PDEBVP_solution(u, t, x, "Vibrating String Solution")
plot_vibrating_string_solution(u, x, t, "Vibrating String Solution")
#anim1 = animate_vibrating_string(u, x, t, "Vibrating String Animation", r'C:\Users\graem\OneDrive\Desktop\M453\P3Q1.gif')

# Problem #2

def f2(x):
    return np.piecewise(x, 
                        [np.abs(x) <= 1, np.abs(x) > 1], 
                        [lambda x: -x**2 + 1, 0])
F2 = lambda x: 0

u, x, t = solve_vibrating_string(f2, F2, 8, 8, 2, 0.1, 0.01)
plot_PDEBVP_solution(u, t, x, "Vibrating String Solution")
plot_vibrating_string_solution(u, x, t, "Vibrating String Solution")
#anim2 = animate_vibrating_string(u, x, t, "Vibrating String Animation", r'C:\Users\graem\OneDrive\Desktop\M453\P3Q2.gif')


# Problem #3
def f3(x):
    return np.piecewise(x, 
                        [x < 2, (x >= 2) & (x <= 3), x > 3],
                        [lambda x: (x/2)**2, 
                         lambda x: (x - 3)**2, 
                         0])

def F3(x):
    return np.piecewise(x, 
                        [x < 2, (x >= 2) & (x <= 3), x > 3],
                        [lambda x: -1/2,
                         lambda x: -2*x + 6, 
                         0])

u, x, t = solve_vibrating_string(f3, F3, 10, 20, 1, 0.1, 0.01)
plot_PDEBVP_solution(u, t, x, "Vibrating String Solution")
plot_vibrating_string_solution(u, x, t, "Vibrating String Solution")
#anim3 = animate_vibrating_string(u, x, t, "Vibrating String Animation", r'C:\Users\graem\OneDrive\Desktop\M453\P3Q3.gif')


# Problem #4
def f4(x, y):
    return np.sin(2 * x) * np.sin(2 * y)

def F4(x, y):
    return np.zeros_like(x * y)

u, x, y, t = solve_wave_2d(f4, F4, np.pi, np.pi, 6, 1, 0.1, 0.1, 0.01)
#anim4 = animate_2d_wave(u, x, y, t, "2D Wave", r'C:\Users\graem\OneDrive\Desktop\M453\P3Q4.gif')

# Problem #5
def f5(x, y):
    return x * y * (1 - x) * (1 - y)

def F5(x, y):
    return np.zeros_like(x * y)

u, x, y, t = solve_wave_2d(f5, F5, 1, 1, 10, 2, 0.1, 0.1, 0.01)
#anim5 = animate_2d_wave(u, x, y, t, "2D Wave", r'C:\Users\graem\OneDrive\Desktop\M453\P3Q5.gif')