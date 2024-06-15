"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods III 
T. Fogarty, Spring 2024
Project 4
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from PDE import solve_vibrating_string
from PDE import solve_wave_2d
from PDE import solve_wave_2d_polar

from Utilities import plot_vibrating_string_solution
from Utilities import plot_PDEBVP_solution

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

def animate_radial_wave(u, r, theta, t, title, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    ax.set_xlim(-np.max(r), np.max(r))
    ax.set_ylim(-np.max(r), np.max(r))
    ax.set_zlim(np.min(u), np.max(u))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Solution u(r,θ,t)')
    ax.set_title(title)
    
    # Initialize the plot with the first frame
    surf = ax.plot_surface(X, Y, u[0, :, :], cmap='viridis')
    
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y, u[frame, :, :], cmap='viridis')
        ax.set_xlim(-np.max(r), np.max(r))
        ax.set_ylim(-np.max(r), np.max(r))
        ax.set_zlim(np.min(u), np.max(u))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Solution u(r,θ,t)')
        ax.set_title(f'{title} at t={t[frame]:.2f}')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(t), interval=100, blit=False)
    
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=30))
    
    plt.show()
    return anim


#Problem #1

def f1(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.where(r<3, np.exp(-4 * r**2), 0)

def F1(x, y):
    return np.zeros_like(x)


u, x, y, t = solve_wave_2d(f1, F1, 3.2, 3.2, 6, 4, 0.1, 0.1, 0.01)
anim1 = animate_2d_wave(u, x, y, t, "Q1 2D Wave", r'C:\Users\graem\OneDrive\Desktop\M453_Graphics\P4Q1.gif')

'''

#Problem #2

f2 = lambda r, theta: np.exp(-4 * r**2)
F2 = lambda r, theta: np.zeros_like(r)

u, r, theta, t = solve_wave_2d_polar(f2, F2, 3, (2 * np.pi), 6, 4, 0.01, (np.pi/12), 0.001)
anim2 = animate_radial_wave(u, r, theta, t, "Q2 2D Wave Radial")
'''