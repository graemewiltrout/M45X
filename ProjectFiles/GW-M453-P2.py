"""
@author: Graeme Wiltrout
MATH 451, Numerical Methods III 
T. Fogarty, Spring 2024
Project 2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


from PDE import CFD_5pt
from Utilities import plot_PDEBVP_solution
from Utilities import plot_PDEBVP_error
from Utilities import compute_PDEBVP_error

from PDE import solve_vibrating_string
from Utilities import plot_vibrating_string_solution
from Utilities import compute_vibrating_string_error
from Utilities import plot_vibrating_string_error
#from Utilities import animate_vibrating_string


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


def F1(x, y):
    return 0

def f1(y):
    return 4 * y

def g1(y):
    return 4 * y

def p1(x):
    return 0

def r1(x):
    return 4

def exact_solution1(x, y):
    return 4 * y * x

c2 = 2

def f2(x):
    return np.sin(x)

def F2(x):
    return np.zeros_like(x)

def exact_solution2(x, t):
    return np.cos(np.sqrt(c2) * t) * np.sin(x)

def exact_solution2b(x, y):
    return np.cos(3+y) * np.sin(x)

# Define the boundary and initial conditions
def F3(x, y):
    return 0

def f3(y):
    return 400  # Bottom edge at 400°F

def g3(y):
    return 250  # Top edge at 250°F

def p3(x):
    return 150  # Left edge at 150°F

def r3(x):
    return 150  # Right edge at 150°F


plt.close('all')

# Problem 1
u, x, y = CFD_5pt(F1, f1, g1, p1, r1, 0, 1, 0, 1, 0.1, 0.1)
plot_PDEBVP_solution(u, x, y, "Numerical Solution of PDE using Five-Point Stencil")
error_matrix = compute_PDEBVP_error(u, exact_solution1, x, y)
plot_PDEBVP_error(error_matrix, x, y, "Error in Numerical Solution of PDE using Five-Point Stencil")

# Problem 2
u, x, t = solve_vibrating_string(f2, F2, np.pi, 12, c2, 0.1, 0.01)
plot_vibrating_string_solution(u, x, t, "Numerical Solution of Vibrating String (c^2=2)")
error_matrix = compute_vibrating_string_error(u, exact_solution2, x, t)
plot_vibrating_string_error(error_matrix, x, t, "Error in Numerical Solution of Vibrating String (c^2=2)")
anim1 = animate_vibrating_string(u, x, t, "Animation of Vibrating String (c^2=2)", r'C:\Users\graem\OneDrive\Desktop\M453\P2Q2.gif')

# Problem 3
c2 = 9
u, x, t = solve_vibrating_string(f2, F2, np.pi, 12, c2, 0.1, 0.01)
plot_vibrating_string_solution(u, x, t, "Numerical Solution of Vibrating String (c^2=9)")
error_matrix = compute_vibrating_string_error(u, exact_solution2b, x, t)
plot_vibrating_string_error(error_matrix, x, t, "Error in Numerical Solution of Vibrating String (c^2=9)")
anim2 = animate_vibrating_string(u, x, t, "Animation of Vibrating String (c^2=9)", r'C:\Users\graem\OneDrive\Desktop\M453\P2Q3.gif')

# Problem 4
u, x, y = CFD_5pt(F3, f3, g3, p3, r3, 0, 10, 0, 6, 1, 1)
plot_PDEBVP_solution(u, x, y, "Numerical Solution of Temperature Distribution in Steel Plate")

# Problem 5
u, x, y = CFD_5pt(F3, f3, g3, p3, r3, 0, 10, 0, 6, 1/6, 1/6)
plot_PDEBVP_solution(u, x, y, "Numerical Solution of Temperature Distribution in Steel Plate")

# Problem 5b 1" Step for shits and gigs
u, x, y = CFD_5pt(F3, f3, g3, p3, r3, 0, 10, 0, 6, 1/12, 1/12)
plot_PDEBVP_solution(u, x, y, "Numerical Solution of Temperature Distribution in Steel Plate")
