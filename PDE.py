"""
Numerical Methods Package: Partial Differential Equations
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np
import LinearAlgebra as M341

def FTCSE(a, b, d, f, g, h, nx, nt, alpha):
    """
    Solves the one-dimensional parabolic PDE using the forward-time centered-space finite difference method.
    
    Parameters:
    a, b: float
        The spatial domain [a, b].
    d: float
        The time interval [0, d].
    f: function
        The initial value function u(x, 0) = f(x).
    g: function
        The boundary condition at x = a, u(a, t) = g(t).
    h: function
        The boundary condition at x = b, u(b, t) = h(t).
    nx: int
        The number of spatial steps.
    nt: int
        The number of time steps.
    alpha: float
        The thermal diffusivity or similar coefficient in the PDE.
    
    Returns:
    u: 2D numpy array
        The solution matrix with shape (nt+1, nx+1).
    x: 1D numpy array
        The spatial grid points.
    t: 1D numpy array
        The time grid points.
    """
    
    # Spatial grid
    x = np.linspace(a, b, nx+1)
    dx = (b - a) / nx
    
    # Time grid
    t = np.linspace(0, d, nt+1)
    dt = d / nt
    
    # Stability condition
    assert alpha * dt / dx**2 <= 0.5, "Stability condition not met, reduce dt or increase dx."
    
    # Initialize solution matrix
    u = np.zeros((nt+1, nx+1))
    
    # Initial condition
    for i in range(nx+1):
        u[0, i] = f(x[i])
    
    # Boundary conditions
    for n in range(nt+1):
        u[n, 0] = g(t[n])
        u[n, nx] = h(t[n])
    
    # Time-stepping loop
    for n in range(0, nt):
        for i in range(1, nx):
            u[n+1, i] = u[n, i] + alpha * dt / dx**2 * (u[n, i+1] - 2 * u[n, i] + u[n, i-1])
    
    return u, x, t

def BTCSI(a, b, d, f, g, h, nx, nt, alpha):
    """
    Solves the one-dimensional parabolic PDE using the backward-time centered-space finite difference method.
    
    Parameters:
    a, b: float
        The spatial domain [a, b].
    d: float
        The time interval [0, d].
    f: function
        The initial value function u(x, 0) = f(x).
    g: function
        The boundary condition at x = a, u(a, t) = g(t).
    h: function
        The boundary condition at x = b, u(b, t) = h(t).
    nx: int
        The number of spatial steps.
    nt: int
        The number of time steps.
    alpha: float
        The thermal diffusivity or similar coefficient in the PDE.
    
    Returns:
    u: 2D numpy array
        The solution matrix with shape (nt+1, nx+1).
    x: 1D numpy array
        The spatial grid points.
    t: 1D numpy array
        The time grid points.
    """
    
    # Spatial grid
    x = np.linspace(a, b, nx+1)
    dx = (b - a) / nx
    
    # Time grid
    t = np.linspace(0, d, nt+1)
    dt = d / nt
    
    # Coefficient
    r = alpha * dt / dx**2
    
    # Initialize solution matrix
    u = np.zeros((nt+1, nx+1))
    
    # Initial condition
    for i in range(nx+1):
        u[0, i] = f(x[i])
    
    # Boundary conditions
    for n in range(nt+1):
        u[n, 0] = g(t[n])
        u[n, nx] = h(t[n])
    
    # Set up the coefficient matrix for the implicit method
    A = np.zeros((nx-1, nx-1))
    np.fill_diagonal(A, 1 + 2*r)
    np.fill_diagonal(A[:-1, 1:], -r)
    np.fill_diagonal(A[1:, :-1], -r)
    
    # Time-stepping loop
    for n in range(0, nt):
        b = u[n, 1:-1]
        b[0] += r * u[n+1, 0]  # Include boundary condition g(t) at x = a
        b[-1] += r * u[n+1, nx]  # Include boundary condition h(t) at x = b
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        u[n+1, 1:-1] = M341.gaussian_elimination_spp(nx-1, augmented_matrix)
    
    return u, x, t

def CrankNicholson(a, b, d, f, g, h, nx, nt, alpha):
    """
    Solves the one-dimensional parabolic PDE using the Crank-Nicholson method.
    
    Parameters:
    a, b: float
        The spatial domain [a, b].
    d: float
        The time interval [0, d].
    f: function
        The initial value function u(x, 0) = f(x).
    g: function
        The boundary condition at x = a, u(a, t) = g(t).
    h: function
        The boundary condition at x = b, u(b, t) = h(t).
    nx: int
        The number of spatial steps.
    nt: int
        The number of time steps.
    alpha: float
        The thermal diffusivity or similar coefficient in the PDE.

    Returns:
    u: 2D numpy array
        The solution matrix with shape (nt+1, nx+1).
    x: 1D numpy array
        The spatial grid points.
    t: 1D numpy array
        The time grid points.
    """
    
    # Spatial grid
    x = np.linspace(a, b, nx+1)
    dx = (b - a) / nx
    
    # Time grid
    t = np.linspace(0, d, nt+1)
    dt = d / nt
    
    # Coefficient
    r = alpha * dt / (2 * dx**2)
    
    # Initialize solution matrix
    u = np.zeros((nt+1, nx+1))
    
    # Initial condition
    for i in range(nx+1):
        u[0, i] = f(x[i])
    
    # Boundary conditions
    for n in range(nt+1):
        u[n, 0] = g(t[n])
        u[n, nx] = h(t[n])
    
    # Set up the coefficient matrices
    A = np.zeros((nx-1, nx-1))
    B = np.zeros((nx-1, nx-1))
    
    np.fill_diagonal(A, 1 + 2*r)
    np.fill_diagonal(A[:-1, 1:], -r)
    np.fill_diagonal(A[1:, :-1], -r)
    
    np.fill_diagonal(B, 1 - 2*r)
    np.fill_diagonal(B[:-1, 1:], r)
    np.fill_diagonal(B[1:, :-1], r)
    
    # Time-stepping loop
    for n in range(0, nt):
        b = np.dot(B, u[n, 1:-1])
        b[0] += r * (u[n, 0] + u[n+1, 0])  # Include boundary condition g(t) at x = a
        b[-1] += r * (u[n, nx] + u[n+1, nx])  # Include boundary condition h(t) at x = b
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        u[n+1, 1:-1] = M341.gaussian_elimination_spp(nx-1, augmented_matrix)
    
    return u, x, t

def CFD_5pt(F, f, g, p, r, a, b, c, d, h, k):
    # Grid points
    x = np.arange(a, b + h, h)
    y = np.arange(c, d + k, k)
    nx, ny = len(x), len(y)

    # Number of unknowns
    N = (nx - 2) * (ny - 2)

    # Initialize matrix and RHS vector for the linear system
    A = np.zeros((N, N))
    B = np.zeros(N)

    # Helper function to map (i, j) to linear index
    def index(i, j):
        return (j - 1) * (nx - 2) + (i - 1)

    # Fill the matrix A and vector B
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            
            idx = index(i, j)
            A[idx, idx] = -2 * (1/h**2 + 1/k**2)

            if i > 1:
                A[idx, index(i-1, j)] = 1/h**2
            else:
                B[idx] -= f(y[j]) / h**2

            if i < nx-2:
                A[idx, index(i+1, j)] = 1/h**2
            else:
                B[idx] -= g(y[j]) / h**2

            if j > 1:
                A[idx, index(i, j-1)] = 1/k**2
            else:
                B[idx] -= p(x[i]) / k**2

            if j < ny-2:
                A[idx, index(i, j+1)] = 1/k**2
            else:
                B[idx] -= r(x[i]) / k**2

            # Add F(x, y) term
            B[idx] -= F(x[i], y[j])

    # Solve the linear system using NumPy's linear solver
    solution_vector = np.linalg.solve(A, B)

    # Reshape solution vector back to grid
    u = np.zeros((nx, ny))
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            u[i, j] = solution_vector[index(i, j)]

    # Apply boundary conditions
    u[0, :] = f(y)    # u(a, y) = f(y)
    u[-1, :] = g(y)   # u(b, y) = g(y)
    u[:, 0] = p(x)    # u(x, c) = p(x)
    u[:, -1] = r(x)   # u(x, d) = r(x)

    return u, x, y

def solve_vibrating_string(f, F, a, T, c2, dx, dt):
    # Discretize the space and time domains
    x = np.arange(0, a + dx, dx)
    t = np.arange(0, T + dt, dt)
    nx, nt = len(x), len(t)

    # Stability condition
    r = c2 * (dt / dx)**2
    assert r <= 1, "Stability condition not met, reduce dt or increase dx."

    # Initialize solution matrix
    u = np.zeros((nt, nx))

    # Apply initial conditions
    u[0, :] = f(x)
    u[1, 1:-1] = f(x[1:-1]) + dt * F(x[1:-1]) + 0.5 * r * (f(x[2:]) - 2 * f(x[1:-1]) + f(x[:-2]))

    # Time-stepping loop
    for n in range(1, nt - 1):
        u[n+1, 1:-1] = 2 * (1 - r) * u[n, 1:-1] - u[n-1, 1:-1] + r * (u[n, 2:] + u[n, :-2])

    return u, x, t

def solve_wave_2d(f, F, a, b, T, c2, dx, dy, dt):
    # Discretize the space and time domains
    x = np.arange(0, a + dx, dx)
    y = np.arange(0, b + dy, dy)
    t = np.arange(0, T + dt, dt)
    nx, ny, nt = len(x), len(y), len(t)

    # Stability condition (CFL condition)
    r = c2 * dt**2 * (1/dx**2 + 1/dy**2)
    assert r <= 1, "Stability condition not met, reduce dt, increase dx or dy."

    # Initialize solution matrices
    u = np.zeros((nt, nx, ny))
    
    # Apply initial conditions
    u[0, :, :] = f(x[:, None], y[None, :])
    u[1, 1:-1, 1:-1] = (f(x[1:-1, None], y[None, 1:-1]) +
                        dt * F(x[1:-1, None], y[None, 1:-1]) +
                        0.5 * r * (
                            f(x[2:, None], y[None, 1:-1]) + 
                            f(x[:-2, None], y[None, 1:-1]) +
                            f(x[1:-1, None], y[None, 2:]) +
                            f(x[1:-1, None], y[None, :-2]) -
                            4 * f(x[1:-1, None], y[None, 1:-1])
                        ))

    # Time-stepping loop
    for n in range(1, nt-1):
        u[n+1, 1:-1, 1:-1] = (2 * (1 - 2 * r) * u[n, 1:-1, 1:-1] -
                              u[n-1, 1:-1, 1:-1] +
                              r * (u[n, 2:, 1:-1] + u[n, :-2, 1:-1] + 
                                   u[n, 1:-1, 2:] + u[n, 1:-1, :-2]))

    return u, x, y, t