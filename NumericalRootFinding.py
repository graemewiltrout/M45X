"""
Numerical Methods Package: Root Finding
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

"""
Root Finding Methods Overview:
    These functions expect you to specify the following for most methods:
        f = the function whose root you're trying to find
        tol = tolerance for the root's accuracy
    Additional specifics are noted per function.
"""

def bisection(f, a, b, tol):
    """
    Bisection Method: Splitting the interval [a, b] until the root is pinpointed within the given tolerance.
    Requires: function f, interval [a, b] where f(a)*f(b) < 0, and tolerance tol.
    """
    if f(a) * f(b) >= 0:
        return "Function does not have opposite signs at a and b."
    
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint  # The midpoint is a root
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2

def unified_newtons_method(f, df, x0, tol, m=1, max_iter=100):
    """
    Parameters:
    f (function): The function for which the root is sought.
    df (function): The derivative of the function f.
    x0 (float): Initial guess for the root.
    tol (float): Tolerance for the root's accuracy.
    m (int, optional): The multiplicity of the root. Defaults to 1, behaving as the standard Newton's method.
    max_iter (int, optional): Maximum number of iterations. Defaults to 100.
    """
    xn = x0
    for _ in range(max_iter):
        fxn = f(xn)
        dfxn = df(xn)
        if abs(fxn) < tol:
            return xn  # Root found within tolerance
        if dfxn == 0:
            return "Derivative is zero. No solution found."  # Prevent division by zero
        xn = xn - m * fxn / dfxn  # Modified Newton's step
        if abs(f(xn)) < tol:
            return xn  # Root found within tolerance after update
    return "Maximum iterations reached. No solution found."

def secant_method(f, x0, x1, tol, max_iter=100):
    """
    Secant Method: Using two initial guesses to approximate the derivative and iteratively find the root.
    Requires: function f, initial guesses x0 and x1, tolerance tol, and optionally max iterations.
    """
    for n in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            return "Division by zero. No solution found."
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return "Maximum iterations reached. No solution found."
