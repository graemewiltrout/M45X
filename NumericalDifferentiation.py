"""
Numerical Methods Package: Differentiation
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

from FiniteDiffCoefficients import Centered_Coefficients, Forward_Diff_Coefficients, Backward_Diff_Coefficients

"""
Finite Difference Methods Overview:
    Each of these functions expect to be called with a minimum of:
    f: function
    x: starting point
    
    If it is not given a step size, order, or accuracy,it will calculate like this:    
    h: Step size defaults to 1e-5
    o: Order defaults to 1st
    a: Coefficients are not included if not called
    
    
    to specify each of these pass them in with the order
    f, x, h, o, a
    
"""

def forward_finite_difference(f, x, h=1e-5, o=None, a=None):
    if o not in Forward_Diff_Coefficients:
        raise ValueError(f"Order '{o}' is not supported for forward finite difference. Available orders are: {list(Forward_Diff_Coefficients.keys())}.")

    if a not in Forward_Diff_Coefficients.get(o, {}):
        raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in forward finite difference. Available accuracies for order {o} are: {list(Forward_Diff_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Forward_Diff_Coefficients table
        coeffs = Forward_Diff_Coefficients[o][a]
        return sum(coeffs[i] * f(x + i*h) for i in range(len(coeffs))) / h**o
    else:
        # Basic forward finite difference (first derivative)
        return (f(x + h) - f(x)) / h

def centered_finite_difference(f, x, h=1e-5, o=None, a=None):
    if o not in Centered_Coefficients:
        raise ValueError(f"Order '{o}' is not supported for centered finite difference. Available orders are: {list(Centered_Coefficients.keys())}.")

    if a not in Centered_Coefficients.get(o, {}):
        raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in centered finite difference. Available accuracies for order {o} are: {list(Centered_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Centered_Coefficients table
        coeffs = Centered_Coefficients[o][a]
        return sum(coeffs[i] * f(x + (i-len(coeffs)//2)*h) for i in range(len(coeffs))) / h**o
    else:
        # Basic centered finite difference (first derivative)
        return (f(x + h) - f(x - h)) / (2*h)

def backward_finite_difference(f, x, h=1e-5, o=None, a=None):
    if o not in Backward_Diff_Coefficients:
        raise ValueError(f"Order '{o}' is not supported for backward finite difference. Available orders are: {list(Backward_Diff_Coefficients.keys())}.")

    if a not in Backward_Diff_Coefficients.get(o, {}):
        raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in backward finite difference. Available accuracies for order {o} are: {list(Backward_Diff_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Backward_Diff_Coefficients table
        coeffs = Backward_Diff_Coefficients[o][a]
        return sum(coeffs[i] * f(x - (len(coeffs) - i - 1)*h) for i in range(len(coeffs))) / h**o
    else:
        # Basic backward finite difference (first derivative)
        return (f(x) - f(x - h)) / h

def richardson_extrapolation(f, x, h=1e-5, method='centered', o=None, a=None):
    """
    Applies Richardson extrapolation using a specified finite difference method. Defaults to
    centered finite difference with a basic approach unless order and accuracy are specified.
    """
    # Function mapping for simplicity
    method_functions = {
        'forward': forward_finite_difference,
        'centered': centered_finite_difference,
        'backward': backward_finite_difference,
        'f': forward_finite_difference,
        'c': centered_finite_difference,
        'b': backward_finite_difference
    }
    
    # Validate method
    if method not in method_functions:
        raise ValueError(f"Unknown method '{method}'. Choose from 'forward', 'centered', 'backward'.")

    finite_diff_func = method_functions[method]

    # Check for the presence of optional parameters
    if o is not None and a is not None:
        # If order and accuracy are specified, use them
        D_h = finite_diff_func(f, x, h, o, a)
        D_h2 = finite_diff_func(f, x, h/2, o, a)
    else:
        # Use the basic version of the method
        D_h = finite_diff_func(f, x, h)
        D_h2 = finite_diff_func(f, x, h/2)

    # Apply Richardson extrapolation
    n = 2  # Assuming we're dealing with the first derivative for simplicity
    D_extrap = (2**n * D_h2 - D_h) / (2**n - 1)

    return D_extrap
