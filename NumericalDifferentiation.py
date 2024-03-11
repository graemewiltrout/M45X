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

def forward_finite_difference(mode, f_or_data, x, h=1e-5, o=None, a=None):
    if mode not in ['f', 'd']:
        raise ValueError("Mode must be 'f' for function or 'd' for data.")
    
    if mode == 'f':
        def eval_point(x_i):
            return f_or_data(x_i)
    elif mode == 'd':
        # Ensure x is an integer index for data mode
        if not isinstance(x, float):
            raise ValueError("In data mode, x must be a float index.")
        def eval_point(i):
            return f_or_data[i][1]
    
    if o is not None:
        if o not in Forward_Diff_Coefficients:
            raise ValueError(f"Order '{o}' is not supported for forward finite difference. Available orders are: {list(Forward_Diff_Coefficients.keys())}.")
        
    if a is not None:
        if a not in Forward_Diff_Coefficients.get(o, {}):
            raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in forward finite difference. Available accuracies for order {o} are: {list(Forward_Diff_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Forward_Diff_Coefficients table
        coeffs = Forward_Diff_Coefficients[o][a]
        if mode == 'f':
            return sum(coeffs[i] * eval_point(x + i*h) for i in range(len(coeffs))) / h**o
        elif mode == 'd':
            # For data, ensure indices do not go out of bounds
            n = len(f_or_data)
            return sum(coeffs[i] * eval_point(min(x + i, n-1)) for i in range(len(coeffs))) / h**o
    else:
        # Basic forward finite difference (first derivative)
        if mode == 'f':
            return (eval_point(x + h) - eval_point(x)) / h
        elif mode == 'd':
            # For data, ensure index does not go out of bounds
            n = len(f_or_data)
            next_index = min(x + 1, n-1)
            return (eval_point(next_index) - eval_point(x)) / h

def centered_finite_difference(mode, f_or_data, x, h=1e-5, o=None, a=None):
    if mode not in ['f', 'd']:
        raise ValueError("Mode must be 'f' for function or 'd' for data.")
    
    if mode == 'f':
        def eval_point(x_i):
            return f_or_data(x_i)
    elif mode == 'd':
        # Ensure x is an integer index for data mode
        if not isinstance(x, float):
            raise ValueError("In data mode, x must be a float index.")
        def eval_point(i):
            # For data, ensure indices do not go out of bounds
            n = len(f_or_data)
            return f_or_data[max(0, min(i, n-1))][1]
    
    if o is not None:
        if o not in Forward_Diff_Coefficients:
            raise ValueError(f"Order '{o}' is not supported for centered finite difference. Available orders are: {list(Forward_Diff_Coefficients.keys())}.")
        
    if a is not None:
        if a not in Forward_Diff_Coefficients.get(o, {}):
            raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in centered finite difference. Available accuracies for order {o} are: {list(Forward_Diff_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Centered_Coefficients table
        coeffs = Centered_Coefficients[o][a]
        if mode == 'f':
            return sum(coeffs[i] * eval_point(x + (i-len(coeffs)//2)*h) for i in range(len(coeffs))) / h**o
        elif mode == 'd':
            # Calculate offset for centered difference in data mode
            offset = len(coeffs) // 2
            return sum(coeffs[i] * eval_point(x + (i-offset)) for i in range(len(coeffs))) / h**o
    else:
        # Basic centered finite difference (first derivative)
        if mode == 'f':
            return (eval_point(x + h) - eval_point(x - h)) / (2*h)
        elif mode == 'd':
            # For data, adjust for potential out-of-bounds at the start/end
            n = len(f_or_data)
            next_index = min(x + 1, n-1)
            prev_index = max(x - 1, 0)
            return (eval_point(next_index) - eval_point(prev_index)) / (2*h)

def backward_finite_difference(mode, f_or_data, x, h=1e-5, o=None, a=None):
    if mode not in ['f', 'd']:
        raise ValueError("Mode must be 'f' for function or 'd' for data.")
    
    if mode == 'f':
        def eval_point(x_i):
            return f_or_data(x_i)
    elif mode == 'd':
        # Ensure x is an integer index for data mode
        if not isinstance(x, float):
            raise ValueError("In data mode, x must be a float index.")
        def eval_point(i):
            # For data, ensure indices do not go out of bounds
            n = len(f_or_data)
            return f_or_data[max(0, min(i, n-1))][1]
    
    if o is not None:
        if o not in Forward_Diff_Coefficients:
            raise ValueError(f"Order '{o}' is not supported for backward finite difference. Available orders are: {list(Forward_Diff_Coefficients.keys())}.")
        
    if a is not None:
        if a not in Forward_Diff_Coefficients.get(o, {}):
            raise ValueError(f"Accuracy '{a}' is not available for order '{o}' in backward finite difference. Available accuracies for order {o} are: {list(Forward_Diff_Coefficients[o].keys())}.")

    if o is not None and a is not None:
        # Use coefficients from the Backward_Coefficients table (assuming existence similar to Forward and Centered tables)
        coeffs = Backward_Diff_Coefficients[o][a]
        if mode == 'f':
            return sum(coeffs[i] * eval_point(x - i*h) for i in range(len(coeffs))) / h**o
        elif mode == 'd':
            return sum(coeffs[i] * eval_point(x - i) for i in range(len(coeffs))) / h**o
    else:
        # Basic backward finite difference (first derivative)
        if mode == 'f':
            return (eval_point(x) - eval_point(x - h)) / h
        elif mode == 'd':
            prev_index = max(x - 1, 0)  # Adjust for potential out-of-bounds at the start
            return (eval_point(x) - eval_point(prev_index)) / h

def calculate_cfd_reach(o, a):
    """
    Calculate the reach for Centered Finite Difference based on order and accuracy.
    """
    coeffs = Centered_Coefficients.get(o, {}).get(a, [])
    if not coeffs:
        raise ValueError(f"Order {o} and accuracy {a} not found in Centered_Coefficients.")
    # Calculate the reach by finding the first and last non-zero coefficient
    non_zero_coeffs = [i for i, coeff in enumerate(coeffs) if coeff != 0]
    if not non_zero_coeffs:
        return 0  # No non-zero coefficients, should not happen with valid input
    left_reach = non_zero_coeffs[0]
    right_reach = len(coeffs) - non_zero_coeffs[-1] - 1
    # The maximum of left_reach and right_reach determines the needed reach
    return max(left_reach, right_reach)

def finite_difference(mode, f_or_data, x, h=1e-5, o=1, a=2):
    if mode == 'd':
        n = len(f_or_data)  # Total number of data points
        # Calculate the reach for centered difference based on order and accuracy
        reach = calculate_cfd_reach(o, a)
        
        if x < reach:
            # Not enough data points on the left, use forward difference
            return forward_finite_difference(mode, f_or_data, x, h, o, a)
        elif x > n - 1 - reach:
            # Not enough data points on the right, use backward difference
            return backward_finite_difference(mode, f_or_data, x, h, o, a)
        else:
            # Enough data points for centered difference
            return centered_finite_difference(mode, f_or_data, x, h, o, a)
    elif mode == 'f':
        # For functions, we assume infinite points; centered difference is preferred
        # Adjust if your function has a known limited domain
        return centered_finite_difference(mode, f_or_data, x, h, o, a)
    else:
        raise ValueError("Mode must be 'f' for function or 'd' for data.")


def richardson_extrapolation(mode, f_or_data, x, h=1e-5, method='centered', o=None, a=None):
    """
    Applies Richardson extrapolation using a specified finite difference method for either function or data. 
    Defaults to centered finite difference with a basic approach unless order and accuracy are specified.
    - mode: 'f' for function or 'd' for data.
    - f_or_data: function or data array based on the mode.
    - x: starting point (or index if mode is 'd').
    - h: Step size, defaults to 1e-5.
    - method: Specifies which finite difference method to use ('forward', 'centered', 'backward').
    - o: Order of the derivative.
    - a: Accuracy of the finite difference approximation.
    """
    if mode == 'd':
        # For data, directly use the finite_difference function with the specified method
        D_h = finite_difference(mode, f_or_data, x, h, o, a, method)
        D_h2 = finite_difference(mode, f_or_data, x, h/2, o, a, method)
    elif mode == 'f':
        # For functions, follow the original approach
        method_functions = {
            'forward': forward_finite_difference,
            'centered': centered_finite_difference,
            'backward': backward_finite_difference
        }
        
        if method not in method_functions:
            raise ValueError(f"Unknown method '{method}'. Choose from 'forward', 'centered', 'backward'.")

        finite_diff_func = method_functions[method]
        
        if o is not None and a is not None:
            D_h = finite_diff_func(mode, f_or_data, x, h, o, a)
            D_h2 = finite_diff_func(mode, f_or_data, x, h/2, o, a)
        else:
            D_h = finite_diff_func(mode, f_or_data, x, h)
            D_h2 = finite_diff_func(mode, f_or_data, x, h/2)
    else:
        raise ValueError("Mode must be 'f' for function or 'd' for data.")
    
    # Apply Richardson extrapolation
    n = 2 if o is None else o  # Assume n=2 if o is not specified for the extrapolation calculation
    D_extrap = (2**n * D_h2 - D_h) / (2**n - 1)

    return D_extrap