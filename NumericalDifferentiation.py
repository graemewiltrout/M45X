"""
Numerical Methods Package: Differentiation
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

from FiniteDiffCoefficients import Centered_Coefficients, Forward_Diff_Coefficients, Backward_Diff_Coefficients

def forward_finite_difference(f, x, h, o, a):
    """
    Approximates the derivative of a function using forward finite differences.

    Parameters:
    - f: function, the function to differentiate.
    - x: float, the point at which to approximate the derivative.
    - h: float, the step size.
    - o: int, the order of the derivative.
    - a: int, the accuracy order of the approximation.

    Returns:
    - float, the approximated derivative.
    """

    if o not in Forward_Diff_Coefficients or a not in Forward_Diff_Coefficients[o]:
        raise ValueError("Invalid order or accuracy for forward finite difference.")
        
    coefficients = Forward_Diff_Coefficients[o][a]
    derivative_approximation = sum(coeff * f(x + i * h) for i, coeff in enumerate(coefficients))
    return derivative_approximation / h**o

def centered_finite_difference(f, x, h, o, a):
    """
    Approximates the derivative of a function using centered finite differences.

    Parameters:
    - f: function, the function to differentiate.
    - x: float, the point at which to approximate the derivative.
    - h: float, the step size.
    - o: int, the order of the derivative.
    - a: int, the accuracy order of the approximation.

    Returns:
    - float, the approximated derivative.
    """

    if o not in Centered_Coefficients or a not in Centered_Coefficients[o]:
        raise ValueError("Invalid order or accuracy for centered finite difference.")
        
    coefficients = Centered_Coefficients[o][a]
    midpoint = len(coefficients) // 2
    derivative_approximation = sum(coeff * f(x + (i - midpoint) * h) for i, coeff in enumerate(coefficients))
    return derivative_approximation / h**o

def backward_finite_difference(f, x, h, o, a):
    """
    Approximates the derivative of a function using backward finite differences.

    Parameters:
    - f: function, the function to differentiate.
    - x: float, the point at which to approximate the derivative.
    - h: float, the step size.
    - o: int, the order of the derivative.
    - a: int, the accuracy order of the approximation.

    Returns:
    - float, the approximated derivative.
    """
    # Ensure the order and accuracy are available
    if o not in Backward_Diff_Coefficients or a not in Backward_Diff_Coefficients[o]:
        raise ValueError("Invalid order or accuracy for backward finite difference.")
        
    coefficients = Backward_Diff_Coefficients[o][a]
    # Accumulate the weighted sum of function evaluations
    derivative_approximation = sum(coeff * f(x - i * h) for i, coeff in enumerate(coefficients))
    return derivative_approximation / h**o

def finite_difference(data, i, o, a, hold_a='n'):
    n = len(data)
    h = data[1][0] - data[0][0]  # Assume uniform spacing
    derivative = 0
    
    if i < a // 2:  # Use forward differences near the beginning
        if hold_a == 'n' and (a + 2) in Forward_Diff_Coefficients[o]:
            # Check if a higher accuracy is available, if not, stick to the current level
            a += 2
        coeffs = Forward_Diff_Coefficients[o][a]
        for j in range(len(coeffs)):
            derivative += coeffs[j] * data[i + j][1]
    elif i > n - 1 - a // 2:  # Use backward differences near the end
        if hold_a == 'n' and (a + 2) in Backward_Diff_Coefficients[o]:
            # Check if a higher accuracy is available, if not, stick to the current level
            a += 2
        coeffs = Backward_Diff_Coefficients[o][a]
        for j in range(len(coeffs)):
            derivative += coeffs[j] * data[i - j][1]
    else:  # Use centered differences in the middle
        coeffs = Centered_Coefficients[o][a]
        for j in range(len(coeffs)):
            derivative += coeffs[j] * data[i + j - a // 2][1]
    
    return derivative / h**o


def richardson_extrapolation(f, x, h=0.001, o=1, a=2):
    """
    Enhances the accuracy of the derivative calculation at a point using Richardson extrapolation.

    Parameters:
    - f: function, the function to differentiate.
    - x: float, the point at which to approximate the derivative.
    - h: float, the step size used in the approximation.
    - o: int, the order of the derivative.
    - a: int, the accuracy order of the approximation.

    Returns:
    - float, the approximated derivative enhanced by Richardson extrapolation.
    """
    # Calculate the centered finite difference using h
    D_h = centered_finite_difference(f, x, h, o, a)
    # Calculate the centered finite difference using h/2
    D_h2 = centered_finite_difference(f, x, h/2, o, a)

    # Apply Richardson extrapolation to enhance accuracy
    D = (4 * D_h2 - D_h) / 3

    return D
