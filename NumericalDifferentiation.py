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
    # Assuming Forward_Diff_Coefficients is correctly defined as per your provided coefficients
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
    # Assuming Centered_Coefficients is correctly defined as per your provided coefficients
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

def finite_difference(data, i, o, a):
    """
    Approximates the derivative at a specific index within a discrete dataset using
    forward, backward, or centered finite differences based on the position of the index.

    Parameters:
    - data: list of tuples, the dataset as (x, y) pairs.
    - i: int, the index of the data point at which to approximate the derivative.
    - o: int, the order of the derivative.
    - a: int, the accuracy of the approximation.

    Returns:
    - float, the approximated derivative at the specified index.
    """
    # Ensure index is within the dataset bounds
    if i < 0 or i >= len(data):
        raise ValueError("Index out of bounds.")

    # Calculate step size h from neighboring points
    # For indices at the boundaries, adjust the approach to use the closest available interval
    if i > 0:
        h = data[i][0] - data[i - 1][0]
    elif i < len(data) - 1:
        h = data[i + 1][0] - data[i][0]

    # Determine the method based on index position
    cfd_range = len(Centered_Coefficients[o][a]) // 2
    method = ''
    coefficients = []

    if i < cfd_range:
        method = 'FFD'
        coefficients = Forward_Diff_Coefficients[o][a]
    elif i > len(data) - cfd_range - 1:
        method = 'BFD'
        coefficients = Backward_Diff_Coefficients[o][a]
    else:
        method = 'CFD'
        coefficients = Centered_Coefficients[o][a]

    # Calculation
    derivative = 0
    if method == 'CFD':
        for j, coeff in enumerate(coefficients):
            index = i + j - cfd_range
            derivative += coeff * data[index][1]
    else:
        start_index = i if method == 'FFD' else i - len(coefficients) + 1
        for j, coeff in enumerate(coefficients):
            derivative += coeff * data[start_index + j][1]

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
