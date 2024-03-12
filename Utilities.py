"""
Numerical Methods Package: Utilities
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np


def data(func, l, r, h):
    """
    Generates x values and their corresponding f(x) values within a given domain.

    Parameters
    ----------
    func : function
        The function for which f(x) values are to be computed.
    l : float
        The left boundary of the domain.
    r : float
        The right boundary of the domain.
    h : float
        The step size for generating x values.

    Returns
    -------
    dataset : list of tuples
        A list of (x, f(x)) tuples.
    """
    # Corrected np.arange and variable names for clarity
    x_values = np.arange(l, r + h, h)
    dataset = [(x, func(x)) for x in x_values]
    
    return dataset
