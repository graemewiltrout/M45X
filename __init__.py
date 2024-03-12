"""
Numerical Methods Package: Initialization
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""
from .NumericalDifferentiation import (
    forward_finite_difference,
    centered_finite_difference,
    backward_finite_difference,
    finite_difference,
    richardson_extrapolation
    )
from .NumericalIntegration import (
    left_endpoint,
    midpoint,
    right_endpoint,
    trapezoid,
    simpson,
    gaussian_quadrature,
    romberg
    )
from .NumericalRootFinding import (
    bisection,
    unified_newtons_method,
    secant_method
    )
from .ODE import (
    eulers_method,
    runge_kutta_2_midpoint,
    runge_kutta_2_heun,
    taylors_method
    )
from .BasicMath import(
    factorial
)
from .SpecialNumbers import(
    pi,
    e
)

from .Utilities import(
    data    
)
__all__ = [
    "forward_finite_difference",
    "centered_finite_difference",
    "backward_finite_difference",
    "finite_difference",
    "richardson_extrapolation",
    "left_endpoint",
    "midpoint",
    "right_endpoint",
    "trapezoid",
    "simpson",
    "gaussian_quadrature",
    "romberg",
    "bisection",
    "unified_newtons_method",
    "secant_method",
    "eulers_method",
    "runge_kutta_2_midpoint",
    "runge_kutta_2_heun",
    "taylors_method",
    "factorial",
    "pi",
    "e",
    "data"
]
