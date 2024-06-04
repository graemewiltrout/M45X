"""
Numerical Methods Package: Initialization
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""


"""
Numerical Methods Package: Initialization
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
    runge_kutta_4,
    adams_bashforth_2,
    adams_bashforth_4,
    taylors_method_O3
)
from .BasicMath import (
    factorial
)
from .SpecialNumbers import (
    pi,
    e
)
from .Utilities import (
    data    
)
from .Complex import (
    Complex
)

from .LinearAlgebra import (
    gaussian_elimination,
    gaussian_elimination_pp,
    gaussian_elimination_spp,
    is_symmetric,
    is_positive_definite,
    cholesky_decomposition,
    cholesky_solve,
    lu_decomposition,
    forward_substitution,
    backward_substitution,
    solve_system,
    power_method
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
    "runge_kutta_4",
    "adams_bashforth_2",
    "adams_bashforth_4",
    "taylors_method_O3",
    "factorial",
    "pi",
    "e",
    "data",
    "Complex",
    "gaussian_elimination",
    "gaussian_elimination_pp",
    "gaussian_elimination_spp",
    "is_symmetric",
    "is_positive_definite",
    "cholesky_decomposition",
    "cholesky_solve",
    "lu_decomposition",
    "forward_substitution",
    "backward_substitution",
    "solve_system",
    "power_method"
]