"""
Numerical Methods Package: Integration
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

"""
Geometric Approximations Overview:
    These are really easy, its just rectangels and trapezoids
    For the endpoint methods, it calculates the chosed endpoint and multiplies by the height
    For the trapezoid, it calculates the trapezoid
    
    Then it sums all the shapes within the given interval
    
    These functions expect you to specify the following:
        f = the function to evaluate
        h = step size (also known as dx)
        l = left bound
        r = right bound
"""

def left_endpoint(f, h, l, r):
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if h <= 0:
        raise ValueError("Step size 'h' must be a positive number.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)) or l >= r:
        raise ValueError("Ensure 'l' is less than 'r' and both are numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")
    total = 0.0
    x = l
    while x < r:
        total += f(x) * h
        x += h
    return total

def midpoint(f, h, l, r):
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if h <= 0:
        raise ValueError("Step size 'h' must be a positive number.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)) or l >= r:
        raise ValueError("Ensure 'l' is less than 'r' and both are numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")
    total = 0.0
    x = l + h/2
    while x < r:
        total += f(x) * h
        x += h
    return total

def right_endpoint(f, h, l, r):
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if h <= 0:
        raise ValueError("Step size 'h' must be a positive number.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)) or l >= r:
        raise ValueError("Ensure 'l' is less than 'r' and both are numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")
    total = 0.0
    x = l + h
    while x <= r:
        total += f(x) * h
        x += h
    return total

def trapezoid(f, h, l, r):
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if h <= 0:
        raise ValueError("Step size 'h' must be a positive number.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)) or l >= r:
        raise ValueError("Ensure 'l' is less than 'r' and both are numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")
    total = 0.0
    n = int((r - l) / h)
    for i in range(n):
        x0 = l + i * h
        x1 = l + (i + 1) * h
        total += (f(x0) + f(x1)) * h / 2
    return total


"""
Advanced Integration Techniques Overview:
    Simpson is a nerd who uses parabolic arcs instead of lines for approzimation
    That trait requires it to have an even number of intervals which I'm sure I'll explain in the writeup
    Instead of returning an error if the number of intervals is odd, I just add an interval
    
    Gauss is also a nerd, it uses roots and weights for its approximation, YouTube will teach you why
    
    Rombone is my personal favorite because of the name, he uses trapezoids with a Richardson extrapolation
    This lets us use bigger steps for the same accuracy as more steps of trap
    
    These functions expect you to specify the following:
        f = the function to evaluate
        l = left bound
        r = right bound   
        n = number of intervals
"""


def simpson(f, l, r, n):
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of intervals 'n' must be a positive integer.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
        raise TypeError("Bounds 'l' and 'r' must be numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")

    if n % 2 == 1:  # Adjust if n is odd
        print(f"Number of intervals 'n' was {n}, which is odd. Adjusting 'n' to {n+1} to be even for Simpson's rule.")
        n += 1
    h = (r - l) / n
    total = f(l) + f(r)
    for i in range(1, n, 2):
        total += 4 * f(l + i * h)
    for i in range(2, n-1, 2):
        total += 2 * f(l + i * h)
    return total * h / 3

def gaussian_quadrature(f, l, r, n):
    """
    Perform Gaussian Quadrature integration using precomputed roots and weights.
    Will eventually implement calculating roots on the fly
    """
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of intervals 'n' must be a positive integer.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
            raise TypeError("Bounds 'l' and 'r' must be numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")

    # Precomputed roots and weights for Legendre polynomials up to degree 5
    roots_weights = {
        2: [(-0.5773502692, 1), (0.5773502692, 1)],
        3: [(-0.7745966692, 0.5555555556), (0, 0.8888888889), (0.7745966692, 0.5555555556)],
        4: [(-0.8611363116, 0.3478548451), (-0.3399810436, 0.6521451549), 
            (0.3399810436, 0.6521451549), (0.8611363116, 0.3478548451)],
        5: [(-0.9061798459, 0.2369268850), (-0.5384693101, 0.4786286705), 
            (0, 0.5688888889), (0.5384693101, 0.4786286705), (0.9061798459, 0.2369268850)]
    }
    
    if n not in roots_weights:
        raise ValueError(f"Roots and weights for n={n} are not implemented. Available: {list(roots_weights.keys())}")
    
    # Transform the roots and weights from the standard interval [-1, 1] to [l, r]
    def transform(x):
        return 0.5 * (x * (r - l) + (r + l))
    
    total = 0
    for root, weight in roots_weights[n]:
        total += weight * f(transform(root))
    return total * 0.5 * (r - l)

def romberg(f, l, r, depth=3):
    """
    Romberg Integration using the rTrapezoid function for initial estimates
    This takes in f, l, and r like the others
    It differs by including depth instead of step size or quantity of intervals
    Depth determines accuracy, you basically trade computations for accuracy
    It defaults to 3 because thats the number I chose
    """
    #Input Error Checks & Statements
    if not callable(f):
        raise TypeError("Expected a callable function for parameter 'f', but received a non-callable type.")
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
        raise TypeError("Bounds 'l' and 'r' must be numbers.")
    if l >= r:
        raise ValueError("Left bound 'l' must be less than right bound 'r'.")
    
    R = [[0 for _ in range(depth)] for _ in range(depth)]
    for i in range(depth):
        n = 2**i
        R[i][0] = rTrapezoid(f, l, r, n)
        for j in range(1, i + 1):
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
    return R[-1][-1]

def rTrapezoid(f, l, r, n):
    """
    rTrapezoid Function for Romberg Integration
    It takes in n for the number of intervals instead of h for the step size
    Might look into unifying them
    """
    if n < 1:
        raise ValueError("Number of intervals n must be at least 1.")
    
    h = (r - l) / n  # Calculate the step size based on the number of intervals
    total = 0.5 * (f(l) + f(r))  # Initialize total with the first and last term
    
    for i in range(1, n):
        x = l + i * h  # Calculate the current x position
        total += f(x)  # Add the function value at this position
    
    total *= h  # Multiply the sum by the step size
    
    return total