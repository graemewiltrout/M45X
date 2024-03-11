"""
Numerical Methods Package: Ordinary Differential Equations
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

def eulers_method(f, h, l, r, y0, t0=0):
    """
    Parameters
    ----------
    f : The function f(t,y)
    h : The step size
    l : Left bound
    r : Right bound
    y0: Initial value of Y
    t0: Initial value of t (default is 0)


    Returns
    ts: The t values as a list.
    ys: The y values as a list.
    """
    
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        y_next = y + h * f(t,y)
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
    return ts, ys


def runge_kutta_2_midpoint(f, h, l, r, y0, t0=0):
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        
        y_next = y + h * k2
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def runge_kutta_2_heun(f, h, l, r, y0, t0=0):
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        
        y_next = y + h * (k1 + k2) / 2
        t_next = t + h
        
        if t_next > r:
            break
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def runge_kutta_4(f, h, l, r, y0, t0=0):
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t + h
        
        if t_next > r:
            break
            
        ts.append(t_next)
        ys.append(y_next)
        
    return ts, ys

def adams_bashforth_2(f, h, l, r, y0):
    # Use Euler's method to get the first two values
   ts_euler, ys_euler = eulers_method(f, h, l, l+h, y0)  # Only need two steps

   # Initialize ts and ys using the results from Euler's method
   ts, ys = [l], [y0]
   if len(ts_euler) > 1:
       ts.append(ts_euler[-1])
       ys.append(ys_euler[-1])

   # Apply AB2 for subsequent steps
   for i in range(2, int((r - l)/h) + 1):
       t = l + i*h
       if t > r:
           break
       y_next = ys[-1] + h * (3/2 * f(ts[-1], ys[-1]) - 1/2 * f(ts[-2], ys[-2]))
       ts.append(t)
       ys.append(y_next)

   return ts, ys

def adams_bashforth_4(f, h, l, r, y0):
    # Use Euler's method to get the first four values
    ts_euler, ys_euler = eulers_method(f, h, l, l+3*h, y0)  # Need four steps for AB4 initialization

    # Initialize ts and ys with Euler method results (if the interval is smaller than 4 steps, adjust accordingly)
    ts, ys = ts_euler[:], ys_euler[:]

    # Apply AB4 for subsequent steps, ensuring there are enough points
    for i in range(len(ts), int((r - l)/h) + 1):
        t = l + i*h
        if t > r:
            break
        y_next = ys[-1] + h * (55/24 * f(ts[-1], ys[-1]) - 59/24 * f(ts[-2], ys[-2]) +
                               37/24 * f(ts[-3], ys[-3]) - 9/24 * f(ts[-4], ys[-4]))
        ts.append(t)
        ys.append(y_next)

    return ts, ys


def taylors_method(f, dd, h, l, r, y0, t0=0):
    """
    Parameters
    ----------
    f : The function f(t,y)
    dd: A list of derivative functions of f with respect to t
    h : The step size
    l : Left bound
    r : Right bound
    y0: Initial value of y
    t0: Initial value of t (default is 0)

    Returns
    ts: The t values as a list.
    ys: The y values as a list.
    """
    from M45X import factorial
    
    ts = [t0]
    ys = [y0]
    
    while ts[-1] < r:
        t, y = ts[-1], ys[-1]
        y_next = y
        
        for i, df in enumerate(dd, start=1):  # Ensure this matches the parameter name (`dd` in this case)
            term = df(t, y) * (h**i) / factorial(i)
            y_next += term
        
        t_next = t + h
        
        if t_next > r:
            break
            
        ts.append(t_next)
        ys.append(y_next)
    
    return ts, ys
