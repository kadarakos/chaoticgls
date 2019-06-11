cdef long double T_map(long double x, long double b):
    """
    Compute  Generalized Luroth Series map function.
    """
    if x >= 0 and x < b:
        return x/b
    elif b <= x and x < 1:
        return (1 - x)/(1 - b)
    else:
        ValueError("invalid value encountered {}".format(x))

cpdef compute_gls(long double x, long double q, long double b, long double epsilon):
    """
    Compute firing time.
    """
    cdef int N_it = 0
    cdef int N_tol = 10000
    cdef long double w = q
    cdef long double I_high = x + epsilon
    cdef long double I_low = x - epsilon 
    
    while True:
        w = T_map(w, b)
        N_it += 1
        if w <= I_high and w >= I_low:
            return N_it
        if N_it == N_tol:
            raise ValueError("Neuron iterated {} times. \nValue: {}".format(N_it, w))
            
