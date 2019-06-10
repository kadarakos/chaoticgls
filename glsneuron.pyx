cdef float T_map(float x, float b):
    """
    Compute  Generalized Luroth Series map function.
    """
    if x >= 0 and x < b:
        return x/b
    elif b <= x and x < 1:
        return (1 - x)/(1 - b)
    else:
        ValueError("invalid value encountered {}".format(x))

cpdef compute_gls(float x, float q, float b, float epsilon):
    """
    Compute firing time.
    """
    cdef int N_it = 0
    cdef int N_tol = 10000
    cdef float w = q
    cdef float I_high = x + epsilon
    cdef float I_low = x - epsilon 
    
    while True:
        w = T_map(w, b)
        N_it += 1
        if w <= I_high and w >= I_low:
            return N_it
        if N_it == N_tol:
            raise ValueError("Neuron iterated {} times. \nValue: {}".format(N_it, w))
            
