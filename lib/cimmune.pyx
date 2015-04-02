# cython: infer_types=True
# cython: boundscheck=True
# cython: wraparound=False

import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

def build_2d_frp_matrix(func, vector):
    """ Builds quadratic frp matrix respecting pbc.

    func: Kernel function
    vector: vector of equally spaced points
    """
    cdef double spacing = np.diff(vector)[0]
    cdef int Nvec = len(vector)
    cdef int N = Nvec**2
    cdef np.ndarray[DTYPE_t, ndim = 2] A = np.zeros((N, N), dtype = DTYPE)
    cdef double func0 = func(0)
    cdef unsigned int row, col
    cdef int x1, y1, x2, y2, xshift, yshift
    cdef double value
    for row in range(N):
        x1 = row % Nvec
        y1 = row / Nvec
        A[row, row] = func0
        for col in range(row):
            x2 = col % Nvec
            y2 = col / Nvec
            value = 0
            for xshift in range(-1,2):
                for yshift in range(-1,2):
                    value += func( spacing * ((x1-x2+xshift*Nvec)**2 + (y1-y2+yshift*Nvec)**2)**.5 )
            A[row, col] = value
            A[col, row] = value
    return A
