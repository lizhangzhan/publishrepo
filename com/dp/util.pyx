import numpy as np

cimport cython
from libc.math cimport exp, log
cimport numpy as np

np.import_array()

cdef double sigm(double x):
    """truncated sigmoid function."""
    return 1 / (1 + exp(-fmax(fmin(x, 20.0), -20.0)))

cdef double sigmoid(double x):
    return sigm(x)

cdef log_gaussian(double x, double mu, double sigma):
    return (-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma)) -
            (x - mu) ** 2 / (2 * sigma ** 2))

cdef log_gaussian_logsigma(double x, double mu, double logsigma):
        return (0.5 * np.log(2 * np.pi) - logsigma * 0.5 -
                (x - mu) ** 2 / (2. * np.exp(logsigma)))

