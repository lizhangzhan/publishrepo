cdef inline double fmax(double a, double b): return a if a >= b else b
cdef inline double fmin(double a, double b): return a if a <= b else b

cdef double sigm(double x)
cdef log_gaussian(double x, double mu, double sigma)
cdef log_gaussian_logsigma(double x, double mu, double logsigma)
