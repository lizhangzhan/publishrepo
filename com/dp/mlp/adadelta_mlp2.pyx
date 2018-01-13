import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np

np.import_array()

cdef class AdaDelta4MLP2:
    """
    Neural Network with 2 ReLU hidden layers online learner.

    Attributes:
    ----------
        n : integer, number of input neurons
        h1 : integer, number of the 1st level hidden neurons
        h2 : integer, number of the 2nd level hidden neurons
        
        alpha : double, per-coordinate step size
        decay : double, per-coordinate decay rate
        epsilon: double, smooth factor
        stdev : double, deviation for weight initialization
        l1    :  l1 regularization
        l2    :  l2 regularization

        w0 : array, weights between the input and 1st hidden layers
        w1 : array, weights between the 1st and 2nd hidden layers
        w2 : array, weights between the 2nd hidden and output layers
        
        z1 : array, 1st level hidden neurons
        z2 : array, 2nd level hidden neurons
        
        g_sqr_avg : double, history grad squre average for bias in h2 layer
        g1_sqr_avg : array, history grad squre average for 1st level hidden neurons
        g2_sqr_avg : array, history grad squre average for 2nd level hidden neurons

        u_sqr_avg : double, history update squre average for bias in h2 layer
        u1_sqr_avg : array, history update squre average for 1st level hidden neurons
        u2_sqr_avg : array, history update squre average for 2nd level hidden neurons
    """

    cdef unsigned int n     # number of input neurons
    cdef unsigned int h1    # number of the 1st level hidden neurons
    cdef unsigned int h2    # number of the 2nd level hidden neurons

    cdef double alpha       # per-coordinate step size
    cdef double decay       # per-coordinate decay rate
    cdef double epsilon     # smooth factor
    cdef double stdev
    
    cdef double l1          # l1 regularization
    cdef double l2          # l2 regularization

    cdef double[:] w0       # weights between the input and 1st hidden layers
    cdef double[:] w1       # weights between the 1st and 2nd hidden layers
    cdef double[:] w2       # weights between the 2nd hidden and output layers
    
    cdef double[:] z1       # 1st level hidden neurons
    cdef double[:] z2       # 2nd level hidden neurons
    
    #[Note] Initialize gx_sum randomly 
    cdef double g_sqr_avg       # average of history grad  squre for bias in h2 layer
    cdef double[:] g0_sqr_avg   # average of history grads squre for weights in input layer
    cdef double[:] g1_sqr_avg   # average of history grads squre for weights in h1 layer 
    cdef double[:] g2_sqr_avg   # average of history grads squre for weights in h2 layer except for bias
    
    cdef double u_sqr_avg       # average of history update squre for bias in h2 layer
    cdef double[:] u0_sqr_avg   # average of history update squre for weights in input layer
    cdef double[:] u1_sqr_avg   # average of history update squre for weights in h1 layer 
    cdef double[:] u2_sqr_avg   # average of history update squre for weights in h2 layer except for bias

    def __init__(self,
                 unsigned int n,
                 unsigned int h1=16,
                 unsigned int h2=8,
                 double a=0.01,
                 double b=1e-6,
                 double decay=(1.0 - 1e-6),
                 double stdev=1e-3,
                 double l1=0.,
                 double l2=1e-3,
                 unsigned int seed=0):
        """
        Multiple layer perception with two hidden layer

        Parameters:
        -----------
            n : integer, number of input neurons
            h1 : integer, number of the 1st level hidden neurons
            h2 : integer, number of the 2nd level hidden neurons
            a : double, per-coordinate step rate
            decay (double): per-coordinate decay_rate
            b : double, smooth factor
            stdev : standard deviation for weight nitialization
            seed (unsigned int): random seed
        """

        rng = np.random.RandomState(seed)

        self.n = n
        self.h1 = h1
        self.h2 = h2

        self.alpha = a
        self.decay = decay
        self.epsilon = b
        self.stdev = stdev
        
        self.l1 = l1
        self.l2 = l2

        # weights between the output and 2nd hidden layer
        self.w2 = (rng.rand(self.h2 + 1) - 0.5) * stdev

        # weights between the 2nd hidden layer and 1st hidden layer
        self.w1 = (rng.rand((self.h1 + 1) * self.h2) - 0.5) * stdev

        # weights between the 1st hidden layer and inputs
        self.w0 = (rng.rand((self.n + 1) * self.h1) - 0.5) * stdev

        # hidden neurons in the 2nd hidden layer
        self.z2 = np.zeros((self.h2,), dtype=np.float64)
        
        # hidden neurons in the 1st hidden layer
        self.z1 = np.zeros((self.h1,), dtype=np.float64)
        
        # sum of history grad squre average, non-zeros
        self.g_sqr_avg = 0.0
        self.g2_sqr_avg = np.zeros((self.h2,), dtype=np.float64)
        self.g1_sqr_avg = np.zeros(((self.h1 + 1) * self.h2), dtype=np.float64)
        self.g0_sqr_avg = np.zeros(((self.n + 1) * self.h1), dtype=np.float64)
         
        # sum of history grad squres
        self.u_sqr_avg = 0.0
        self.u2_sqr_avg = np.zeros((self.h2,), dtype=np.float64)
        self.u1_sqr_avg = np.zeros(((self.h1 + 1) * self.h2), dtype=np.float64)
        self.u0_sqr_avg = np.zeros(((self.n + 1) * self.h1), dtype=np.float64)
        
    def __repr__(self):
        return ('AdaDelta4MLP2(n={}, h1={}, h2={}, alpha={}, epsilon={}, \
            decay={}, stdev={}, l1={}, l2={})').format(self.n, self.h1, self.h2, 
            self.alpha, self.beta, self.epsilon, self.stdev, self.l1, self.l2)

    def read_sparse(self, path):
        """Read the libsvm format sparse file line by line.

        Parameters:
        -----------
            path (str): a file path to the libsvm format sparse file

        Yields:
            idx (list of int): a list of index of non-zero features
            val (list of double): a list of values of non-zero features
            y (int): target value
        """
        for line in open(path):
            xs = line.rstrip().split(' ')

            y = int(xs[0])
            idx = []
            val = []
            for item in xs[1:]:
                i, v = item.split(':')
                idx.append(abs(hash(i)) % self.n)
                val.append(float(v))

            yield zip(idx, val), y

    def predict_one(self, list x):
        """Predict for features.

        Parameters:
        -----------
            x (list of tuple): a list of (index, value) of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef double p
        cdef int k
        cdef int j
        cdef int i
        cdef double v

        # starting from the bias in the 2nd hidden layer(no regularization)
        p = self.w2[self.h2]

        # calculating and adding values of 1st level hidden neurons
        for j in range(self.h1):
            # starting with the bias in the input layer(no regularization)
            index0 = self.n * self.h1 + j
            self.z1[j] = self.w0[index0]

            # calculating and adding values of input neurons
            for i, v in x:
                index0 = self.h1 * i + j
                self.z1[j] += self.w0[index0] * v

            # apply the ReLU activation function to the first level hidden unit
            self.z1[j] = self.z1[j] if self.z1[j] > 0. else 0.

        # calculating and adding values of 2nd level hidden neurons
        for k in range(self.h2):
            # staring with the bias in the 1st hidden layer(no regularization)
            index1 = self.h1 * self.h2 + k
            self.z2[k] = self.w1[index1]
            for j in range(self.h1):
                index1 = self.h2 * j + k
                self.z2[k] += self.w1[index1] * self.z1[j]

            # apply the ReLU activation function to the 2nd level hidden layer
            self.z2[k] = self.z2[k] if self.z2[k] > 0. else 0.
            p += self.w2[k] * self.z2[k]

        # apply the sigmoid activation function to the output layer
        return sigm(p)

    def update_one(self, list x, double e):
        """Update the model.

        Parameters:
        -----------
            x (list of tuple): a list of (index, value) of non-zero features
            e (double): error between the prediction of the model and target

        Returns:
            updated model weights and counts
        """
        cdef int k
        cdef int j
        cdef int i
        cdef double dl_dy
        cdef double dl_dz1
        cdef double dl_dz2
        cdef double dl_dw0
        cdef double dl_dw1
        cdef double dl_dw2
        cdef double v
        cdef double rho

        rho = self.decay
        epsilon = self.epsilon
        alpha = self.alpha
        l1 = self.l1
        l2 = self.l2

        # error: diff between prediction and true label
        dl_dy = e

        # adapt grad squre average for bias in 2nd hidden layer(no regularization)
        self.g_sqr_avg = rho * self.g_sqr_avg + (1 - rho) * dl_dy * dl_dy
        # update bias weight
        delta_upd = (alpha * sqrt(self.u_sqr_avg + epsilon) / sqrt(self.g_sqr_avg + epsilon)) * dl_dy
        self.w2[self.h2] -= delta_upd
        # adapt update squre average for bias in 2nd hidden layer 
        self.u_sqr_avg = rho * self.u_sqr_avg + (1 - rho) * delta_upd * delta_upd
        
        # backpropagation for weight update
        for k in range(self.h2): # for h2 layer
            # update weights related to non-zero 2nd level hidden neurons
            if self.z2[k] == 0.0:
                    continue

             # update weights between the 2nd hidden neurons and output
            # dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
            dl_dw2 = (dl_dy * self.z2[k] + l1 * np.sign(self.w2[k]) + l2 * self.w2[k])

            # adapt grad squre average in 2nd hidden layer
            self.g2_sqr_avg[k] = (rho * self.g2_sqr_avg[k] + (1 - rho) * dl_dw2 * dl_dw2)
            # update weights in 2nd hidden layer
            delta_upd = alpha * ( sqrt(self.u2_sqr_avg[k] + epsilon) / sqrt(self.g2_sqr_avg[k] + epsilon) ) * dl_dw2
            self.w2[k] -= delta_upd
            # adapt update squre average in 2nd hidden layer 
            self.u2_sqr_avg[k] = (rho * self.u2_sqr_avg[k] + (1 - rho) * delta_upd * delta_upd)

            # starting with the bias in the 1st hidden layer(no regularization)
            # dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
            dl_dz2 = dl_dy * self.w2[k]
            index1 = self.h1 * self.h2 + k
            
            # adapt grad squre average in 1st hidden layer
            self.g1_sqr_avg[index1] = (rho * self.g1_sqr_avg[index1] + (1 - rho) * dl_dz2 * dl_dz2)
            # update bias weight in 1st hidden layer
            delta_upd = alpha * ( sqrt(self.u1_sqr_avg[index1] + epsilon) / sqrt(self.g1_sqr_avg[index1] + epsilon) ) * dl_dz2
            self.w1[index1] -= delta_upd
            # adapt update squre average for bias in 1st hidden layer 
            self.u1_sqr_avg[index1] = (self.u1_sqr_avg[index1] * rho + (1 - rho) * delta_upd * delta_upd)
            
            # backpropagation for h1 layer
            for j in range(self.h1):
                # update weights realted to non-zero hidden neurons
                if self.z1[j] == 0.0:
                    continue

                # update weights between the 1st layer and 2nd layer
                # dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
                index1 = self.h2 * j + k
                dl_dw1 = (dl_dz2 * self.z1[j] + l1 * np.sign(self.w1[index1]) + l2 * self.w1[index1])
                
                # adapt grad squre average in 1st hidden layer
                self.g1_sqr_avg[index1] = (rho * self.g1_sqr_avg[index1] + (1 - rho) * dl_dw1 * dl_dw1)
                # update bias weight in 1st hidden layer
                delta_upd = alpha * ( sqrt(self.u1_sqr_avg[index1] + epsilon) / sqrt(self.g1_sqr_avg[index1] + epsilon) ) * dl_dw1
                self.w1[index1] -= delta_upd
                # adapt update squre average for bias in 1st hidden layer 
                self.u1_sqr_avg[index1] = (rho * self.u1_sqr_avg[index1] + (1 - rho) * delta_upd * delta_upd)

                # starting with the bias in the input layer(no regularization)
                # dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
                dl_dz1 = dl_dz2 * self.w1[j * self.h2 + k]
                
                # bias index for 1st hidden layer
                index0 = self.n * self.h1 + j
                # adapt grad squre average in 1st hidden layer
                self.g0_sqr_avg[index0] = (rho * self.g0_sqr_avg[index0] + (1 - rho) * dl_dz1 * dl_dz1)
                # update bias weight in 1st hidden layer
                delta_upd = alpha * ( sqrt(self.u0_sqr_avg[index0] + epsilon) / sqrt(self.g0_sqr_avg[index0] + epsilon) ) * dl_dz1
                self.w0[index0] -= delta_upd
                # adapt update squre average for bias in 1st hidden layer 
                self.u0_sqr_avg[index0] = (rho * self.u0_sqr_avg[index0] + (1 - rho) * delta_upd * delta_upd)

                # update weights related to non-zero input neurons
                for i, v in x:
                    # update weights between the hidden unit j and input i
                    # dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
                    index0 = self.h1 * i + j
                    dl_dw0 = (dl_dz1 * v + l1 * np.sign(self.w0[index0]) + l2 * self.w0[index0])
                    # adapt grad squre average in 1st hidden layer
                    self.g0_sqr_avg[index0] = (rho * self.g0_sqr_avg[index0] + (1 - rho) * dl_dw0 * dl_dw0)
                    # update bias weight in 1st hidden layer
                    delta_upd = alpha * ( sqrt(self.u0_sqr_avg[index0] + epsilon) / sqrt(self.g0_sqr_avg[index0] + epsilon) ) * dl_dw0
                    self.w0[index0] -= delta_upd
                    # adapt update squre average for bias in 1st hidden layer 
                    self.u0_sqr_avg[index0] = (rho * self.u0_sqr_avg[index0] + (1 - rho) * delta_upd * delta_upd)
