import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np

np.import_array()

cdef class SVRG4MLP2:
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

        weights: array, weights for edges connecting layers

        delayed_weights: array, delayed weights for edges connecting layers

        z : array, hidden neuron activations

        avg_grads : array, history grad squre exponential decay average

        grads: array, grads of weights for edges connecting layers

        delayed_grads: array, grads of delayed weights for edges connecting layers
    """
    cdef unsigned int n     # num of input neurons
    cdef unsigned int h1    # num of the 1st level hidden neurons
    cdef unsigned int h2    # num of the 2nd level hidden neurons

    cdef double alpha       # per-coordinate step size
    cdef double decay       # per-coordinate decay rate
    cdef double epsilon     # smooth factor
    cdef double stdev

    cdef double l1          # l1 regularization
    cdef double l2          # l2 regularization

    cdef double[:] weights  # w for edges connecting adjacent layers

    # a delayed w in each m iterations
    cdef double[:] delayed_weights

    cdef double[:] z         # hidden neuron activations

    #[Note] approximate mean of history graident (exponential decay)
    cdef double[:] avg_grads # mean of history grads
    
    # anxilliary array
    cdef double[:] grads
    cdef double[:] delayed_grads

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
        
        param_num = (h2 + 1) + (h1 + 1) * h2 + (n + 1) * h1
        # weights for edges connecting adjacent layers
        self.weights = (rng.rand(param_num) - 0.5) * stdev


        # hidden neurons activations
        self.z = np.zeros((h1 + h2,), dtype=np.float64)

        # sum of history grad squre average, non-zeros
        self.grad_avgs = np.zeros(((self.n + 1) * self.h1), dtype=np.float64)

        # a snapshot w in each m iterations with deep copy
        self.sw2 = np.copy(self.w2)
        self.sw1 = np.copy(self.w1)
        self.sw0 = np.copy(self.w0)

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

    def forward(self, tuple weights, bool ):
        """Predict for features.

        Parameters:
        -----------
            weights: a tuple of weights
            x      : a list of (index, value) of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef double p
        cdef int k
        cdef int j
        cdef int i
        cdef double v

        w0, w1, w2 = weights

        # starting from the bias in the 2nd hidden layer(no regularization)
        p = w2[self.h2]

        # calculating and adding values of 1st level hidden neurons
        for j in range(self.h1):
            # starting with the bias in the input layer(no regularization)
            index0 = self.n * self.h1 + j
            self.z1[j] = w0[index0]

            # calculating and adding values of input neurons
            for i, v in x:
                index0 = self.h1 * i + j
                self.z1[j] += w0[index0] * v

            # apply the ReLU activation function to the first level hidden unit
            self.z1[j] = self.z1[j] if self.z1[j] > 0. else 0.

        # calculating and adding values of 2nd level hidden neurons
        for k in range(self.h2):
            # staring with the bias in the 1st hidden layer(no regularization)
            index1 = self.h1 * self.h2 + k
            self.z2[k] = w1[index1]
            for j in range(self.h1):
                index1 = self.h2 * j + k
                self.z2[k] += w1[index1] * self.z1[j]

            # apply the ReLU activation function to the 2nd level hidden layer
            self.z2[k] = self.z2[k] if self.z2[k] > 0. else 0.
            p += w2[k] * self.z2[k]

        # apply the sigmoid activation function to the output layer
        return sigm(p)

    def backword(self, tuple weights, list x, double e):
        """ compute the gradients for the given weight snapshot

        Parameters:
        -----------
          weights(tuple of w): a tuple of w for adjacent layers connection
          x (list of tuple): a list of (index, value) of non-zero features
          e (double): error between the prediction of the model and target

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
        cdef double[:] w0
        cdef double[:] w1
        cdef double[:] w2

        w0, w1, w2 = weights
        rho = self.decay
        l1 = self.l1
        l2 = self.l2

        # error: diff between prediction and true label
        dl_dy = e

        # adapt grad average for bias in 2nd hidden layer(no regularization)
        self.g_avg = rho * self.g_avg + (1 - rho) * dl_dy

        # backpropagation for weight update
        for k in range(self.h2): # for h2 layer
            # update weights related to non-zero 2nd level hidden neurons
            if self.z2[k] == 0.0:
                    continue

             # update weights between the 2nd hidden neurons and output
            # dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
            dl_dw2 = (dl_dy * self.z2[k] + l1 * np.sign(w2[k]) + l2 * self.w2[k])

            # adapt grad average in 2nd hidden layer
            self.g2_avg[k] = (rho * self.g2_avg[k] + (1 - rho) * dl_dw2)

            # starting with the bias in the 1st hidden layer(no regularization)
            # dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
            dl_dz2 = dl_dy * w2[k]
            index1 = self.h1 * self.h2 + k

            # adapt grad average in 1st hidden layer
            self.g1_avg[index1] = (rho * self.g1_avg[index1] + (1 - rho) * dl_dz2)

            # backpropagation for h1 layer
            for j in range(self.h1):
                # update weights realted to non-zero hidden neurons
                if self.z1[j] == 0.0:
                    continue

                # update weights between the 1st layer and 2nd layer
                # dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
                index1 = self.h2 * j + k
                dl_dw1 = (dl_dz2 * self.z1[j] + l1 * np.sign(self.w1[index1]) + l2 * self.w1[index1])

                # adapt grad average in 1st hidden layer
                self.g1_avg[index1] = (rho * self.g1_avg[index1] + (1 - rho) * dl_dw1)

                # starting with the bias in the input layer(no regularization)
                # dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
                dl_dz1 = dl_dz2 * self.w1[j * self.h2 + k]

                # bias index for 1st hidden layer
                index0 = self.n * self.h1 + j
                # adapt grad squre average in 1st hidden layer
                self.g0_avg[index0] = (rho * self.g0_avg[index0] + (1 - rho) * dl_dz1)

                # update weights related to non-zero input neurons
                for i, v in x:
                    # update weights between the hidden unit j and input i
                    # dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
                    index0 = self.h1 * i + j
                    dl_dw0 = (dl_dz1 * v + l1 * np.sign(self.w0[index0]) + l2 * self.w0[index0])
                    # adapt grad average in 1st hidden layer
                    self.g0_avg[index0] = (rho * self.g0_avg[index0] + (1 - rho) * dl_dw0)
