import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np

np.import_array()

cdef class FTRL4MLP2:
    """
    Neural Network with 2 ReLU hidden layers online learner.

    Attributes:
    -----------
        n : integer, number of input neurons
        h1 : integer, number of the 1st level hidden neurons
        h2 : integer, number of the 2nd level hidden neurons
        alpha : double, step rate (adadelta) in hidden layers 
        decay : double, decay rate (adadelta) in hidden layers 
        epsilon : double, smooth factor (adadelta) in hidden layers 

        alpha0 : double, initial learning rate (ftrl) in input layer
        beta0 : double, initial learning rate (ftrl) in input layer
        decay0 : double, decay rate in input layer

        l1 : double, L1 regularization parameter
        l2 : double, L2 regularization parameter
        
        w0 : array, shape(input_num, hidden1_num)
            weights between the input and 1st hidden layers
        
        w1 : array, shape(hidden1_num, hidden2_num)
            weights between the 1st and 2nd hidden layers
        
        w2 : array, shape(hidden2_num,)
            weights between the 2nd hidden and output layers
        
        z1 : array, shape(hidden1_num,)
            value in 1st level hidden neurons
        z2 : array, shape(hidden2_num,)
            values in 2nd level hidden neurons
        
        # auxiliary structures for adadelta
        g_sqr_avg : double, for adadelta
           history grad squre average for bias in h2 layer

        g1_sqr_avg : array, shape(hidden1_num, hidden2_num)
           history grad squre average for 1st level hidden neurons

        g2_sqrt_avg : array, shape(hidden2_num,)
           history grad average for 2nd level hidden neurons

        u_sqr_sum : double, for adadelta
           history grad squre sum for bias in h2 layer

        u1_sqr_sum : array, shape(hidden1_num, hidden2_num)
           history grad squre sum for 1st level hidden neurons
        
        u2_sqr_sum : array, shape(hidden2_num,)
           history grad squre sum for 2nd level hidden neurons
        
        # auxiliary structures for ftrl
        g0_sum : array, shape(input_num, hidden1_num)
           history grad sum for input layer neurons

        g0_sqr_sum : array, shape(input_num, hidden1_num)
           history grad sum for input layer neurons
    """

    cdef unsigned int n     # number of input neurons
    cdef unsigned int h1    # number of the 1st level hidden neurons
    cdef unsigned int h2    # number of the 2nd level hidden neurons

    cdef double alpha       # step  rate for 1st/2nd hidden layer (adadelta)
    cdef double decay       # decay rate for 1st/2nd hidden layer(adadelta)
    cdef double epsilon     # smooth factor for 1st/2st hidden layer(adadelta)

    cdef double alpha0      # learning rate for input layer(ftrl)
    cdef double beta0       # learning rate for input layer(ftrl)
    cdef double decay0      # decay rate for for input layer(ftrl)
    
    cdef double l1          # L1 regularization parameter
    cdef double l2          # L2 regularization parameter
    
    cdef double[:] w0       # weights between the input and 1st hidden layers
    cdef double[:] w1       # weights between the 1st and 2nd hidden layers
    cdef double[:] w2       # weights between the 2nd hidden and output layers
    
    cdef double[:] z1       # 1st level hidden neurons
    cdef double[:] z2       # 2nd level hidden neurons
    
    #[Note] Initialize gx_sum randomly (ftrl)
    cdef double[:] g0_sum       # sum of history grads for weights in input layer
    cdef double[:] g0_sqr_sum   # sum of history grads squre for weights in input layer
    
    # [Note] Initialize [g|u]_sqr_avg with default value, 0 
    cdef double g_sqr_avg       # average of history grad squre for bias in h2 layer
    cdef double[:] g1_sqr_avg   # average of history grads squre for weights in h1 layer 
    cdef double[:] g2_sqr_avg   # average of history grads squre for weights in h2 layer except for bias

    cdef double u_sqr_avg       # average of history update squre for bias in h2 layer
    cdef double[:] u1_sqr_avg   # average of history update squre for weights in h1 layer 
    cdef double[:] u2_sqr_avg   # average of history update squre for weights in h2 layer except for bias
    
    def __init__(self,
                 unsigned int n,
                 unsigned int h1=16,
                 unsigned int h2=8,
                 double a=.1,
                 double decay=0.95,
                 double epsilon=1e-7,
                 double a0=0.1,
                 double b0=0.1,
                 double d0=1.0,
                 double l1=0.,
                 double l2=0.,
                 unsigned int seed=0):
        """
        Multiple layer perception with two hidden layer

        Parameters:
        ----------
            n :     integer, number of input neurons
            h1 :    integer, number of the 1st level hidden neurons
            h2 :    integer, number of the 2nd level hidden neurons
            a :     integer, step rate (adadelta)
            decay : double, decay rate (adadelta)
            epsilon: double, smooth factor, default smaller value
            a0:     double, learning rate (ftrl)
            b0:     double, learning rate (ftrl)
            d0:     double, decay rate (ftrl)
            l1 :    double, L1 regularization parameter
            l2 :    double, L2 regularization parameter
            seed:   unsigned integer, random seed
        """

        rng = np.random.RandomState(seed)

        self.n = n
        self.h1 = h1
        self.h2 = h2

        self.alpha = a
        self.decay = decay
        self.epsilon = epsilon
        
        self.alpha0 = a0
        self.beta0 = b0
        self.decay0 = d0
        self.l1 = l1
        self.l2 = l2
        
        cdef double stdev = 0.01

        # weights between the output and 2nd hidden layer
        self.w2 = (rng.rand(self.h2 + 1) - .5) * stdev

        # weights between the 2nd hidden layer and 1st hidden layer
        self.w1 = (rng.rand((self.h1 + 1) * self.h2) - .5) * stdev

        # weights between the 1st hidden layer and inputs
        self.w0 = (rng.rand((self.n + 1) * self.h1) - .5) * stdev

        # hidden neurons in the 2nd hidden layer
        self.z2 = np.zeros((self.h2,), dtype=np.float64)
        
        # hidden neurons in the 1st hidden layer
        self.z1 = np.zeros((self.h1,), dtype=np.float64)
        
        # sum of history grad, non-zeros for ftrl in input layer
        self.g0_sum = (rng.rand((self.n + 1) * self.h1) - .5) * stdev
        # sum of history grad squre, non-zeros for ftrl in input layer
        self.g0_sqr_sum = np.zeros(((self.n + 1) * self.h1), dtype=np.float64)

        # average of history grad squres for 1st/2nd hidden layer(adadelta)
        self.g_sqr_avg = 0.0
        self.g2_sqr_avg = np.zeros((self.h2,), dtype=np.float64)
        self.g1_sqr_avg = np.zeros(((self.h1 + 1) * self.h2), dtype=np.float64)
        
        # average of history grad squres for 1st/2nd hidden layer(adadelta)
        self.u_sqr_avg = 0.0
        self.u2_sqr_avg = np.zeros((self.h2,), dtype=np.float64)
        self.u1_sqr_avg = np.zeros(((self.h1 + 1) * self.h2), dtype=np.float64)
        
    def __repr__(self):                                                         
        return ("FTRL4MLP2(n={}, h1={}, h2={},\
            alpha={}, decay={}, epsilon={}, \
            alpha0={}, beta0={}, decay0={}, l1={}, l2={})").format(
            self.n, self.h1, self.h2, 
            self.alpha, self.decay, self.epsilon,
            self.alpha0, self.beta0, self.decay0, self.l1, self.l2
        )

    def read_sparse(self, path):
        """Read the libsvm format sparse file line by line.

        Parameters:
            path (str): a file path to the libsvm format sparse file

        Return: generator
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
            x : array-like
               a list of (index, value) of non-zero features

        Returns:
            p : double, a probability prediction for input features
        """
        cdef double p
        cdef int k
        cdef int j
        cdef int i
        cdef double v

        # Accumulate contribution of the bias in the 2nd hidden layer(no regularization)
        p = self.w2[self.h2]

        # 1. full connected layer, forward contribution from input layer to 1st hidden layer
        for j in range(self.h1):
            # lazily compute weight of the bias in the input layer(no regularization)
            index0 = self.n * self.h1 + j
            self.w0[index0] = (-1.0 * self.g0_sum[index0] / ((self.beta0 + sqrt(self.g0_sqr_sum[index0])) / self.alpha0))
            self.z1[j] = self.w0[index0]

            # accumulate the contribution for jth hidden neoron forwarded from input layer
            for i, v in x:
                index0 = self.h1 * i + j
                self.w0[index0] = (-1.0 * self.g0_sum[index0] / (self.l2 + (self.beta0 + sqrt(self.g0_sqr_sum[index0])) / self.alpha0))
                self.z1[j] += self.w0[index0] * v

        # 2. Active layer, apply the ReLU activation function to 1sth hidden inputs
        for j in range(self.h1):
            self.z1[j] = self.z1[j] if self.z1[j] > 0. else 0.
   
        # 3. full connected layer, forward contribution from 1st active layer to 2st hidden layer
        for k in range(self.h2):
            # Accumulate contribution of the bias in the 1st hidden layer(no regularization)
            index1 = self.h1 * self.h2 + k
            self.z2[k] = self.w1[index1]
            # accumulate the contribution for jth hidden neoron forwarded from 1st hidden layer
            for j in range(self.h1):
                index1 = self.h2 * j + k
                self.z2[k] += self.w1[index1] * self.z1[j]

        # 4. Active layer, apply the ReLU activation function to 1sth hidden inputs
        for k in range(self.h2):
            # apply the ReLU activation function to the 2nd hidden layer
            self.z2[k] = self.z2[k] if self.z2[k] > 0. else 0.
            p += self.w2[k] * self.z2[k]

        # apply the sigmoid activation function to the output layer
        return sigm(p)

    def update_one(self, list x, double e):
        """Update the model.

        Parameters:
        -----------
            x : array-like, input feature vector
               a list of (index, value) of non-zero features
            e : double
               error between the prediction of the model and target

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

        cdef double decay = self.decay
        cdef double epsilon = self.epsilon
        
        cdef double rho = self.decay0
        cdef double rho2 = rho * rho
        # error: diff between prediction and true label
        # 1. Full connected layer, backprop gradient from output layer to 2nd hidden layer
        dl_dy = e

        # update params of the bias in 2nd hidden layer
        # dl/db2 = dl/dy * dy/db2 = dl/dy * 1
        self.g_sqr_avg = decay * self.g_sqr_avg + (1. - decay) * dl_dy * dl_dy
        delta_up = self.alpha * sqrt((self.u_sqr_avg + epsilon) / (self.g_sqr_avg + epsilon)) * dl_dy
        self.w2[self.h2] -= delta_up
        # update overall grad squre for bias in 2nd layer
        self.u_sqr_avg = decay * self.u_sqr_avg + (1. - decay) * delta_up * delta_up
        
        # backpropagation to upate Parameters in 2nd hidden layer
        for k in range(self.h2): # for h2 layer
            # update weights related to non-zero neurons in the 2nd layer
            if self.z2[k] == 0.0:
                continue

            # update weights between the 2nd hidden neurons and output
            # dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
            dl_dw2 = dl_dy * self.z2[k]
            self.g2_sqr_avg[k] = decay * self.g2_sqr_avg[k] + (1. - decay) * dl_dw2 * dl_dw2
            delta_up = self.alpha * sqrt((self.u2_sqr_avg[k] + epsilon) / (self.g2_sqr_avg[k] + epsilon)) * dl_dw2
            self.w2[k] -= delta_up
            self.u2_sqr_avg[k] = decay * self.u2_sqr_avg[k] + (1 - decay) * delta_up * delta_up 

            # 2. Active layer, backprop gradient from 2nd hidden layer to 1st active layer
            # dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
            dl_dz2 = dl_dy * self.w2[k]
            
            # update params of bias in the 1st hidden layer
            # dl/db1 = dl/dz2 * dz2/db1 = dl/z2 * 1
            index1 = self.h1 * self.h2 + k
            self.g1_sqr_avg[index1] = decay * self.g1_sqr_avg[index1] + (1. - decay) * dl_dz2 * dl_dz2
            delta_up = self.alpha * sqrt((self.u1_sqr_avg[index1] + epsilon) / (self.g1_sqr_avg[index1] + epsilon)) * dl_dz2 
            self.w1[index1] -= delta_up
            self.u1_sqr_avg[index1] = decay * self.u1_sqr_avg[index1] + (1. - decay) * delta_up * delta_up
            # 3. Full connected layer, backprop gradient from 1st active layer to 1st hidden layer
            for j in range(self.h1):
                # update weights realted to non-zero hidden neurons
                if self.z1[j] == 0.0:
                    continue

                # update weights in 1st hidden layer
                # dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
                dl_dw1 = dl_dz2 * self.z1[j]
                index1 = self.h2 * j + k
                self.g1_sqr_avg[index1] = decay * self.g1_sqr_avg[index1] + (1. - decay) * dl_dw1 * dl_dw1
                delta_up = self.alpha * sqrt((self.u1_sqr_avg[index1] + epsilon) / (self.g1_sqr_avg[index1] + epsilon)) * dl_dw1
                self.w1[index1] -= delta_up
                self.u1_sqr_avg[index1] = decay * self.u1_sqr_avg[index1] + (1. - decay) * delta_up * delta_up

                # 4. Active layer, backprop gradient from 2nd hidden layer to 1st active layer
                # dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
                dl_dz1 = dl_dz2 * self.w1[j * self.h2 + k]
                
                # starting with the bias in the input layer
                # dl/db0 = dl/dz1 * dz1/db0 = dl/dz1 * 1
                index0 = self.n * self.h1 + j
                sigma = ((sqrt(rho2 * self.g0_sqr_sum[index0] + dl_dz1 * dl_dz1) - sqrt(rho2 * self.g0_sqr_sum[index0])) / self.alpha0)
                self.g0_sum[index0] = rho * self.g0_sum[index0]  + dl_dz1 - sigma * self.w0[index0]
                self.g0_sqr_sum[index0] = rho2 * self.g0_sqr_sum[index0] + dl_dz1 * dl_dz1

                # 5. Full connected layer, backprop gradient from 1st hidden layer to input layer
                for i, v in x:
                    # update weights in input layer
                    # dl/dw0 = dl/dz1 * dz/dw0 = dl/dz1 * v
                    dl_dw0 = dl_dz1 * v
                    index0 = self.h1 * i + j
                    sigma = (sqrt(rho2 * self.g0_sqr_sum[index0] + dl_dw0 * dl_dw0) - sqrt(rho2 * self.g0_sqr_sum[index0])) / self.alpha0
                    self.g0_sum[index0] = rho * self.g0_sum[index0] + dl_dw0 - sigma * self.w0[index0]

                    # update history grad squre for weight of edge connecting input to h1 layer
                    self.g0_sqr_sum[index0] = rho2 * self.g0_sqr_sum[index0] + dl_dw0 * dl_dw0
