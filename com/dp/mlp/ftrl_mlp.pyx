import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np

np.import_array()

cdef class FTRL4MLP:
    """
    Neural Network with a single hidden layer online learner.

    Attributes:
    -----------
        n : integer, number of input units
        h : integer, number of hidden units

        node_type : string, type of activation function in hidden layer

        alpha : double, learning rate
        beta :  double, learning rate
        decay : double, decay rate
        l1 :    double, L1 regularization parameter
        l2 :    double, L2 regularization parameter
        
        w0 : array, shape(input_num, hidden1_num)
          weights between the input and hidden layers
        
        w1 : array, shape(hidden1_num,)
          weights between the hidden and output layers
        
        z : array, shape(hidden1_num,)
          values in hidden1 layer 
        
        g_sum : double
          sum of grad for bias in hidden layer
        
        g_sqr_sum: double
          sum of grad squre for bias in hidden layer
        
        g0_sum: array, shape(input_num, hidden1_num)
          sum of history grads for input units
        
        g1_sum: array, shape(hidden1_num,)
          sum of history grads for hidden units
        
        g0_sqr_sum: array, shape(input_num, hidden1_num)
          sum of history grad squre for input units
        
        g1_sqr_sum: array, shape(hidden1_num,)
          sum of history grad for hidden units
    """
    cdef unsigned int n     # number of input units
    cdef unsigned int h     # number of hidden units
    cdef char* node_type    # type of neoron activation function

    cdef double alpha       # parameter of per-coordinate learning rate
    cdef double beta        # parameter of per-coordinate learning rate
    cdef double decay       # parameter of per-coordinate decay rate
    cdef double l1          # L2 regularization parameter
    cdef double l2          # L2 regularization parameter
    
    cdef double[:] w0       # weights between the input and hidden layers
    cdef double[:] w1       # weights between the hidden and output layers
    
    cdef double[:] z        # hidden units
    
    cdef double g_sum       # sum of grad for bias in hidden layer
    cdef double g_sqr_sum   # sum of grad squre for bias in hidden layer
    
    cdef double[:] g0_sum   # sum of history grads for input units
    cdef double[:] g1_sum   # sum of history grads for hidden units

    cdef double[:] g0_sqr_sum       # sum of history grad squre for input units
    cdef double[:] g1_sqr_sum       # sum of history grad squre for hidden units

    def __init__(self,
                 unsigned int n,
                 unsigned int h=10,
                 char* node_type='relu',
                 double a=0.1,
                 double b=1.0,
                 double d=1.0,
                 double l1=0.,
                 double l2=0.,
                 double stdev = 1e-6,
                 unsigned int seed=0):
        """
        Multiple Layer perception with a single hidden layer

        Parameter:
        ----------
            n :  integer, number of input units
            h :  integer, number of the hidden units
            a :  double, initial learning rate
            b :  double, initial learning rate
            d :  double, decay rate
            l1 : double, L1 regularization parameter
            l2 : double, L2 regularization parameter
            seed : unsigned integer, random seed
        """
        rng = np.random.RandomState(seed)

        self.n = n
        self.h = h
        self.node_type = node_type

        self.alpha = a
        self.beta = b
        self.decay = d
        self.l1 = l1 # unused
        self.l2 = l2
        
        self.w1 = (rng.rand(self.h + 1) - .5) * stdev
        self.w0 = (rng.rand((self.n + 1) * self.h) - .5) * stdev

        # hidden units in the hidden layer
        self.z = np.zeros((self.h,), dtype=np.float64)
        
        # sum of history grad
        self.g_sum = 0.
        self.g1_sum = (rng.rand(self.h + 1) - .5) * stdev
        self.g0_sum = (rng.rand((self.n + 1) * self.h) - .5) * stdev

        # sum of history grad squres
        self.g_sqr_sum = 0.
        self.g1_sqr_sum = np.zeros((self.h,), dtype=np.float64)
        self.g0_sqr_sum = np.zeros(((self.n + 1) * self.h), dtype=np.float64)

    def __repr__(self):
        return ('FTRL4MLP(n={}, h={}, alpha={}, beta={}, decay={}, l1= {}, l2={})').format(
            self.n, self.h, self.alpha, self.beta, self.decay, self.l1, self.l2
        )

    def read_sparse(self, path):
        """Read a libsvm format sparse file line by line.

        Parameter:
        ----------
            path (str): a file path to the libsvm format sparse file

        Returns: 
        --------
            A yield generator, ( list of (idx, val), label)
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
                idx.append(int(i) % self.n)
                val.append(float(v))

            yield zip(idx, val), y
    
    def active_forward(self, double z, str node_type='relu'):
        if node_type == 'relu':
            ret = z if z > 0. else 0.
        elif node_type == 'sigmoid':
            z = np.max((-25., np.min((25., z))))
            ret = 1. / (1. + np.exp(-1.0 * z))
        elif node_type == 'tanh':
            ret = np.tanh(z)
        return ret 
    
    def active_backward(self, double x, str node_type='relu'):
        if node_type == 'relu':
          ret = np.sign(x)
        elif node_type == 'sigmoid':
          ret = x * (1 - x)
        elif node_type == 'tanh':
          ret = 1. - np.squre(x)
        return ret

    def predict_one(self, list x):
        """
        Predict for features.

        Parameter:
        ---------
            x (list of tuple): a list of (index, value) of non-zero features

        Returns:
        --------
            p : double, a prediction for input features
        """
        cdef double p
        cdef int j
        cdef int i
        cdef double v

        # Accumulate the bias contribution in the hidden layer
        self.w1[self.h] = -1.0 * self.g_sum / ((self.beta + sqrt(self.g_sqr_sum)) / self.alpha)
        p = self.w1[self.h]

        # 1. Full connected layer, forward contribution from input layer to hidden1 layer
        for j in range(self.h):
            # accumulate the bias contribution in the input layer
            index = self.n * self.h + j
            self.w0[index] = -1.0 * self.g0_sum[index] / ((self.beta + sqrt(self.g0_sqr_sum[index])) / self.alpha)
            self.z[j] = self.w0[index]

            # accumulate the contribution for jth hidden neoron forwarded from input layer
            for i, v in x:
                index = i * self.h + j
                self.w0[index] = -1.0 * self.g0_sum[index] / (self.l2 + (self.beta + sqrt(self.g0_sqr_sum[index])) / self.alpha)
                self.z[j] += self.w0[index] * v
        

        # 2. Active layer, apply the ReLU activation function to the 1st hidden inputs
        for j in range(self.h):
            #self.z[j] = self.z[j] if self.z[j] > 0. else 0.
            self.z[j] = self.active_forward(self.z[j], node_type=self.node_type) 
        # 3. Full connected layer, forward contribution from hidden1 layer to output layer
        for j in range(self.h):
            self.w1[j] = -1.0 * self.g1_sum[j] / (self.l2 + (self.beta + sqrt(self.g1_sqr_sum[j])) / self.alpha)
            p += self.w1[j] * self.z[j]

        # 4. apply the sigmoid activation function to the output probability
        return sigm(p)
    
    def update_one(self, list x, double e):
        """
        Update the model with one observation.

        Parameter:
        ----------
            x : array-like,
              a list of (index, value) of non-zero features
            e : double
              error between the prediction of the model and target

        Returns:
        --------
            updated model parameter
        """
        cdef int j
        cdef int i
        cdef double dl_dy
        cdef double dl_dz
        cdef double dl_dw1
        cdef double dl_dw0
        cdef double v
        
        cdef double rho = self.decay
        cdef double rho2 = rho * rho
        # 1. Full connected layer, backprop gradient from output layer to 1st hidden layer
        dl_dy = e
        
        # update params of the bias in 1st hidden layer
        # dl/db1 = dl/dy * dy/db1 = dl/dy * 1
        sigma = (sqrt(rho2 * self.g_sqr_sum + dl_dy * dl_dy) - sqrt(rho2 * self.g_sqr_sum)) / self.alpha
        self.g_sum = rho * self.g_sum + dl_dy - sigma * self.w1[self.h]
        self.g_sqr_sum = rho2 * self.g_sqr_sum + dl_dy * dl_dy
        for j in range(self.h):
            # update weights w.r.t non-zero hidden neuron
            if self.z[j] == 0.:
                continue

            # compute grad for jth neuron in the hidden layer
            # dl/dw1 = dl/dy * dy/dw1 = dl/dy * z
            dl_dw1 = dl_dy * self.z[j]
            sigma = (sqrt(rho2 * self.g1_sqr_sum[j] + dl_dw1 * dl_dw1) - sqrt(rho2 * self.g1_sqr_sum[j])) / self.alpha
            self.g1_sum[j] = rho * self.g1_sum[j] + dl_dw1 - sigma * self.w1[j]
            
            # update history grad squre sum for the hidden unit j
            self.g1_sqr_sum[j] = rho2 * self.g1_sqr_sum[j] + dl_dw1 * dl_dw1
             
            # 2. Active layer, backprop gradient from hidden1 layer to active layer
            # dl/dz = dl/dy * dy/dz = dl/dy * w1
            dl_dz = dl_dy * self.w1[j]
            dl_dz *= self.active_backward(self.z[j], node_type=self.node_type)

            # update params of bias in input layer
            # dl/db0 = dl/dz * dz/db0 = dl/dz * 1
            index = self.n * self.h + j
            sigma = (sqrt(rho2 * self.g0_sqr_sum[index] + dl_dz * dl_dz) - sqrt(rho2 * self.g0_sqr_sum[index])) / self.alpha
            self.g0_sum[index] = rho * self.g0_sum[index] + dl_dz - sigma * self.w0[index]
            self.g0_sqr_sum[index] = rho2 * self.g0_sqr_sum[index] + dl_dz * dl_dz

            # 3. Full connected layer, backprop gradient from active layer to input layer
            # update weights w.r.t non-zero input neuron
            for i, v in x:
                # compute grad for ith neoron in input layer
                # dl/dw0 = dl/dz * dz/dw0 = dl/dz * v
                dl_dw0 = dl_dz * v

                # update params of ith neoron params in the input layer
                index = self.h * i + j
                sigma = (sqrt(rho2 * self.g0_sqr_sum[index] + dl_dw0 * dl_dw0) - sqrt(rho2 * self.g0_sqr_sum[index])) / self.alpha
                self.g0_sum[index] = rho * self.g0_sum[index] + dl_dw0 - sigma * self.w0[index]

                # update history grad squre sum for the input i
                self.g0_sqr_sum[index] = rho2 * self.g0_sqr_sum[index] + dl_dw0 * dl_dw0
