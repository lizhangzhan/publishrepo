import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np

np.import_array()

cdef class AdaDelta4MLP:
    """
    Neural Network with a single ReLU hidden layer online learner.

    [Done] Extend AdaDelta with Nesterov's accelerated gradient 
    For AdaDelta, adaptive learning rate method, refer to,
    [1] http://climin.readthedocs.org/en/latest/adadelta.html#id1
    [2] https://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html
    [3] adadelta, http://120.52.73.78/arxiv.org/pdf/1212.5701v1.pdf

    Attributes:
    -----------
        n : integer, number of input units
        h : integer, number of hidden units
        alpha :   double, step rate
        decay :   double, decay rate
        epsilon : double, smooth factor, default a smaller values
        momentum : double, momentum term with Nesterov's accelerated gradient

        l1 : l1 regularization term
        l2 : l2 regularization term
         
        w0 : array, shape(input_num, hidden1_num)
          weights between the input and hidden layers
        
        w1 : array, shape(hidden1_num,)
          weights between the hidden and output layers
        
        z : array, shape(hidden1_num,)
          values in hidden1 layer 
        
        g_sqr_avg: double,
          average of bias history grad squre in hidden layer    
         
        g0_sqr_avg: array, shape(input_num, hidden1_num)
          average of history grad squres for input neurons
        
        g1_sqr_avg: array, shape(hidden1_num,)
          average of history grad squres for hidden neurons
        
        u_avg: double,
          average of bias history update in hidden layer
        
        u0_avg: array, shape(input_num, hidden1_num)
          average of history update for input neurons
        
        u1_avg: array, shape(hidden1_num,)
          average of history update for hidden neurons
        
        u_sqr_avg: double,
          average of bias history update squre in hidden layer
        
        u0_sqr_avg: array, shape(input_num, hidden1_num)
          average of history update squres for input neurons
        
        u1_sqr_avg: array, shape(hidden1_num,)
          average of history update squres for hidden neurons
    """
    cdef unsigned int n     # number of input units
    cdef unsigned int h     # number of hidden units
    
    cdef double alpha       # parameter of per-coordinate step rate
    cdef double decay       # parameter of per-coordinate decay rate
    cdef double epsilon     # smooth factor, default smaller value
    cdef double momentum    # a momentum for Nesterov's accelerated gradient
    cdef double l1          # l1 regularization term 
    cdef double l2          # l2 regularization term 

    cdef double[:] w0       # weights between the input and hidden layers
    cdef double[:] w1       # weights between the hidden and output layers
    
    cdef double[:] z        # hidden units
    
    cdef double g_sqr_avg   # average of grad squre for bias in hidden layer
    cdef double u_avg       # average of history bias update for input units
    cdef double u_sqr_avg   # average of update squre for bias in hidden layer
    
    cdef double[:] u0_avg   # average of history weight update for input units
    cdef double[:] u1_avg   # average of history weight update for hidden units
    
    cdef double[:] g0_sqr_avg   # average of history grads squre for input units
    cdef double[:] g1_sqr_avg   # average of history grads squre for hidden units

    cdef double[:] u0_sqr_avg   # average of history update squre for input units
    cdef double[:] u1_sqr_avg   # average of history update squre for hidden units

    def __init__(self,
                 unsigned int n,
                 unsigned int h=10,
                 double a=0.1,
                 double decay=0.95,
                 double epsilon=1e-6,
                 double momentum=0.,
                 double stdev = 1e-6,
                 double l1 = 0.,
                 double l2 = 1e-3,
                 unsigned int seed=0):
        """
        Multiple Layer perception with a single hidden layer

        Parameter:
        ----------
            n :  integer, number of input units
            h :  integer, number of the hidden units
            a :  double,  step rate
            decay :  double, decay rate
            epsilon : double, smooth factor
            l1 : double, L1 regularization parameter
            l2 : double, L2 regularization parameter
            seed : unsigned integer, random seed
        """
        rng = np.random.RandomState(seed)

        self.n = n
        self.h = h

        self.alpha = a
        self.decay = decay
        self.epsilon = epsilon
        self.momentum = momentum
        self.l1 = l1
        self.l2 = l2

        self.w1 = (rng.rand(self.h + 1) - .5) * stdev
        self.w0 = (rng.rand((self.n + 1) * self.h) - .5) * stdev

        # hidden units in the hidden layer
        self.z = np.zeros((self.h,), dtype=np.float64)
        
        # average of history grad squres
        self.g_sqr_avg = 0.
        self.g1_sqr_avg = (rng.rand(self.h + 1) - .5) * stdev
        self.g0_sqr_avg = (rng.rand((self.n + 1) * self.h) - .5) * stdev
        
        # average of history update
        self.u_avg = 0.
        self.u1_avg = (rng.rand(self.h + 1) - .5) * stdev
        self.u0_avg = (rng.rand((self.n + 1) * self.h) - .5) * stdev

        # average of history update squres
        self.u_sqr_avg = 0.
        self.u1_sqr_avg = np.zeros((self.h,), dtype=np.float64)
        self.u0_sqr_avg = np.zeros(((self.n + 1) * self.h), dtype=np.float64)

    def __repr__(self):
        return ('AdaDelta4MLP(n={}, h={}, alpha={}, decay={}, epsilon={}, \
            momentum={}, l1={}, l2={})').format(
            self.n, self.h, self.alpha, self.decay, self.epsilon, 
            self.momentum, self.l1, self.l2
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
        
        self.w1[self.h] -= self.momentum * self.u_avg
        # Accumulate the bias contribution in the hidden layer
        p = self.w1[self.h]

        # 1. Full connected layer, forward contribution from input layer to hidden1 layer
        for j in range(self.h):
            # accumulate the bias contribution in the input layer
            index = self.n * self.h + j
            self.w0[index] -= self.momentum * self.u0_avg[index]
            self.z[j] = self.w0[index]

            # accumulate the contribution for jth hidden neoron forwarded from input layer
            for i, v in x:
                index = i * self.h + j
                self.w0[index] -= self.momentum * self.u0_avg[index]
                self.z[j] += self.w0[index] * v
        

        # 2. Active layer, apply the ReLU activation function to the 1st hidden inputs
        for j in range(self.h):
            self.z[j] = self.z[j] if self.z[j] > 0. else 0.
       
        # 3. Full connected layer, forward contribution from hidden1 layer to output layer
        for j in range(self.h):
            self.w1[j] -= self.momentum * self.u1_avg[j]
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
        
        cdef double alpha = self.alpha
        cdef double decay = self.decay
        cdef double epsilon = self.epsilon
        cdef double momentum = self.momentum
        cdef double l1 = self.l1
        cdef double l2 = self.l2
        # 1. Full connected layer, backprop gradient from output layer to 1st hidden layer
        dl_dy = e
        
        # update params of the bias in 1st hidden layer(no regularization)
        # dl/db1 = dl/dy * dy/db1 = dl/dy * 1
        self.g_sqr_avg = decay * self.g_sqr_avg + (1. - decay) * dl_dy * dl_dy
        delta_up = alpha * sqrt((self.u_sqr_avg + epsilon)/(self.g_sqr_avg + epsilon)) * dl_dy
        self.w1[self.h] -= delta_up

        delta_up = self.u_avg * momentum + delta_up
        self.u_sqr_avg = decay * self.u_sqr_avg + (1. - decay) * delta_up * delta_up
        self.u_avg = delta_up

        for j in range(self.h):
            # update weights w.r.t non-zero hidden neuron
            if self.z[j] == 0.:
                continue

            # compute grad for jth neuron in the hidden layer
            # dl/dw1 = dl/dy * dy/dw1 = dl/dy * z
            dl_dw1 = (dl_dy * self.z[j] + l1 * np.sign(self.w1[j]) + l2 * self.w1[j])
            self.g1_sqr_avg[j] = decay * self.g1_sqr_avg[j] + (1. - decay) * dl_dw1 * dl_dw1
            delta_up = alpha * sqrt((self.u1_sqr_avg[j] + epsilon)/(self.g1_sqr_avg[j] + epsilon)) * dl_dw1
            self.w1[j] -= delta_up
            
            delta_up = momentum * self.u1_avg[j] + delta_up
            self.u1_sqr_avg[j] = decay * self.u1_sqr_avg[j] + (1. - decay) * delta_up * delta_up
            self.u1_avg[j] = delta_up

            # 2. Active layer, backprop gradient from hidden1 layer to active layer
            # dl/dz = dl/dy * dy/dz = dl/dy * w1
            dl_dz = dl_dy * self.w1[j]
            
            # update params of bias in input layer(no regularization)
            # dl/db0 = dl/dz * dz/db0 = dl/dz * 1
            index = self.n * self.h + j
            self.g0_sqr_avg[index] = self.g0_sqr_avg[index] * decay + (1. - decay) * dl_dz * dl_dz 
            delta_up = alpha * sqrt((self.u0_sqr_avg[index] + epsilon) / (self.g0_sqr_avg[index] + epsilon)) * dl_dz 
            self.w0[index] -= delta_up

            delta_up = momentum * self.u0_avg[index] + delta_up
            self.u0_sqr_avg[index] = decay * self.u0_sqr_avg[index] + (1. - decay) * delta_up * delta_up
            self.u0_avg[index] = delta_up

            # 3. Full connected layer, backprop gradient from active layer to input layer
            # update weights w.r.t non-zero input neuron
            for i, v in x:
                # compute grad for ith neoron in input layer
                # dl/dw0 = dl/dz * dz/dw0 = dl/dz * v
                index = self.h * i + j
                dl_dw0 = (dl_dz * v + l1 * np.sign(self.w0[index]) + l2 * self.w0[index])

                # update params of ith neoron params in the input layer
                self.g0_sqr_avg[index] = self.g0_sqr_avg[index] * decay + (1. - decay) * dl_dw0 * dl_dw0 
                delta_up = alpha * sqrt((self.u0_sqr_avg[index] + epsilon) / (self.g0_sqr_avg[index] + epsilon)) * dl_dw0
                self.w0[index] -= delta_up
                
                delta_up = momentum * self.u0_avg[index] + delta_up
                self.u0_sqr_avg[index] = decay * self.u0_sqr_avg[index] + (1. - decay) * delta_up * delta_up
                self.u0_avg[index] = delta_up 

    def dump_model(self, model_file):
        with open(model_file, 'wb') as writer:
            for w in self.w0:
              writer.write("%s\n" % w)

            for w in self.w1:
              writer.write("%s\n" % w)
