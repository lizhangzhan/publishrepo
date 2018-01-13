# -*- encoding: utf-8 -*-
import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm, fmax
cimport numpy as np

np.import_array()

cdef class AdaDelta4BMLP:
    """
    Online Bayesian Neural Network with one ReLU hidden layers.

    Bayesian Network Network with variational inference
    [1] Charles Blundell, etc. Weight uncertainty in neural networks, 2015
    [2] Alex Graves, Practical variational inference in neural networks, 2011

    AdaDelta learning algorithm with Nesterov's accelerated gradient
    [1] http://climin.readthedocs.org/en/latest/adadelta.html#id1
    [2] https://cs.stanford.edu/people/karpathy/convertjs/demo/trainer.html
    [3] AdaDelta, http://120.52.73.78/arxiv.org/pdf/1212.5701v1.pdf

    Performance
    [1] cython performance using def, cdef and cpdef,
        https://notes-on-cython.readthedocs.org/en/latest/classes.html
    [2] cython performance of python, cython, c on a vector
        https://notes-on-cython.readthedocs.org/en/latest/std_dev.html
    """
    cdef unsigned int n         # number of input units in input layer
    cdef unsigned int h1        # number of hidden units in h1 layer

    # parameters for adadelta
    cdef double alpha           # parameter of per-coordinate step rate
    cdef double decay           # parameter of per-coordinate decay rate
    cdef double epsilon         # smooth factor, default a very small values

    cdef double sigma_prior     # sigma of prior distribution over weights and bias

    # all parameter to optimize
    cdef double[:] w0_mu       # means of normal distribution over weights w0
    cdef double[:] w0_logsigma # log sigma of normal distribution over weights w0

    cdef double[:] w1_mu       # means of normal distribution over weight w1
    cdef double[:] w1_logsigma # log sigma of normal distribution over weight w1

    # auxiliary variables to sample weights in train phase
    cdef double[:] epsilon_w0  # epsilon sampling between input and h1 layers
    cdef double[:] epsilon_w1  # epsilon sampling between h1 and output layers

    cdef double[:] w0          # weights sampling between input and h1 layers
    cdef double[:] w1          # weights sampling between h1 and output layers

    # store the values in feedforward process
    cdef double[:] z1          # first hidden units

    # average of history grad square for weights between input and h1 layers
    cdef double[:] mean_g0_sqr_avg
    cdef double[:] logsigma_g0_sqr_avg
    # average of history update square for weights between input and h1 layers
    cdef double[:] mean_u0_sqr_avg
    cdef double[:] logsigma_u0_sqr_avg

    # average of history grad square for weights between h1 and output layers
    cdef double[:] mean_g1_sqr_avg
    cdef double[:] logsigma_g1_sqr_avg
    # average of history update square for weights between h1 and output layers
    cdef double[:] mean_u1_sqr_avg
    cdef double[:] logsigma_u1_sqr_avg

    # regularization
    cdef bint   use_regular
    cdef double l2
    cdef double a
    cdef double b
    cdef double n_batches
    cdef int seed

    def __init__(self,
                unsigned int n,
                unsigned int h1=48,
                double alpha=0.1,
                double epsilon=1e-6,
                double decay=0.95,
                double sigma=1e-3,
                double l2=1e-6,
                unsigned int seed=0
                ):
        """
        Bayesian Multiple layer perception with two hidden layer

        Parameters:
        -----------
         n    : integer, number of neurons in input layer
         h1   : integer, number of neurons in h1 layer
         alpha: double, per-coordinate step rate
         epsilon: double, smooth factor in learning step equation
         decay: double, decay rate
         sigma: double, sigma prior
         l2   : regularization for KL divergence between prior and posterior
         seed : unsigned int, random seed
        """

        #self.rng = np.random.RandomState(seed)
        self.seed = seed

        # hyper-parameters
        self.n  = n
        self.h1 = h1

        self.alpha   = alpha
        self.epsilon = epsilon
        self.decay   = decay

        self.sigma_prior = sigma
        self.l2 = l2

        # initialize main model parameters
        self.w0_mu       = self.init_random(h1 * (n + 1))
        self.w0_logsigma = self.init_random(h1 * (n + 1))


        self.w1_mu       = self.init_random(h1 + 1)
        self.w1_logsigma = self.init_random(h1 + 1)

        self.z1 = np.zeros((self.h1,), dtype=np.float64)

        # initialize auxilary structures for learning optimization algorithm
        self.mean_g0_sqr_avg     = np.zeros((n + 1) * h1, dtype=np.float64)
        self.logsigma_g0_sqr_avg = np.zeros((n + 1) * h1, dtype=np.float64)
        self.mean_u0_sqr_avg     = np.zeros((n + 1) * h1, dtype=np.float64)
        self.logsigma_u0_sqr_avg = np.zeros((n + 1) * h1, dtype=np.float64)

        self.mean_g1_sqr_avg     = np.zeros((h1 + 1,), dtype=np.float64)
        self.logsigma_g1_sqr_avg = np.zeros((h1 + 1,), dtype=np.float64)
        self.mean_u1_sqr_avg     = np.zeros((h1 + 1,), dtype=np.float64)
        self.logsigma_u1_sqr_avg = np.zeros((h1 + 1,), dtype=np.float64)

        self.a = 2.
        self.b = 2.0
        self.n_batches = 0.
        self.use_regular = 0 if l2 == 0.0 else 1

    cdef init_random(self, int size, double stdev=0.01):
       rng = np.random.RandomState(self.seed)
       return np.asarray(rng.normal(0, stdev, size=size))

    cdef normal_sample(self, int shape, double avg, double std):
       rng = np.random.RandomState(self.seed)
       return np.asarray(rng.normal(avg, std, size=shape))

    cdef get_adapative_regular(self):
        """
        Regularization coefficient for KL divergence between prior and posterior
        """
        return fmax(self.l2 / ((self.a + self.n_batches * 0.1) ** self.b), 1e-8)

    cpdef predict_one(self, list input_vector):
        """
        Predict probability of positive label (click) given input vector

        Parameters
        ----------
         x : list, a list of (index, value) of input vector

        Return
        ---------
         a probability of positive label for input vector
        """
        cdef int n = self.n
        cdef int h1 = self.h1

        cdef double[:] w0_mu = self.w0_mu
        cdef double[:] w1_mu = self.w1_mu

        cdef double[:] z1 = self.z1
        cdef double raw_y

        cdef int i
        cdef int j
        cdef int k
        cdef double v
        cdef int index
        cdef noise

        #1. forward input msg from input to h1 layer
        for i in xrange(h1):
            # bias in h1 layer
            index = n * h1 + i
            z1[i] = w0_mu[index]

            for k, v in input_vector:
                index = k * h1 + i
                noise = w0_mu[index] / np.log(1. + np.exp(self.w0_logsigma[index]))
                if noise > 0.0001:
                    z1[i]  += w0_mu[index] * v

            # ReLu activation
            z1[i] = z1[i] if z1[i] > 0. else 0.

        raw_y = w1_mu[h1] # include bias
        for i in xrange(h1):
            noise = w1_mu[i] / np.log(1. + np.exp(self.w1_logsigma[i]))
            if noise > 0.0001:
                raw_y += w1_mu[i] * z1[i]

        return sigm(raw_y)

    cpdef update_one(self, list input_vector, double t, int n_step):
        cdef int i
        cdef double pred_prob

        self.n_batches += 1
        for i in xrange(n_step):
            pred_prob = self.forward(input_vector)
            # print "({0}, {1})".format(pred_prob, t)
            self.backword(input_vector, (pred_prob - t))

    cdef forward(self, list input_vector):
        """
        forward the input vector to gengerte probability of positive label

        Parameters
        ----------
         x : list, a list of (index, value) of input vector

        Return
        ---------
         predicted probability of positive label for input vector
        """
        cdef unsigned int n = self.n
        cdef unsigned int h1 = self.h1
        cdef double sigma_prior = self.sigma_prior

        cdef double[:] epsilon_w0
        cdef double[:] epsilon_w1

        cdef double[:] w0
        cdef double[:] w1

        cdef double[:] z1 = self.z1

        cdef int i
        cdef int j
        cdef int k
        cdef double v
        cdef int index
        cdef double raw_y

        raw_y = 0

        #s1. sample weights for connection between inputs and h1 layers
        epsilon_w0 = self.normal_sample((n + 1) * h1, 0., sigma_prior);
        # element-wise dot
        w0 = self.w0_mu + np.log(1. + np.exp(self.w0_logsigma)) * epsilon_w0

        self.epsilon_w0 = epsilon_w0
        self.w0 = w0

        #s1.2 forward input message from input layer to h1 layer
        for i in xrange(h1):
            # bias in the input layers
            index = h1 * n + i
            z1[i] = w0[index]

            # forward input vector to h1 layers
            for k, v in input_vector:
                index = k * h1 + i
                z1[i]  += w0[index] * v
            # ReLU activation
            z1[i] = z1[i] if z1[i] > 0. else 0.

        #s2. sample weights for connection between h1 and output layers
        epsilon_w1 = self.normal_sample((h1 + 1), 0., sigma_prior);
        w1 = self.w1_mu + np.log(1. + np.exp(self.w1_logsigma)) * epsilon_w1

        self.epsilon_w1 = epsilon_w1
        self.w1 = w1

        #s3.1 accumulate the activations
        raw_y = w1[h1];
        for i in xrange(h1):
            raw_y += w1[i] * z1[i]

        # sigmoid activation function
        return sigm(raw_y)


    cdef compute_h1_grad(self, int index, double dl_dy, double val, double reg):
        return self.compute_grad_impl(dl_dy, val, reg,
                                    self.w1[index], self.w1_mu[index],
                                    self.w1_logsigma[index])

    cdef compute_h0_grad(self, int index, double dl_dy, double val, double reg):
        return self.compute_grad_impl(dl_dy, val, reg,
                                    self.w0[index], self.w0_mu[index],
                                    self.w0_logsigma[index])

    cdef compute_grad_impl(self, double dl_dy,  double val, double reg,
                            double w, double w_mu, double w_logsigma):
        if self.use_regular == 1: # weight regularization
            return (dl_dy * val  + reg * (w / (self.sigma_prior ** 2) -
                (w - w_mu) / np.exp(w_logsigma * 2)))

        return dl_dy * val

    cdef backword(self, list input_vector, double e):
        """
        backpropogation the err info to each layer and update weights

        [NOTE]
        KL divergence between the prior and approximate posterior

        Parameters:
        -----------
        x : list of tuple (k, v), input vector
        e : double, error between the prediction and the target
        """
        cdef unsigned int n = self.n
        cdef unsigned int h1 = self.h1
        cdef double sigma_prior = self.sigma_prior

        # the below structures has been filled in forward phase
        cdef double[:] epsilon_w0 = self.epsilon_w0
        cdef double[:] epsilon_w1 = self.epsilon_w1

        cdef double[:] w0 = self.w0
        cdef double[:] w1 = self.w1

        cdef double[:] z1 = self.z1

        # derivatives for auxiliary variables
        cdef double dl_dy
        cdef double dl_dw0
        cdef double dl_dw1

        cdef double dl_dz1

        cdef double v
        cdef int i
        cdef int j
        cdef int index
        cdef double reg = self.get_adapative_regular()

        # derivative of raw_y
        dl_dy = e

        # mu and logsigma update for bias in h1 layer (no regularization)
        self.update_h1_layer(h1, dl_dy, 0.0)

        # backprop err msg to update weights between h1 and output layer
        for j in xrange(h1):
            # skip zero input
            if z1[j] == 0.0:
                continue
            assert (sigma_prior ** 2) != 0, 'sigma_prior** 2 is 0'
            assert np.exp(self.w1_logsigma[j] * 2) != 0,\
                    'w1_logsigma[{0}] = 0'.format(j)
            #dl_dw1 = self.compute_h1_grad(k, dl_dy, z1[k], reg)
            dl_dw1 = dl_dy * z1[j]
            if self.use_regular:
                dl_dw1 += reg * (w1[j] / (sigma_prior ** 2) -
                    (w1[j] - self.w1_mu[j]) / np.exp(self.w1_logsigma[j] * 2))

            self.update_h1_layer(j, dl_dw1, reg)

            # bias between input and h1 layer (no regularization)
            dl_dz1 = dl_dy * w1[j]
            index = h1 * n + j
            self.update_h0_layer(index, dl_dz1, 0)

            # update mean and logsigma of weights between input and h1 layer
            for i, v in input_vector:
                index = i * h1 + j
                assert ((np.exp(self.w0_logsigma[index] * 2) != 0),
                    'w0_logsigma[{0}] = {1}'.format(index, self.w0_logsigma[index]))
                #dl_dw0 = self.compute_h0_grad(index, dl_dz1, v, reg)
                dl_dw0 = dl_dz1 * v
                if self.use_regular:
                    dl_dw0 += reg * (w0[index] / (sigma_prior ** 2) -
                        (w0[index] - self.w0_mu[index]) / np.exp(self.w0_logsigma[index] * 2))
                self.update_h0_layer(index, dl_dw0, reg)

    cdef update_h1_layer(self, int idx, double dl_dw, double reg):
        self.update_impl(idx, dl_dw, reg,
                        self.w1, self.w1_mu, self.w1_logsigma,
                        self.mean_g1_sqr_avg, self.mean_u1_sqr_avg,
                        self.logsigma_g1_sqr_avg, self.logsigma_u1_sqr_avg,
                        self.epsilon_w1)

    cdef update_h0_layer(self, int idx, double dl_dw, double reg):
        self.update_impl(idx, dl_dw, reg,
                        self.w0, self.w0_mu, self.w0_logsigma,
                        self.mean_g0_sqr_avg, self.mean_u0_sqr_avg,
                        self.logsigma_g0_sqr_avg, self.logsigma_u0_sqr_avg,
                        self.epsilon_w0)

    cdef update_impl(self, int idx, double dl_dw, double reg,
                        double[:] w, double[:] w_mu, double[:] w_logsigma,
                        double[:] mean_g_sqr_avg, double[:] mean_u_sqr_avg,
                        double[:] logsigma_g_sqr_avg, double[:] logsigma_u_sqr_avg,
                        double[:] epsilon_w):
        """
        update means and logsigmas of weights connecting two layers

        Parameters:
        -----------
        Refer to the definition
        """

        # auxiliary variables for adadelta
        cdef double rho        = self.decay
        cdef double epsilon    = self.epsilon
        cdef double alpha      = self.alpha

        # mu derivative refer to the equation in the [1]
        # dl_dw * dw_dw_mu + dl_dw_mu
        dl_dw_mu = dl_dw
        if self.use_regular == 1:
            dl_dw_mu += reg * (w[idx] - w_mu[idx]) / np.exp(w_logsigma[idx] * 2)

        # p1 update gradient average
        mean_g_sqr_avg[idx]= (rho * mean_g_sqr_avg[idx] + (1. - rho) * dl_dw_mu * dl_dw_mu)

        # p2 update step
        delta_upd = (alpha * (sqrt(mean_u_sqr_avg[idx] + epsilon) /
            sqrt(mean_g_sqr_avg[idx] + epsilon)) * dl_dw_mu)

        assert np.abs(delta_upd) < 1.0, \
                "delta_upd is bigger, {0}".format(delta_upd)
        # p3 update weight
        w_mu[idx] -= delta_upd

        # p4 update update average
        mean_u_sqr_avg[idx] = (rho * mean_u_sqr_avg[idx] +
                    (1. - rho) * delta_upd * delta_upd)

        # logsigma derivatives
        # dl_dw * dw_dw_logsigma + dl_dw_logsigma
        assert np.abs(epsilon_w[idx]) < 1., "epsilon_w[{0}] is bigger, \
                {1}".format(idx, epsilon_w[idx])
        dl_dw_logsigma = dl_dw * epsilon_w[idx] / (1 + np.exp(-w_logsigma[idx]))
        if self.use_regular == 1:
            dl_dw_logsigma += reg * ((w[idx] - w_mu[idx])**2 / np.exp(w_logsigma[idx] * 2) - 1.0)

        assert np.abs(dl_dw_logsigma) < 10.0, \
                "dl_dw_logsigma is bigger, {0}".format(dl_dw_logsigma)
        logsigma_g_sqr_avg[idx] = (rho * logsigma_g_sqr_avg[idx] +
                (1. - rho) * dl_dw_logsigma * dl_dw_logsigma)

        delta_upd = (alpha * (sqrt(logsigma_u_sqr_avg[idx] + epsilon) /
                sqrt(logsigma_g_sqr_avg[idx] + epsilon)) * dl_dw_logsigma)

        assert np.abs(delta_upd) < 1.0, \
                "delta_upd is bigger, {0}".format(delta_upd)
        w_logsigma[idx] -= delta_upd
        logsigma_u_sqr_avg[idx] = (rho * logsigma_u_sqr_avg[idx] +
                (1. - rho) * delta_upd * delta_upd)

    cdef _check_value(self, value, min_v, max_v):
        if (value < min_v) or (max_v > max_v):
            raise Exception("value = {0}, out of bound, [{1}, \
                    {2}]".format(min_v, max_v))

    cdef _check_values(self, values, min_v, max_v):
        for i in xrange(len(values)):
            if ((values[i] < min_v) or (values[i] > max_v)):
               raise Exception("values[{0}] = {1}, out of bound, [{2}, \
                       {3}]".format(values, i, min_v, max_v));

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

            y = float(xs[0])
            idx = []
            val = []
            for item in xs[1:]:
                i, v = item.split(':')
                idx.append(int(i) % self.n)
                val.append(float(v))

            yield zip(idx, val), y


    #[TODO]
    cpdef dump_model(self, str path):
        with open(path, 'wb') as writer:
            for w in self.w0_mu:
                writer.write('%s\n', w)
            for w in self.w1_mu:
                writer.write('%s\n', w)

    def __repr__(self):
        return ('AdaDelta4BMLP(n={}, h1={}, alpha={}, epsilon={}, \
                decay={}, sigma_prior={}, l2={})'.format(self.n, self.h1, self.alpha,
                self.epsilon, self.decay, self.sigma_prior, self.l2))

