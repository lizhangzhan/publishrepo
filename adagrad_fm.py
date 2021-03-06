from __future__ import division
import numpy as np

from math import exp, copysign, log, sqrt

class AdaGradFM(object):
    """
    Factorization Machine SGD online Learner

    Attributes:
        n : int, number of features after hashing trick
        k : int, number of factors for interaction
        a : float, initial learning rate
        w0 : float, weight for bias
        c0 : float, counters,
        w : an array of float, feature weights
        c1 : an array of float, counters for weights
        v : array of float, feature weights for factors
        c2 : an array of float, counters for factor weights
    """
    def __init__(self, n, k, a, l2 = 0.001, seed=0):
        """
        Parameters:
        -----------
        n : int, number of features after hashing trick
        k : int, number of factors for interaction
        a : float, initial learning rate
        seed : int, random seed
        """
        self.n = n
        self.k = k
        self.a = a
        self.l2 = l2

        rand = np.random.RandomState(seed)
        # initialize weights, factorized interaction and counts
        self.w0 = 0.0
        self.c0 = 0.0
        
        self.w =  np.zeros((self.n,), dtype=np.float64)
        self.c1 = np.zeros((self.n,), dtype=np.float64)
        
        self.v = np.random.normal(0, 0.01, size=(self.n, self.k))
        self.c2 = np.zeros((self.n, self.k), dtype=np.float64)
        
        self.wx_sum = np.zeros((self.k,), dtype=np.float64)
        self.wx2_sum = np.zeros((self.k,), dtype=np.float64)
    
    def read_sparse(self, path):
        """
        Apply hashing trick to the libsvm format sparse file. 

        Parameters:
        -----------
        path : string, point to libsvm format sparse file

        Returns:
        -----------
        An generator, like, (nozero_feature_idx, feature_val), y
        """
        for line in open(path):
            xs = line.rstrip().split(' ')
            y = int(xs[0])
            nonzeros_indexes = []
            feat_vals = []
            for item in xs[1:]:
                idx, val = item.split(':')
                nonzeros_indexes.append(int(idx))
                feat_vals.append(int(val))
            yield zip(nonzeros_indexes, feat_vals), y

    def predict(self, x):
        """
        Predict probability of instance `x` with postive label

        Parameters
        ----------
        x : a list of tuple, a list of non-zero feature id and val pair

        Returns
        --------
        p : float, a predicted probability for x with postive label
        """
        self.wx_sum = np.zeros((self.k,), dtype=np.float64)
        self.wx2_sum = np.zeros((self.k,), dtype=np.float64)
        
        wx = 0
        for feat_id, feat_val in x:
            wx += self.w[feat_id] * feat_val
            for k in range(self.k):
                self.wx_sum[k] += self.v[feat_id, k] * feat_val
                self.wx2_sum[k] += (self.v[feat_id, k] * feat_val)**2

        p = self.w0 + wx
        for k in range(self.k):
            p += .5 * (self.wx_sum[k] ** 2 - self.wx2_sum[k])

        return sigmoid(p)

    def update(self, x, p, y):
        """
        Update the model with SGD learner.

        Parameters:
        ----------
        x : a list of tuple, like (feat_id, feat_val), nozero feature index and value pair
        err: float, a error between the prediction of the model and target.

        Returns:
        ---------
        weight : array, model weight
        counts: array, model counts
        """
        err = p - y
        
        self.w0 -= self.a / (sqrt(self.c0) + 1) * err
        self.c0 += err * err
        
        for fid, fval in x:
            grad = err * fval
            dl_dw = self.a / (sqrt(self.c1[fid]) + 1) * grad
            self.w[fid] -= dl_dw
            self.c1[fid] = grad * grad
            for f in range(self.k):
                grad = err * fval * (self.wx_sum[f] - self.v[fid, f] * fval)
                try:
                  self.v[fid, f] -= self.a / (sqrt(self.c2[fid, f]) + 1.0) * (grad + self.l2 * self.v[fid, f])
                except:
                  import pdb
                  pdb.set_trace()
                  print "except"
                self.c2[fid, f] += grad*grad
    
    def dump_model(self, dump_file):
      with open(dump_file, 'wb') as fwriter:
        for w in self.w:
          fwriter.write("%s\n" % w)

        for wv in self.v:
          for w in wv:
            fwriter.write("%s\n" % w)

    def plot_w_hist(self, nbins = 50):
        weights = []
        for w in self.w:
          weights.append(w)

        for vi in self.v:
          for w in vi:
            weights.append(w)
        
        import matplotlib.pyplot as plt
        import numpy as np
        hist, bins = np.histogram(weights, bins=50, normed=True)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, label='adagrad-fm', color='green')
        plt.legend()
        plt.show()
    
def sigmoid(x):
    ''' 
    predict the logit
    '''
    return 1. / (1. + exp(- max(min(x, 35.), -35.)))
    

def logLoss(p, y):
    ''' 
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    p = max(min(p, 1. - 1e-15), 1e-15)
    return - log(p) if y == 1. else -log(1. - p)

