from math import exp, copysign, log, sqrt
from datetime import datetime
import random

class FTRLFM(object):
    
    def __init__(self, fm_dim=3, fm_initDev = 0.1, L1 = 1.0, L2=1.0, L1_fm = 1.0,
            L2_fm=1.0, D=2**20, alpha=.15, beta=1.0, alpha_fm = .15, beta_fm =
            1.0, dropoutRate = 1.0):
        ''' initialize the factorization machine.'''
        
        self.alpha = alpha              # learning rate parameter alpha
        self.beta = beta                # learning rate parameter beta
        self.L1 = L1                    # L1 regularizer for first order terms
        self.L2 = L2                    # L2 regularizer for first order terms
        self.alpha_fm = alpha_fm        # learning rate parameter alpha for factorization machine
        self.beta_fm = beta_fm          # learning rate parameter beta for factorization machine
        self.L1_fm = L1_fm              # L1 regularizer for factorization machine weights. Only use L1 after one epoch of training, because small initializations are needed for gradient.
        self.L2_fm = L2_fm              # L2 regularizer for factorization machine weights.
        self.fm_dim = fm_dim            # dimension of factorization.
        self.fm_initDev = fm_initDev    # standard deviation for random intitialization of factorization weights.
        self.dropoutRate = dropoutRate  # dropout rate (which is actually the inclusion rate), i.e. dropoutRate = .8 indicates a probability of .2 of dropping out a feature.
        
        self.D = D
        
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        
        # let index 0 be bias term to avoid collisions.
        self.n = [0.] * (D + 1) 
        self.z = [0.] * (D + 1)
        self.w = [0.] * (D + 1)
        
        self.n_fm = {}
        self.z_fm = {}
        self.w_fm = {}
    
        
    def init_fm(self, i):
        ''' initialize the factorization weight vector for variable i.
        '''
        if i not in self.n_fm:
            self.n_fm[i] = [0.] * self.fm_dim
            self.w_fm[i] = [0.] * self.fm_dim
            self.z_fm[i] = [0.] * self.fm_dim
            
            for k in range(self.fm_dim): 
                self.z_fm[i][k] = random.gauss(0., self.fm_initDev)
    
    def predict_raw(self, x):
        ''' predict the raw score prior to logit transformation.
        '''
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        alpha_fm = self.alpha_fm
        beta_fm = self.beta_fm
        L1_fm = self.L1_fm
        L2_fm = self.L2_fm
        
        # first order weights model
        n = self.n
        z = self.z
        w = self.w
        
        # FM interaction model
        n_fm = self.n_fm
        z_fm = self.z_fm
        w_fm = self.w_fm
        
        raw_y = 0.
        
        # calculate the bias contribution
        for i in [0]:
            # no regularization for bias
            w[i] = (- z[i]) / ((beta + sqrt(n[i])) / alpha)
            
            raw_y += w[i]
        
        # calculate the first order contribution.
        for i in x:
            sign = -1. if z[i] < 0. else 1. # get sign of z[i]
            
            if sign * z[i] <= L1:
                w[i] = 0.
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            
            raw_y += w[i]
        
        len_x = len(x)
        # calculate factorization machine contribution.
        for i in x:
            self.init_fm(i)
            for k in range(self.fm_dim):
                sign = -1. if z_fm[i][k] < 0. else 1.   # get the sign of z_fm[i][k]
                
                if sign * z_fm[i][k] <= L1_fm:
                    w_fm[i][k] = 0.
                else:
                    w_fm[i][k] = (sign * L1_fm - z_fm[i][k]) / ((beta_fm + sqrt(n_fm[i][k])) / alpha_fm + L2_fm)
        
        for i in range(len_x):
            for j in range(i + 1, len_x):
                for k in range(self.fm_dim):
                    raw_y += w_fm[x[i]][k] * w_fm[x[j]][k]
        
        return raw_y
    
    def predict(self, x):
        ''' predict the logit
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x), 35.), -35.)))
    
    def dropout(self, x):
        ''' dropout variables in list x
        '''
        for i, var in enumerate(x):
            if random.random() > self.dropoutRate:
                del x[i]
    
    def dropoutThenPredict(self, x):
        ''' first dropout some variables and then predict the logit using the dropped out data.
        '''
        self.dropout(x)
        return self.predict(x)
    
    def predictWithDroppedOutModel(self, x):
        ''' predict using all data, using a model trained with dropout.
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x) * self.dropoutRate, 35.), -35.)))
    
    def update(self, x, p, y):
        ''' Update the parameters using FTRL (Follow the Regularized Leader)
        '''
        alpha = self.alpha
        alpha_fm = self.alpha_fm
        
        # model
        n = self.n
        z = self.z
        w = self.w
        
        # FM model
        n_fm = self.n_fm
        z_fm = self.z_fm
        w_fm = self.w_fm
        
        # cost gradient with respect to raw prediction.
        g = p - y
        
        fm_sum = {}      # sums for calculating gradients for FM.
        len_x = len(x)
        
        for i in x + [0]:
            # update the first order weights.
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g
            
            # initialize the sum of the FM interaction weights.
            fm_sum[i] = [0.] * self.fm_dim
        
        # sum the gradients for FM interaction weights.
        for i in range(len_x):
            for j in range(len_x):
                if i != j:
                    for k in range(self.fm_dim):
                        fm_sum[x[i]][k] += w_fm[x[j]][k]
        
        for i in x:
            for k in range(self.fm_dim):
                g_fm = g * fm_sum[i][k]
                sigma = (sqrt(n_fm[i][k] + g_fm * g_fm) - sqrt(n_fm[i][k])) / alpha_fm
                z_fm[i][k] += g_fm - sigma * w_fm[i][k]
                n_fm[i][k] += g_fm * g_fm
    
    def write_w(self, filePath):
        ''' write out the first order weights w to a file.
        '''
        with open(filePath, "w") as f_out:
            for i, w in enumerate(self.w):
                f_out.write("%i,%f\n" % (i, w))
    
    def write_w_fm(self, filePath):
        ''' write out the factorization machine weights to a file.
        '''
        with open(filePath, "w") as f_out:
            for k, w_fm in self.w_fm.iteritems():
                f_out.write("%i,%s\n" % (k, ",".join([str(w) for w in w_fm])))
    
    def dump_model(self, dump_file):
      with open(dump_file, 'wb') as fwriter:
        #fwriter.write(len(self.w))
        for w in self.w:
          fwriter.write("%s\n" % w)

        for k, factor_ws in self.w_fm.iteritems():
            for w in factor_ws:
              fwriter.write("%s\n" % w)

    def  plot_w_hist(self, nbins = 50):
        weights = []
        for w in self.w:
          weights.append(w)

        for k, factor_ws in self.w_fm.iteritems():
            for w in factor_ws:
              weights.append(w)
        
        import matplotlib.pyplot as plt
        import numpy as np
        hist, bins = np.histogram(weights, bins=50, normed=True)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, label='ftrl-fm')
        plt.legend();
        plt.show()

    def read_sparse(self, path):
        """
        Apply hashing trick to the libsvm format sparse file. 

        Parameters:
        -----------
        path : string, point to libsvm format sparse file

        Returns:
        -----------
        An generator, like, (nozero_feature_idxes, y)
        """
        for line in open(path):
            xs = line.rstrip().split(' ')
            y = int(xs[0])
            nonzeros_feat_idxes = []
            for item in xs[1:]:
                fidx, _ = item.split(':')
                nonzeros_feat_idxes.append(int(fidx))
            yield nonzeros_feat_idxes, y


def logLoss(p, y):
    ''' 
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    p = max(min(p, 1. - 1e-15), 1e-15)
    return - log(p) if y == 1. else -log(1. - p)
