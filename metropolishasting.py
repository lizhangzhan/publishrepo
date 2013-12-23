#! /usr/bin/python
# -*- coding: utf-8 -*-
import random
import time
from itertools import groupby

"""
   @author lizhangzhan@outlook.com
    
   I am used to take a long time to learn the mcmc algorithm. For a layman,
   I have experienced how difficult to understand and apply it flexibly.
   Although I can not fully understand this algorithm util now, I hope I can
   share some my thoughts with you.
   References:
   [1] http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
   [2] http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/
   [3] 11th chapter, Pattern Recognition and Machine Learning
   
   This is toy example of sampling from a discrete distribution.
   For the example of continues distribution, please refer to
   http://www.ece.sunysb.edu/~zyweng/MCMCexample.html
"""

def delta2sum(v1, v2):
   return sum(map(lambda x, y: (x - y)**2, v1, v2))

class metroplishasting(object):
    def __init__(self, pi, transProbs):
        assert(len(pi) == len(transProbs))
        self.N = len(pi)
        self.pi = pi # the prob distribution sampled from
        self.transProbs = transProbs # transition probability matrix
        self.maxIter = 1000 
        random.seed(time.time())
    
    # Initilize a prob distibrution stochastically
    def __initprob(self):
        p = [random.random() for i in range(len(self.transProbs))]
        norm = sum(p)
        for i in range(len(p)):
            p[i] /= norm
        return p
    
    # Compute the staionary distribution of the proposal transition matrix
    # This statinary distribtion is possible to be inconsistent with the target
    # distribution sampled from or unrelated with it at all.
    def convergeprob(self):
        n = self.N
        p = self.__initprob()
        up = [0.0 / n for i in range(n)] 
        for t in range(self.maxIter):
            up = [0.0 / n for i in range(n)] 
            for c in range(self.N):
                for r in range(self.N):
                    up[c] += p[r] * self.transProbs[r][c]
            if delta2sum(p, up) < 1e-5:
                break
            p = up[:]
             
        return p

    # Get the next proposal state stochastically
    def __next(self, prob):
        cpd = prob[:]
        cpd.insert(0, 0)

        u = random.random()
        for i in range(1, len(prob) + 1):
            cpd[i] += cpd[i-1]
            if cpd[i] > u:
                return (i - 1) 
        return len(prob)

    # the probabiltiy of accepting a new state
    def __alpha(self, x, y):
        return min(self.pi[y] * self.transProbs[y][x] / (self.pi[x] * self.transProbs[x][y]), 1.0)

    def sample1d(self, num):
        x = 0
        samples = [x]
        for t in range(num):
            y = self.__next(self.transProbs[x])
            alpha = self.__alpha(x, y)
            u = random.random()
            if u < alpha:
                x = y
            
            samples.append(x)
        return samples

def main():
    N = 500000
    # the expected target distribution
    pi = [0.6, 0.3, 0.1]

    # a proposal transition matrix
    transProbs = [[0.8, 0.15, 0.05],[0.4, 0.5, 0.1], [0.6, 0.3, 0.1]]
    
    simulator = metroplishasting(pi, transProbs)
    sp = simulator.convergeprob()
    print "The stationary distribution of the proposal transition matrix:"
    for i in xrange(len(sp)):
        print "prob(%d) = %f" % (i, sp[i])

    samples = simulator.sample1d(N)
    counter = [len(list(group)) for key, group in groupby(sorted(samples))]
    print "predicted distribution:"
    for i in xrange(len(pi)):
        print "prob(%d) = %f" % (i, 1.0 * counter[i] / N)
    
    print "expected distribution:"
    for i in xrange(len(pi)):
        print "prob(%d) = %f" %(i, pi[i])

if __name__ == "__main__":
    main()
