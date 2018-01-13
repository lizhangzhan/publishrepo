# -*- encoding: utf-8 -*-
import numpy as np
from scipy import stats

import pylab as pl

class GaussianDatasetGenerator:

  def __init__(self, N=1000, mini_batch = 50, mu = 0.0, sigma =1.0):
    self.N = N
    self.mini_batch = mini_batch
    self.mu = mu
    self.sigma = sigma

    self.X = mu + sigma * np.random.randn(N)
  
    # statistic for posterior distribution
    self.post_norm_mu = self.X.mean()
    self.pos_norm_var = 1.0 / N * sigma * sigma
    self.pos_norm_std = np.sqrt(self.pos_norm_var)

    self.true_posterior = stats.norm(self.post_norm_mu, self.pos_norm_std)

  def potential_energy(self, p):
    """
    Potential Energy function w.r.t position (likelihood function given observation X and q)
    
    Parameters:
    ----------
    p :  double or array, position or parameters values of interest
    """
    try:
      return -stat.norm(p, self.sigma).logpdf(self.X).sum()
    except:
      u = [-stats.norm(pi, self.sigma).logpdf(self.X).sum() for pi in p]

      return np.array(u)

  def kinetic_energy(self, v):
    """
    Kinetic Energy function w.r.t momentum, a auxiliary variable

    Parameters:
    ---------
    v : double or array, velocity / momentum value for kinetic energy
    """
    return 0.5 * v * v

  def hamiltonian(self, p, v):
    """
    Hamiltonian ( sum of Potential and Energy function) for given velocity and
    position

    Parameters:
    -----------
    p : double or array, postions values
    v : double or array, velocity values

    Returns:
    --------
    Hamiltonian enery at position p and velocity v
    """
    return self.U(p) + self.K(v)
  
  def dev_potential(self, p):
    """
    Derivative of Potential Energy function given parameters
    
    Parameters:
    ----------
    p :  double or array, position or parameters values of interest
    """
    ids = np.random.permutation(self.N)[:self.mini_batch]
    return 1.0 / self.pos_norm_var * (p - self.X[ids].mean())

  def dev_kinetic(self, v):
    """
    Derivative of Kinetic Energy function auxiliary variables velocity values
    
    Parameters:
    ----------
    v : double or  array, velocity values
    """
    return v;

  def true_posterior_prob(self, thetas):
    """
    probability of sampled parameters in true posterior distribution
    
    Parameters:
    ----------
    thetas : double array, parameters values of interest

    Returns:
    probability of given parameters
    """
    return self.true_posterior.pdf(thetas)

  def true_posterior_pdf(self):
    return true_posterior
  
  def theta_range(self, bin_num):
    return np.linspace(-1.0, 1.0, bin_num)

  
  def bin_density(self, bins):
    p = np.zeros(len(bins) - 1)

    i = 0
    for l, r in zip(bins[:-1], bins[1:]):
      # diff for a bin
      p[i] = true_posterior.cdf(r) - true_posterior.cdf(l)
      i += 1

    return p
  
  def sample_density(self, samples, bins):
    """
      statistic density of given samples
    """
    cnts, bins, patches = pl.hist(samples, bins)
    probs = cnts / float(len(samples))

    return probs

  def prior_gradient(self, p):
    # norm(0, 1) prior
    return 0.5 * p

