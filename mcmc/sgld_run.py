#! /usr/bin/python 
# -*- encoding: utf-8 -*-

import numpy as np
import pylab as pl
"""
Stochastic gradient Langevin Dynamics (SGLD)
  
  Key points:
  ----------
  1. Introduce Langevin dynamics as a proposal
  2. Exact gradient is replaced with stochastic gradient calculated using
  mini-batch dataset
  3. Update is just SGD plus Gaussian noise. Noise variance is balanced with 
  graident step size. step size decrease to 0 slowly

  [Reference]
  1. Welling and Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics", 2011
  2. Hong Ge, Jes Frellsen. "Scalable MCMC", 2015
"""
def run_sgld_sampler(dataset, T, a, b, gamma, init_theta):
  """
    SGLD sampler

    Parameters:
    ----------
    T :  Integer,     number of iteration
    a :  Double,      a constant term for epsilon computation
    b :  Double,      a constant smooth term for epsilon computation
    gamma : Double,   a constant term for epsilon computation
    init_theta : Double (array)  initial theta

    Returns:
     arrays of sampled theta and epsilon
  """
  #constant term
  #a  = 0.1
  #b  = 100
  #gamma = 0.55
  gen_epsilon = lambda iteration: a * (b + iteration) ** (-gamma) 
  
  dev_loss = dataset.dev_potential

  theta = init_theta
  
  thetas = [theta]
  epsilons  = []
  for t in range(T):
    # compute mini-batch gradients, refer to keypoints[2]
    grad = dev_loss(theta)
    epsilon = np.max(gen_epsilon(t), 1e-4)
    # epsilon = gen_epsilon(t)
    # update parameters, refer to key points[3]
    theta = theta - epsilon * 0.5 * grad + np.random.normal(0, np.sqrt(2 * epsilon))
    
    # collect samples
    thetas.append(theta)
    epsilons.append(epsilon)
  
  return np.squeeze(np.array(thetas)), np.squeeze(np.array(epsilons))


if __name__ == "__main__":
  pl.close('all')
  T        = 10 ** 4
  nburn    = 1000
  a  = 0.1
  b  = 10.0
  gamma = 1.0
  bin_nums = 100
  
  from dataset import GaussianDatasetGenerator

  dataset = GaussianDatasetGenerator()
  init_theta = np.random.normal(0, 1) 
  thetas, epsilons = run_sgld_sampler(dataset, T, a, b, gamma, init_theta)
  # import pdb; pdb.set_trace()
  pl.figure(1)
  pl.clf()

  pl.plot(dataset.theta_range(bin_nums), dataset.true_posterior_prob(dataset.theta_range(bin_nums)), 'k--', lw=3)
  pl.hist(thetas[nburn:], 20, histtype='step', normed=True, alpha=0.75, linewidth=3)

  ax = pl.axis()
  pl.axis([-0.6, 1, 0, ax[3]])
  pl.title('sgld-sampler')
  pl.legend(['true', 'sgld'])
  pl.show()

  print "diff = ", np.abs(dataset.post_norm_mu - thetas.mean())
