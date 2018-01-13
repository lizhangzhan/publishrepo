#! /usr/bin/python 
# -*- encoding: utf-8 -*-

import numpy as np
import pylab as pl
"""
Stochastic gradient Decent (SGD)  -- Baseline
"""
def run_sgd_optimizer(dataset, T, init_theta):
  """
    SGD Optimizer

    Parameters:
    ----------
    T :  Integer,     number of iteration
    init_theta : Double (array)  initial theta

    Returns:
      optimized thetas
  """
  gen_epsilon = lambda iteration: 1.0 / (10. + iteration)
  
  dev_loss = dataset.dev_potential

  theta = init_theta
  
  thetas = [theta]
  epsilons  = []
  for t in range(T):
    # compute mini-batch gradients, refer to keypoints[2]
    grad = dev_loss(theta)
    epsilon = gen_epsilon(t)
    # update parameters, refer to key points[3]
    theta = theta - epsilon * 0.5 * grad
    
    thetas.append(theta)
  
  return np.squeeze(np.array(thetas))


if __name__ == "__main__":
  pl.close('all')
  T        = 1000
  bin_nums = 100

  from dataset import GaussianDatasetGenerator

  dataset = GaussianDatasetGenerator()
  init_theta = np.random.normal(0, 1) 
  thetas = run_sgd_optimizer(dataset, T, init_theta)
  # import pdb; pdb.set_trace()
  pl.figure(1)
  pl.clf()

  pl.plot(dataset.theta_range(bin_nums), dataset.true_posterior_prob(dataset.theta_range(bin_nums)), 'k--', lw=3)
  pl.plot(thetas[-10:], dataset.true_posterior_prob(thetas[-10:]), 'ro')
  pl.plot(thetas[-1:], dataset.true_posterior_prob(thetas[-1:]), 'bs')

  ax = pl.axis()
  pl.axis([-0.6, 1, 0, ax[3]])
  pl.title('sgd-Optimizer')
  pl.legend(['true', 'thetas-last-10', 'theta-last-1'])
  pl.show()

  print "diff = ", np.abs(dataset.post_norm_mu - thetas[-1])
