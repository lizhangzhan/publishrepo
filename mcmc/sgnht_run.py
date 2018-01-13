#! /usr/bin/python 
# -*- encoding: utf-8 -*- 

import numpy as np
import pylab as pl
"""
  Stochastic Gradient Nose-Hoover thermostats dynamics

  key contributes
  1. Introduce Nose-Hoover thermostats dynamics as a proposal
  2. Exact gradient is replaced with stochastic gradient calculated using
  mini-batch dataset
  3. Ignore the expensive MH accept/reject step by introducing a fraction term
  B into the Hamiltonian dynamics. With this fraction term, the model posterior is
  preserved.
  4. Treat fraction term as a part of the dynamics system (an improvement in
  compare with sg-hmc) 

  Compared with sg-hmc, sg-nht can update fraction term adaptively

  [Reference]
  1. Ding, N., etc. "Bayesian Sampling Using Stochastic Gradient thermostats". 2014
  2. Hong Ge, Jes Frellsen. "Scalable MCMC", 2015
  3. Edward Meeds, etc. "Hamiltonian ABC",   2015
"""
def run_sgnht_sampler(dataset, T, h, A, init_pos, init_vel = None, init_frac = None):
  """
    SGNHT sampler

    Parameters:
    ----------
    T :  Integer,     number of iteration
    h :  Double,      step size
    A :  Double,      diffusion factor term
    init_p : Double (array)  initial position
    init_v : Double (array)  initial velocity
    init_frac : Double (array)  initial fraction

    Returns:
     arrays of sampled positions, velocities and fractions(adaptive)
  """
  dev_potential = dataset.dev_potential
  dev_kinetic = dataset.dev_kinetic
  
  pos, vel = init_pos, init_vel
  fraction = init_frac

  if vel is None:
    vel = np.random.randn()
  
  if fraction is None:
    fraction = A

  positions = [pos]
  velocities = [vel]
  fractions = [fraction]
  for t in range(T):
    # compute gradients of loss function  
    grad = dev_potential(pos)

    # update velocity
    vel = vel - fraction * vel * h - grad * h + np.sqrt(2.0 * A) * np.random.normal(0, h)
    
    # update position of interest
    pos = pos + vel * h
    
    # update fraction term adaptively, independent on iteration number
    fraction = fraction + (vel * vel - 1.0) * h
    
    # collect samples
    positions.append(pos)
    velocities.append(vel)
    fractions.append(fraction)

  return np.squeeze(np.array(positions)), np.squeeze(np.array(velocities)), np.squeeze(np.array(fractions))
  
if __name__ == "__main__":
  pl.close('all')
  T       = 10 ** 3
  nburn   = 100
  epsilon = 1e-2
  A       = 1.0  # insensitive to this factor
  bin_num = 100

  init_pos   = 0
  init_vel   = None
  init_frac  = None
  from dataset import GaussianDatasetGenerator

  dataset = GaussianDatasetGenerator(N=100, mini_batch=10)
  
  positions, velocities, fractions = run_sgnht_sampler(dataset, T, epsilon, A, init_pos, init_vel, init_frac)

  pl.figure(1)
  pl.clf()

  pl.plot(dataset.theta_range(bin_num), dataset.true_posterior_prob(dataset.theta_range(bin_num)), 'k--', lw=3)
  pl.hist(positions[nburn:], 20, histtype='step', normed=True, alpha=0.75, linewidth=3)

  ax = pl.axis()
  pl.axis([-0.6, 1, 0, ax[3]])
  pl.title('sgnht-sampler')
  pl.legend(['true', 'sg-nht'])
  #pl.show()
  
  #import pdb;pdb.set_trace()
  pl.figure(2)
  pl.clf()
  pl.plot(fractions, 'k--', lw=1.5)
  pl.plot( fractions.cumsum()/(np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  pl.title('adaptive-fraction')
  pl.legend(['fraction-change', 'fraction-cummean'])
  pl.show()
  print "diff = ", np.abs(dataset.post_norm_mu - positions.mean())
