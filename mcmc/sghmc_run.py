#! /usr/bin/python 
# -*- encoding: utf-8 -*-

import numpy as np
import pylab as pl
"""
Stochastic gradient Hamiltonian Monte Carlo Algorithm

  Key contributes:
  1. Introduce Hamiltonian dynamics as a proposal
  2. Exact gradient is replaced with stochastic gradient calculated using
  mini-batch dataset
  3. Ignore the expensive MH accept/reject step by introducing a fraction term
  B into the Hamiltonian dynamics. With this fraction term, the model posterior is
  preserved.

  Unsolved issues:
  1. In proctise, fraction term B is assumed to be known, leading to a
  troublesome issue to practitioners

  [Reference]
  1. Chen,T., E.Fox, and C.Guestrin. "Stochastic Gradient Hamiltonian Monte
  Carlo", ICML, 2014
  2. Hong Ge, Jes Frellsen. "Scalable MCMC", 2015
"""
def run_sghmc_sampler(dataset, T, h, C, Bhat, init_p, init_v = None):
  """
    SGHMC sampler

    Parameters:
    ----------
    T :  Integer,     number of iteration
    h :  Double,      step size
    C :  Double,      fraction constant term (user-specified)
    Bhat : Double,    estimated B
    init_p : Double (array)  initial positions
    init_v : Double (array)  initial velocities

    Returns:
     arrays of sampled positions and velocities 
  """
  dev_potential = dataset.dev_potential
  dev_kinetic = dataset.dev_kinetic
  
  pos, vel = init_p, init_v

  if vel is None:
    vel = np.random.randn()

  positions = [pos]
  velocities = [vel]
  
  # analysis of noise fraction change
  fraction = 1.0
  fractions = [fraction]
  for t in range(T):
    pos = pos + vel * h
    grad = dev_potential(pos)
    vel = vel - C * vel * h - grad * h + np.random.normal(0, 2.0 * (C - Bhat) * h)

    # update fraction term adaptively, independent on iteration number
    fraction = fraction + (vel * vel - 1.0) * h
    
    positions.append(pos)
    velocities.append(vel)
    fractions.append(fraction)

  return np.squeeze(np.array(positions)), np.squeeze(np.array(velocities)), np.squeeze(np.array(fractions))


if __name__ == "__main__":
  pl.close('all')
  T        = 10 ** 4
  nburn    = 1000
  epsilon  = 1e-2
  C        = 6.0 # sensitive to this factor
  Bhat     = 0.0

  bin_nums = 100
  init_pos   = 0
  init_vel   = None
  from dataset import GaussianDatasetGenerator

  dataset = GaussianDatasetGenerator(N=100, mini_batch=10)
  
  positions, velocities, fractions = run_sghmc_sampler(dataset, T, epsilon, C, Bhat, init_pos, init_vel)

  pl.figure(1)
  pl.clf()

  pl.plot(dataset.theta_range(bin_nums), dataset.true_posterior_prob(dataset.theta_range(bin_nums)), 'k--', lw=3)
  pl.hist(positions[nburn:], 20, histtype='step', normed=True, alpha=0.75, linewidth=3)

  ax = pl.axis()
  pl.axis([-0.6, 1, 0, ax[3]])
  pl.title('sghmc-sampler')
  pl.legend(['true', 'sg-hmc'])
  pl.show()
  
  pl.figure(2)
  pl.clf()
  pl.plot(fractions, 'k--', lw=1.5)
  pl.plot( fractions.cumsum()/(np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  pl.title('fraction')
  pl.legend(['fraction-change', 'fraction-cummean'])
  pl.show()
  print "diff = ", np.abs(dataset.post_norm_mu - positions.mean())
