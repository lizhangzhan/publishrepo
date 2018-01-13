#! /usr/bin/python 
import numpy as np
import time
  
"""
   hamiltonian monte carlo sampling can achieve higher accuracy than sgd in logistic regression,
   at expense of more time and computation cost 
"""

class Options:
  def __init__(self, n_samples, n_leaps, step_size):
    self.n_samples = n_samples
    self.n_leaps = n_leaps
    self.step_size = step_size

class Params:
  def __init__(self, X, y, sigma, num_leaps, step_size):
    self.X = X
    self.y = y
    self.sigma = sigma # std deviation of Gaussian prior
    self.num_leaps = num_leaps # leapfrog number
    self.step_size = step_size

def sigmoid(z):
  return 1.0 / (1. + np.exp(-z))

def log_norm_pdf(values, mean, stdev):
  """
  Parameters:
  -----------
    values : double array, shape=(D, 1)
    mean:    double array, shape=(D, 1)
    stdev:   double array(diagnal matrix), shape=(D, 1)
  Returns:
    log of weights in probability in normal distribution N(mean, stdev)
  """
  (D, _) = values.shape
  ret = (-1. * np.ones((D, 1)) * (0.5 * np.log(2 * np.pi * stdev)) - \
      ((values - mean)**2 / (2 * np.ones((D, 1)) * stdev)))

  return ret

def bayes_neglogloss(weights, params, fit_bias=True):
  """
    negative log likelihood analog to potential energy function
    Parameters:
    -----------
    params : struct with the data
      X and y to fit, sigma for std deviation of Gaussian prior
    
    weights: model weights, shape=(D, 1)

    Returns:
    --------
    log joint probability of the model
  """
  X = params.X # shape=(N, D)
  y = params.y # shape=(N, 1)
  sigma = params.sigma
  (N, D) = X.shape
  
  # add bias dim 
  if fit_bias:
    X = np.hstack((np.ones((N, 1)), X))
  
  # log Gaussian prior
  logp = np.sum(log_norm_pdf(weights, np.zeros((D+1, 1), dtype=np.float64),
    params.sigma * np.ones((D+1, 1), dtype=np.float64)))

  raw_y = np.dot(X, weights)
  logp += np.dot(y.T, raw_y) - np.sum(np.log(1. + np.exp(raw_y)))
  
  return logp * -1.0

def bayes_grad_neglogloss(weights, params, fit_bias=True):
  """
    Parameters:
    -----------
    params : struct with the data
      X and y to fit, l2 for l2 regularization
    
    weights: model weights, shape=(D, 1)

    Returns:
    --------
    grad of model log loss
  """
  X = params.X
  y = params.y
  sigma = params.sigma
  (N, D) = X.shape
  # add bias 
  if fit_bias:
    X = np.hstack((np.ones((N, 1)), X))

  raw_y = np.dot(X, weights)
  # grad of log Gaussian prior
  grad_logp = (-1. / sigma) * weights
  grad_logp += np.dot(X.T, (y - sigmoid(raw_y)))

  return grad_logp * -1.0

def kinetic(v):
  """
    kinetic energy function
  """
  return np.sum(np.dot(v.T, v)) / 2

def check_nan(values):
  if np.any(np.isnan(values)):
    print "nan exception"
    return True
  return False

def hmc_lr(params, num_samples, seed=0):
  """
    logistic regression with hamiltonian monte carlo sampling
  """
  rng = np.random.RandomState(seed)
  max_leaps = params.num_leaps
  max_step_size = params.step_size

  (N, D) = params.X.shape
  thetas = rng.randn((D + 1), 1)
  logp = bayes_neglogloss(thetas, params)
  grad = bayes_grad_neglogloss(thetas, params)
  
  sample_thetas = None
  accept_rate = 0.
  for i in xrange(num_samples):
    p = rng.randn((D + 1), 1) # column vector (include bias)
    h = kinetic(p) + logp
    
    thetas_new = thetas 
    grad_new = grad
    # Random leapfrog
    num_leaps = int(np.ceil(rng.rand() * max_leaps))
    step_size = max_step_size
    
    # leapfrog method to simulate discretization of hamiltonian dynamics
    p = p - 0.5 * step_size * grad_new # half step in momentum
    for t in xrange(num_leaps):
      # one step in position
      thetas_new = thetas_new + step_size * p
      if check_nan(thetas_new):
        break
      # one step in momentum
      grad_new = bayes_grad_neglogloss(thetas_new, params)
      p = p - step_size * grad_new # half step in momentum
      if check_nan(grad_new):
        break
    p = p - 0.5 * step_size * grad_new # half step in momentum

    accept = False
    logp_new = bayes_neglogloss(thetas_new, params)
    h_new = kinetic(p) + logp_new
    delta_h = h_new - h
    if (min(1, np.exp(-delta_h)) > rng.rand()):
      accept = True
      grad = grad_new
      logp = logp_new
      thetas = thetas_new
      accept_rate += 1.
    #if not accept:
    #  continue
    if sample_thetas is None:
      sample_thetas = thetas
    else:
      sample_thetas = np.hstack((thetas, sample_thetas))
  
  print "accept_rate in hmc = %f" % (accept_rate / num_samples)
  return sample_thetas

def check_hmc_lr_fit(X, y):
  """
    summary lr with hmc in time cost and logloss 
  """
  num_burnin = 100
  start = time.time() 
  params = Params(X, y, sigma=0.01, num_leaps=10, step_size=0.01)
  sample_thetas = hmc_lr(params, 1000)
  #import pdb;pdb.set_trace()
  thetas_mean = np.mean(sample_thetas[:, num_burnin:], axis=1)
  end = time.time()
  print "params for lr with hmc: %s" % thetas_mean
  print "cost time %f seconds" % (end - start)
  
  logp = bayes_neglogloss(thetas_mean.reshape(3, 1), params)
  print "logloss=%f" % logp

def check_sgd_lr_fit(X, y):
  """
    summary lr with sgd in time cost and logloss 
  """
  from sklearn.linear_model import LogisticRegression
  alpha = 0.01
  start = time.time()
  model = LogisticRegression(C=0.01, fit_intercept=True)
  model.fit(X, y)
  end = time.time()
  print "params for lr with sgd: %s" % np.hstack((model.intercept_, model.coef_[0]))
  print "cost time %f seconds" % (end - start)
  
  probs = model.predict_proba(X)
  logprob = 0
  for i in xrange(len(y)):
    logprob += np.log(probs[i][y[i]])
  print "logprob=%f" % logprob
   
if __name__ == "__main__":
  from sklearn import datasets
  iris = datasets.load_iris()
  idx = [i for i in xrange(iris.target.shape[0]) if iris.target[i] in [0, 1]]
  X = iris.data[idx, :2]
  y = iris.target[idx]
  
  # experiment comparasion in accuracy and time cost
  check_sgd_lr_fit(X, y)
  print "====================="
  check_hmc_lr_fit(X, np.reshape(y, (X.shape[0], 1)))
