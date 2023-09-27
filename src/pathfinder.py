import numpy as np
import scipy  as sp
from numpy.typing import NDArray
from typing import Iterator, Optional

from .typing import LogDensityModel
from .typing import DrawAndLogP, GradModel, Seed, VectorType
from typing import Callable, Iterator
from numpy.typing import ArrayLike
from scipy import optimize
import hmc

DensityFunction = Callable[[VectorType], float]
Kernel = Callable[[VectorType, DensityFunction], ArrayLike]

class Pathfinder:
  # One path Pathfinder from an initial point init_point
   
  def __init__(self, 
               init_point: VectorType,
               init_bound: float,#initial bound of random initials for each dimension
               fn:  DensityFunction,
               grad: GradModel,
               number_iter: int,
               explore_hmc_from_initial: bool, #logical; if TRUE, generate Hamiltonian search path from initials and reinitialize 
               #the Pathfinder randomly along the search path
               seed: Optional[Seed] = None):
    self._init_point = init_point
    self._n_dim = init_point.shape[0]
    self._fn = fn
    self._grad = grad
    self._seed = seed
    self._rng = np.random.default_rng(seed)
    self._init_bound = init_bound
    self._number_iter = number_iter
    self._num_fn_eval = 0
    self._num_grad_eval = 0
  
    self._explore_hmc_from_initial = explore_hmc_from_initial
  

  def update_init_from_hmc(self, stepsize, steps):
    hmc =  hmc(model= self.grad,
                stepsize = stepsize,
                steps = steps,
                init  = self._init_point,
                seed = self._seed
               )
    theta, logp = hmc.sample()
    self._init_point = theta
    self._logp = logp

  def optim_path(self) -> VectorType:

    if (self._explore_hmc_from_initial):
      #run HMC and take the initil point as the result os HMC sampling 
      update_init_from_hmc(stepsize = 0.005, steps = 800 )
    else:
      Y = 	np.empty((1, self._n_dim +1),float)
      Y[1, ] = self._init_point
      init_grad = self._grad(self._init_point)
      
      if (np.isnan(init_grad).any()):
        Y[1, ] = self.reinitialize()
      
      opt_result = sp.optimize.maximize(fun = (self._fn, self._grad), x0 = self._init_point, method ='L-BFGS-B', 
                           jac = True,  
                           tol = 0.00001,  options = {'xatol': 1e-8,'maxiter': self._number_iter,'disp': True}})
      
      opt_trajectory = opt_result.x
      
      self._num_fn_eval, self._num_grad_eval =  opt_result.nfev, opt_result.njev, opt_result.nhevint
      return (opt_trajectory,opt_result.fun, opt_result.jac, opt_result.hess )


  def reinitialize(max_ntries= 30):
    LBFGS_fail = True
    current_try = 1

    while(LBFGS_fail &  current_try < max_ntries ):
      proposal =  -1*self._init_bound + 2*self._init_bound* np.random.random(self._n_dim) 
      prop_grad = self._grad(proposal)
      if (np.isnan(prop_grad).any()):
        LBFGS_fail = True
    return proposal

  def check_codition(Yk,
                      Sk)-> bool:
    #' check whether the updates of the optimization path should be used in the 
    #' inverse Hessian estimation or not
    Dk = sum(Yk * Sk)
    if (Dk == 0):
      return False
    else:
      #TODO 




 