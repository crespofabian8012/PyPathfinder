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

class CallbackFunctor:
    def __init__(self, obj_f_and_grad):
        self.intermediate_fun_vals = []
        self.intermediate_sols = []
        self.intermediate_grad_vals = []
        self.num_calls = 0
        self.obj_fun_grad = obj_f_and_grad
      
    
    def __call__(self, x):
        fun_val, grad_val = self.obj_fun_grad(x)
        self.num_calls += 1
        self.intermediate_sols.append(x)
        self.intermediate_fun_vals.append(fun_val)
        self.intermediate_grad_vals.append(grad_val)
   
    def save_sols(self, filename):
        sols = np.array([sol for sol in self.intermediate_sols])
        np.savetxt(filename, sols)

class Pathfinder:
  #Find the best multivariate normal approximation encountered while maximizing a log density.

 # From an optimization trajectory from an initial point init_point, Pathfinder constructs a sequence of (multivariate normal)
 # approximations to the distribution specified by a log density function. The approximation
 # that maximizes the evidence lower bound (ELBO), or equivalently, minimizes the KL divergence
 # between the approximation and the true distribution, is returned.

 # The covariance of the multivariate normal distribution is an inverse Hessian approximation
 # constructed using at most the previous `history_length` steps.
   
  def __init__(self, 
               init_point: VectorType,
               init_bound: float,#initial bound of random initials for each dimension
               fn:  DensityFunction,
               grad: GradModel,
               number_iter: int,
               explore_hmc_from_initial: bool, # if TRUE, generate Hamiltonian search path from initials and reinitialize 
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
    def f_and_grad(x):
      return (fn(x), grad(x))
    self._f_and_grad= f_and_grad
  
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

  def optim_path(self, method= "L-BFGS-B") -> VectorType:
    # or method ='L-BFGS-B', "trust-ncg", ‘Nelder-Mead’, ‘trust-exact’ ,‘trust-constr’,‘trust-krylov’

    if (self._explore_hmc_from_initial):
        self.update_init_from_hmc(stepsize = 0.005, steps = 800 )
    else:
      Y = 	np.empty((1, self._n_dim +1),float)
      Y[1, :] = self._init_point
      init_grad = self._grad(self._init_point)
      
      if (np.isnan(init_grad).any()):
        Y[1, : ] = self.reinitialize()

      cb = CallbackFunctor(self._f_and_grad)
      opt_result = sp.optimize.maximize(fun = self._fn_and_grad, x0 = self._init_point, method = method.
                           jac = True,  
                           tol = 0.00001,  options = {'xatol': 1e-8,'maxiter': self._number_iter,'disp': True}}, callbacl = cb)
      
      print(opt_result.message)
      opt_trajectory = cb.intermediate_sols
      num_iter = len(cb.intermediate_sols)
     

      X =  np.asmatrix(np.reshape(cb.intermediate_sols, (len(cb.intermediate_sols),self._n_dim)))#intermediate points 
      G = np.asmatrix(np.reshape(cb.intermediate_grad_vals, (len(cb.intermediate_grad_vals),self._n_dim)))#intermediate gradients vectors
      F = np.asmatrix(np.reshape(cb.intermediate_fun_vals, (len(cb.intermediate_fun_vals),1)))#intermediate function vals 
      
      Ykt = X[2:, :] - X[:-1,:]
      Skt = G[2:, :] - G[:-1,:]

      y= np.c_[X,F]

      list_flags = [self.check_condition(Ykt[i,:], Skt[i,:]) for i in range(num_iter)]
      list_true_cond = np.where(list_flags)


      # estimate DIV for all approximating Gaussians and save results

      
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

  def check_condition(update_theta,
                      update_grad)-> bool:
    #' check whether the updates of the optimization path should be used in the 
    #' inverse Hessian estimation or not
    Dk = sum(update_theta * update_grad)
    if (Dk == 0):
      return False
    else:
       thetak = sum(update_theta**2) / Dk  # curvature checking
       if((Dk <= 0) or  (abs(thetak) > 1e12)):
         return False
       else:
         return True
    
    
  def build_init_diag_inv_hessian(E0, update_theta, update_grad):
  
    #' Form the initial diagonal inverse Hessian in the L-BFGS update 
    #' 
    #' @param E0       initial diagonal inverse Hessian before updated
    #' @param update_theta       update in parameters 
    #' @param update_grad      update in gradient 
    #' 
    #' @return 
  
    Dk = sum(update_theta * update_grad)
    thetak = sum(update_theta**2) / Dk   
    a = (sum(E0 * update_theta**2) / Dk)
    E = 1 / (a / E0 + update_theta**2 / Dk - a * (update_grad / E0)^2 / sum(update_grad**2 / E0))
    return E




 