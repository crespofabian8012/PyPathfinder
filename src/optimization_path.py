import numpy as np
import scipy as sp

from numpy.typing import NDArray
from numpy.typing import ArrayLike
from typing import Callable
from typing import  Optional

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from costum_typing import  GradModel, Seed, VectorType
from scipy import optimize


DensityFunction = Callable[[VectorType], float]
Kernel = Callable[[VectorType, DensityFunction], ArrayLike]


class SaveIntermediateValues:
    '''the optimization path from L-BFGS(or other method) including all parameter values visited during optimization"""
    '''
    def __init__(self, obj_f_and_grad) -> None:
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

class OptimPath:
    '''the optimization path from L-BFGS(or other method) including all parameter values visited during optimization"""
    '''
    def __init__(self,
               init_point: VectorType,
               init_bound: float,#initial bound of random initials for each dimension
               log_density_grad: GradModel,
               number_iter: int = 1000,
               explore_hmc_from_initial: bool = False, # if TRUE, generate Hamiltonian search path from initials and reinitialize
               #the optimization path randomly along the search path
               J: int = 6,# number of points used to calculate the
               seed: Optional[Seed] = None):
        self._init_point = init_point
        self._n_dim = init_point.shape[0]
        self._log_density_grad = log_density_grad
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._init_bound = init_bound
        self._number_iter = number_iter
        self._num_fn_eval = 0
        self._num_grad_eval = 0
        self._explore_hmc_from_initial = explore_hmc_from_initial

        def minus_log_density_grad(x):
            log_dens, log_grad = self._log_density_grad(x, jacobian=False)
            return (-log_dens, -log_grad)

        self._minus_log_density_grad = minus_log_density_grad
  
    def get_objective_function_grad(self):
         return  self._minus_log_density_grad


    def optim_path(self, method= "L-BFGS-B"):
     # or method ='L-BFGS-B', "trust-ncg", ‘Nelder-Mead’, ‘trust-exact’ ,‘trust-constr’,‘trust-krylov’

            Y = 	np.zeros((1, self._n_dim +1))
            Y[0][0:self._n_dim] = self._init_point
            init_log_dens, init_grad = self._minus_log_density_grad(self._init_point)

            if (np.isnan(init_grad).any()):
                Y[0][0:self._n_dim] = self.reinitialize()

            init_tuple = self._minus_log_density_grad(self._init_point)
            cb = SaveIntermediateValues(self._minus_log_density_grad)

            opt_result = sp.optimize.minimize(fun = self._minus_log_density_grad, x0 = self._init_point, method = method,
                           jac = True,  
                           tol = 0.00001,  options = {'maxiter': self._number_iter,'disp': True}, callback = cb)

            opt_trajectory = cb.intermediate_sols
            num_iter = len(cb.intermediate_sols)

            X =  np.asmatrix(np.reshape(cb.intermediate_sols, (len(cb.intermediate_sols),self._n_dim)))#intermediate points
            G = np.asmatrix(np.reshape(cb.intermediate_grad_vals, (len(cb.intermediate_grad_vals),self._n_dim)))#intermediate gradients vectors

            fun_vals = cb.intermediate_fun_vals
            fun_vals = [init_log_dens] + fun_vals
            F = np.asmatrix(np.reshape(fun_vals, (len(fun_vals),1)))#intermediate function vals

            X = np.concatenate((np.matrix(self._init_point), X))
            G = np.concatenate((np.matrix(init_grad), G))

            Ykt = X[1:, :] - X[:-1,:]
            Skt = G[1:, :] - G[:-1,:]

            y= np.c_[X,F]

            list_flags = [self.check_condition( np.squeeze(np.asarray(Ykt[i,:])),  np.squeeze(np.asarray(Skt[i,:]))) for i in range(Ykt.shape[0])]

            list_true_cond = np.where(list_flags)
            E0 = np.ones(self._n_dim)

            Ykt_history = [np.squeeze(np.asarray(Ykt[i,:])) for i in list_true_cond[0]]
            Skt_history = [ np.squeeze(np.asarray(Skt[i,:])) for i in list_true_cond[0]]

            return (X, G, F, Ykt_history, Skt_history, list_flags )


    def reinitialize(self, max_ntries= 30):
        optim_fail = True
        current_try = 1

        while(optim_fail &  current_try < max_ntries ):
            proposal =  -1*self._init_bound + 2*self._init_bound* np.random.random(self._n_dim)
            prop_theta, prop_grad = self._log_density_grad(proposal)
            if (np.isnan(prop_grad).any()):
                optim_fail = True
        return proposal

    def check_condition(self, update_theta,
                      update_grad)-> bool:
        ''' check whether the updates of the optimization path should be used in the
        inverse Hessian estimation or not
        '''
        Dk = sum(update_theta * update_grad)
        if (Dk == 0):
            return False
        else:
            thetak = sum(update_theta**2) / Dk  # curvature checking
            if((Dk <= 0) or  (abs(thetak) > 1e12)):
                return False
            else:
                return True




