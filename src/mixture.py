import numpy as np
import scipy  as sp
from numpy.typing import NDArray
from typing import Iterator, Optional
from typing import List, Optional, Tuple
from costum_typing import  GradModel, Seed, VectorType
import random

class PathMixture:

    def __init__(self,
                n_dim: int,
                minus_log_density_grad: GradModel,
                Ykts:List[VectorType],#history of updates along optimization trajectory
                Skts:List[VectorType],#history of updates of gradients along optimization trajectory
                E: VectorType#initial  values of the diagonal matrix of the  inverse Hessian
                ):
      self._Ykts = Ykts
      self._Skts = Skts
      self._E = E
      self._n_dim = n_dim
      self._trajectory_length = len(Ykts)
      self._minus_log_density_grad = minus_log_density_grad

      self._list_sample_info = self.init()
      self._list_sample_DIV= self.estimate_div()

    def init(self) -> List[dict]:
        list_dicts = [self.init_one_sample(self._Ykts[i], i) for i in range(self._trajectory_length) ]
        return list_dicts

    def init_one_sample(self, x_center,  num_previous_points) -> dict:


        Ykts = self._Ykts[0:num_previous_points]
        Skts = self._Skts[0:num_previous_points]
        m = len(Ykts)

        Dk = sum(Ykts[num_previous_points] * Skts[num_previous_points])
        thetak = sum(Ykts[num_previous_points] ** 2) / Dk # curvature checking
        a = (sum(self._E * Ykts[num_previous_points] ** 2) / Dk)
        E = 1.0 / (a / self._E + Ykts[num_previous_points]**2 / Dk - a * (Skts[num_previous_points] / self._E)**2 / sum(Skts[num_previous_points]** 2 / self._E))

        Rk = np.zeros((num_previous_points, num_previous_points))

        Ykts_matrix = np.matrix(Ykts)
        Skts_matrix = np.matrix(Skts)
        Rk = Ykts_matrix.dot(Skts_matrix.T)

        # Rk will be upper triangular matrix
        Rk = np.triu(Rk.tolist())

        label = "ill_formed"
        log_det_cholesky_Hk = -np.Inf
        Qk = None
        Rk_tilde = None
        Mkbar = None
        Wkbart = None
        theta_D =  None
        cholesky_Hk = None
        try:
            ninvRST = -sp.linalg.solve_triangular( Rk,  Skts_matrix)
            E_mat = np.diag( E)
            temp = Ykts_matrix.dot(E_mat)
            temp2 = np.diag(Dk) +  Ykts_matrix.dot(np.diag(np.sqrt(E))).T
            temp3 = temp2.dot(ninvRST)
            # hessian
            if (2 * m > self._n_dim):
                Hk = np.diag(E) + temp.dot(ninvRST) + ninvRST.dot(temp) + ninvRST.dot(temp3)
                cholesky_Hk = np.linalg.cholesky(Hk)
                log_det_cholesky_Hk = np.linalg.det(cholesky_Hk)
                label = "full"
            else:
                Wkbart = np.concatenate((temp, ninvRST.dot(np.diag(np.sqrt(1.0 /  E)))))
                temp4 = np.c_[np.zeros((m, m)), np.ones((m, m))]
                temp5 = np.c_[np.ones((m, m)), temp2]
                Mkbar = np.concatenate(temp4, temp5)
                Qk, Rkbar = np.linalg.qr(Wkbart.T)
                temp6 = Rkbar.dot(Mkbar)
                n_row_Mkbar = Mkbar.shape[0]
                Rk_tilde = np.linalg.cholesky(temp6.dot(Rkbar) + np.diag(n_row_Mkbar))

                Qk = Qk
                Rk_tilde = Rk_tilde
                log_det_cholesky_Hk = sum(np.log(np.diag(Rk_tilde))) + 0.5 * sum(np.log(E))
                Mkbar = Mkbar
                Wkbart = Wkbart
                theta_D = 1.0 / E
                label = "sparse"
        except Exception as e:
            print(e)
            label = "ill_formed"


        return {"label": label,
                "x_center": x_center,
                "log_det_cholesky_Hk": log_det_cholesky_Hk,
                "cholesky_Hk": cholesky_Hk,
                "Qk": Qk,
                "Rk_tilde": Rk_tilde,
                "Wkbart": Wkbart,
                "theta_D": theta_D
                }

    def estimate_div_for_one_sample(self, sample_pos, n_samples = 5) -> dict:
        # ' estimate divergence based on Monte Carlo samples given the output of

       repeat_draws  = np.zeros((self._n_dim, n_samples))
       fn_draws = np.full(n_samples, np.Inf)
       lp_approx_draws = np.full(n_samples, 0.0)
       if (len(self._list_sample_info) == 0) or (sample_pos >= len(self._list_sample_info)):
           return None
       else:
         sample_info = self._list_sample_info[sample_pos]
         if (sample_info["label"] == "ill_formed"):
           return None
         else:
           draw_idx = 0
           u = np.random.normal(size=self._n_dim)
           while (draw_idx < n_samples):
             if (sample_info["label"] == "full"):

               cholesky_Hk = sample_info["cholesky_Hk"]
               u2 = cholesky_Hk.dot(u) + sample_info["x_center"]
             else: #sample_info["label"] == "sparse"
               Qk = sample_info["Qk"]
               u1 = Qk.dot(u)
               temp = Qk.dot(sample_info["Rktilde"].dot(u1))+(u - Qk.dot(u1)) + sample_info["x_center"]
               u2 = np.diag(np.sqrt(1.0 / sample_info["theta_D"])) * temp

             skip = False
           try:
             f_test_DIV, grad_test_DIV = self._minus_log_density_grad(u2)
           except Exception as e:
             print(e)
             skip = True

           if (skip):
             continue
           else:
             fn_draws[draw_idx] = f_test_DIV
             lp_approx_draws[draw_idx] =  - sample_info["logdetcholHk"] - 0.5 * sum(u **2) -0.5 * self._n_dim * np.log(2 * np.pi)
             repeat_draws[:, draw_idx] = u2
             draw_idx = draw_idx + 1
       ELBO = -np.mean(fn_draws) - np.mean(lp_approx_draws)
       if (np.isnan(ELBO)):
         ELBO = -np.Inf
       DIV = ELBO
       if (np.isnan(DIV) or (np.isinf(DIV))):
         DIV  = -np.Inf


       return {"DIV": DIV,
                "repeat_draws": repeat_draws,
                "fn_draws": fn_draws,
                 "sample_info": sample_info,
                 "lp_approx_draws": lp_approx_draws}
    def estimate_div(self,
                     n_samples: int) -> List[dict]:

        list_dicts = [self.estimate_div_for_one_sample( i) for i in range(self._trajectory_length)]
        return list_dicts

    def sample(self, m):
       # this corresponds to function Form_N_apx
       if (self._label == "ill_formed"):
         return
       else:
         print("---")



       
class Mixture:
    # mixture of Gaussian Approximate distributions with associated weights

    def __init__(self, 
               list_qs,
               list_weights
               ):
        
      self._list_qs = list_qs
      self._list_weights = list_weights


    def sample(self, m):
        pass
        #TODO
  
  
  