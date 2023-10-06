import numpy as np
import scipy  as sp

from typing import List, Optional, Tuple
from costum_typing import  GradModel, Seed, VectorType
from itertools import compress
import random

class PathApproximation:

    def __init__(self,
                n_dim: int,
                minus_log_density_grad: GradModel,
                x_centers,
                Ykts:List[VectorType],#history of parameter updates along optimization trajectory
                Skts:List[VectorType],#history of updates of gradients along optimization trajectory
                list_flags: List,
                num_samples_estimate_div: int = 5,
                J :int = 6,# size of history to approximate  the inverse Hessian
                E: VectorType = None#initial  values of the vector in  the diagonal matrix of the  inverse
                 # Hessian at the initial point
                ):
      self._Ykts = Ykts
      self._Skts = Skts
      self._x_centers = x_centers
      self._list_flags = list_flags
      self._J = J
      self._num_samples_estimate_div = num_samples_estimate_div

      if (E is None):
        self._E = np.full(n_dim, 1.0)
      else:
        self._E = E
      self._n_dim = n_dim
      self._trajectory_length = len(Ykts)
      self._minus_log_density_grad = minus_log_density_grad

      self._list_diag_inv_hessian_approximations = self.init_diag_inv_hessian_approximations()
      self._list_approximations = self.init_approximations()
      self._list_approximations_with_DIV= self.estimate_div(self._num_samples_estimate_div)

    def init_diag_inv_hessian_approximations(self) -> VectorType:
        res = [None] * self._trajectory_length
        res[0] = self._E
        for i in range(1, len(self._list_flags)):
            if (self._list_flags[i]):
                res[i] = self.build_init_diag_inv_hessian(res[i-1], self._Ykts[i], self._Skts[i])
            else:
                res[i] = res[i-1]
        return res

    def init_approximations(self) -> List[dict]:
        list_dicts = [self.init_approximation_on_position(i) for i in range(1,self._trajectory_length) ]
        return list_dicts

    def init_approximation_on_position(self,  pos) -> dict:

        x_center = self._x_centers[pos]

        previous_points_idxs = self._list_flags[:pos]
        previous_points_idxs = self._list_flags[:pos] + [False]*(self._trajectory_length-pos)
        # if the number previous points  that  hold the condition   is greater than J, take the last J
        if (sum(previous_points_idxs) > self._J):
            idxs_hold_condition = list(compress(range(len(previous_points_idxs)), previous_points_idxs))
            last_idxs_hold_condition = idxs_hold_condition[-self._J:]
            previous_points_idxs = [ False]*self._trajectory_length
            for pos in last_idxs_hold_condition:
               previous_points_idxs[pos] = True

        Ykts = np.array(list(compress(self._Ykts, previous_points_idxs)))
        Skts = np.array(list(compress(self._Skts, previous_points_idxs)))
        m = len(list(compress(Ykts, previous_points_idxs)))
        if (m== 0):
            return None

        E = self._list_diag_inv_hessian_approximations[pos]
        comb = np.multiply(Ykts, Skts)
        Dk = list(map(sum, comb))
        thetak = np.true_divide(list(map(sum, np.multiply(Ykts, Ykts))), Dk)

        Rk = np.zeros((m, m))

        Ykts_matrix = np.matrix(Ykts)
        Skts_matrix = np.matrix(Skts)
        Rk = Ykts_matrix.dot(Skts_matrix.T)

        # Rk will be upper triangular matrix
        Rk = np.triu(Rk.tolist())#this corresponds to E matrix in algorithm 4 in the paper

        label = "ill_formed"
        log_det_cholesky_Hk = -np.Inf
        Qk = None
        Rk_tilde = None
        Mkbar = None
        Wkbart = None
        theta_D =  None
        cholesky_Hk = None
        try:
            ninvRST = -sp.linalg.solve_triangular(Rk,  Skts_matrix)
            E_mat = np.diag(E)
            temp = Ykts_matrix.dot(E_mat)
            p = Ykts_matrix.dot(np.diag(np.sqrt(E)))
            temp2 = np.diag(Dk) + p.dot(p.T)
            temp3 = temp2.dot(ninvRST)
            # hessian
            if (2 * m > self._n_dim):
                Hk = np.diag(E) + temp.dot(ninvRST) + ninvRST.dot(temp) + ninvRST.dot(temp3)
                cholesky_Hk = np.linalg.cholesky(Hk)
                log_det_cholesky_Hk = np.linalg.det(cholesky_Hk)
                label = "full"
            else:
                Wkbart = np.concatenate((temp, ninvRST.dot(np.diag(np.sqrt(1.0 /  E)))))
                temp4 = np.c_[np.zeros((m, m)), np.identity(m)]
                temp5 = np.c_[np.identity(m), temp2]
                Mkbar = np.concatenate((temp4, np.asarray(temp5)), axis=0)
                Qk, Rkbar = np.linalg.qr(Wkbart.T)
                temp6 = Rkbar.dot(Mkbar)
                n_row_Mkbar = Mkbar.shape[0]
                Rk_tilde = np.linalg.cholesky(temp6.dot(Rkbar.T) + np.identity(n_row_Mkbar))

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

    def estimate_div_for_approximation_on_pos(self, pos, n_samples) -> dict:
        # ' estimate divergence based on Monte Carlo samples given the output of

       repeat_draws  = np.zeros((self._n_dim, n_samples))
       fn_draws = np.full(n_samples, np.Inf)
       lp_approx_draws = np.full(n_samples, 0.0)
       if (len(self._list_approximations) == 0) or (pos >= len(self._list_approximations)):
           return None
       else:
         approximation_info = self._list_approximations[pos]
         if (approximation_info  is None) or (approximation_info["label"] == "ill_formed"):
           return None
         else:
           draw_idx = 0
           u = np.random.normal(size=(self._n_dim,1))
           while (draw_idx < n_samples):
             if (approximation_info["label"] == "full"):

               tcholesky_Hk = approximation_info["cholesky_Hk"].T
               u2 = tcholesky_Hk.dot(u) + approximation_info["x_center"]
             else: #approximation_info["label"] == "sparse"
               tQk = approximation_info["Qk"].T
               Qk = approximation_info["Qk"]
               u1 = tQk.dot(u)
               tRktilde = approximation_info["Rk_tilde"].T
               temp = Qk.dot(tRktilde.dot(u1))+(u - Qk.dot(u1))
               x = np.sqrt(1.0 / approximation_info["theta_D"])
               u2 = np.diag(x).dot(temp) + approximation_info["x_center"].T

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
               lp_approx_draws[draw_idx] =  - approximation_info["log_det_cholesky_Hk"] - 0.5 * sum(u **2) -0.5 * self._n_dim * np.log(2 * np.pi)
               repeat_draws[:, draw_idx] = np.squeeze(np.asarray(u2))
               draw_idx = draw_idx + 1
       ELBO = -np.mean(fn_draws) - np.mean(lp_approx_draws)
       if (np.isnan(ELBO)):
         ELBO = -np.Inf
       DIV = ELBO
       if (np.isnan(DIV) or (np.isinf(DIV))):
         DIV  = -np.Inf

       return   {"DIV": DIV,
                "repeat_draws": repeat_draws,
                "fn_draws": fn_draws,
                "approximation_info": approximation_info,
                "lp_approx_draws": lp_approx_draws}
    def estimate_div(self,
                     n_samples: int) -> List[dict]:

        list_dicts = [self.estimate_div_for_approximation_on_pos( i, self._num_samples_estimate_div) for i in range(self._trajectory_length)]
        return list_dicts


    def build_init_diag_inv_hessian(self, E0, update_theta, update_grad):

        # ' Form the initial diagonal inverse Hessian in the optim path update
        # '
        # ' @param E0       initial diagonal inverse Hessian before updated
        # ' @param update_theta       update in parameters
        # ' @param update_grad      update in gradient
        # '
        # ' @return

        Dk = sum(update_theta * update_grad)
        thetak = sum(update_theta ** 2) / Dk
        a = (sum(E0 * update_theta ** 2) / Dk)
        E = 1 / (a / E0 + update_theta ** 2 / Dk - a * (update_grad / E0) ** 2 / sum(update_grad ** 2 / E0))
        return E
    def get_best_approximation(self) -> Tuple:
        list_DIVs = [ap_DIV["DIV"] for ap_DIV in self._list_approximations_with_DIV]
        pos_max = np.max(list_DIVs)
        return(pos_max, self._list_approximations_with_DIV[pos_max])

    def sample_from_approximation(self,
                                  pos:int,
                                  n_samples:int ) -> dict:

        if (pos<0 or pos >= len(self._list_approximations_with_DIV)):
            return None

        approximation_with_DIV  = self._list_approximations_with_DIV[pos]
        approximation_info = approximation_with_DIV["approximation_info"]


        if (approximation_with_DIV is None) or (approximation_info["label"] == "ill_formed"):
            return None
        else:
            approximation_info = self._list_approximations[pos]

            u = np.matrix(np.random.normal(size=self._n_dim* n_samples).reshape(self._n_dim, n_samples))
            if (approximation_info["label"] == "full"):
                tcholesky_Hk = approximation_info["cholesky_Hk"].T
                u2 = tcholesky_Hk.dot(u) + approximation_info["x_center"]
                cholesky_Hk = approximation_info["cholesky_Hk"]
            else:  # approximation_info["label"] == "sparse"
                tQk = approximation_info["Qk"].T
                Qk = approximation_info["Qk"]
                u1 = tQk.dot(u)
                tRktilde = approximation_info["Rk_tilde"].T
                temp = Qk.dot(tRktilde.dot(u1)) + (u - Qk.dot(u1))
                x = np.sqrt(1.0 / approximation_info["theta_D"])
                u2 = np.diag(x).dot(temp) + approximation_info["x_center"].T

            lp_approx_draws = - approximation_info["log_det_cholesky_Hk"] - 0.5 * sum(
                        u ** 2) - 0.5 * self._n_dim * np.log(2 * np.pi)

            return {"samples": u2,
                    "lp_draws": lp_approx_draws}


    def extract_list_approximations(self):
        pass
        # TODO

  
  
  