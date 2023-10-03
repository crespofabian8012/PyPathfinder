import numpy as np
import scipy  as sp
from numpy.typing import NDArray
from typing import Iterator, Optional
from costum_typing import  GradModel, Seed, VectorType
import random

class GaussianApproxDistrib:

    def __init__(self, 
                x_center: VectorType,
                list_Ykts: list,#history of updates along optimization trajectory
                list_Skts: list,#history of updates of gradients along optimization trajectory
                E#initial diagonal inverse Hessian
                ):
      self._x_center = x_center
      self._list_Ykts = list_Ykts
      self._list_Skts = list_Skts
      self._E = E
      self._n_dim  = x_center.shape[0]
      self._trajectory_length  = len(list_Ykts)
      
      # curvature checking
      Dk = [sum(list_Ykts[i, :] * list_Skts[i,: ]) for i in range(self._trajectory_length  )]
      thetak = [sum(list_Ykts[i, :]**2) / Dk[i] for i in range(self._trajectory_length  )]
      
      Rk = np.zeros((self._trajectory_length , self._trajectory_length ))
      #Rk will be upper triangular matrix 
  #     Rk = np.dot(vecs, mat)

  # for(s in 1:m):
  #   for(i in 1:s):
  #     Rk[i, s] = sum(list_Skts[i, :] * list_Ykts[s, :])
    
  
  #     ninvRST = -backsolve(Rk, Skt_h)
  


    def sample(m):
       
       #TODO
       #this corresponds to function Form_N_apx

       
class Mixture:
    # mixture of Gausiian Approximate distributions with associated weights 

    def __init__(self, 
               list_qs,
               list_weights
               ):
        
      self._list_qs = list_qs
      self._list_weights = list_weights


    def sample(m):
        pass
        #TODO
  
  
  