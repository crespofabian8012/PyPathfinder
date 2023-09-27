import numpy as np
from numpy.typing import NDArray
from typing import Iterator, Optional

from .typing import LogDensityModel
from .typing import DrawAndLogP, GradModel, Seed, VectorType

class N3_mult_IS:
     

    def __init__(self, 
                 list_proposal_distrib: list,
                 seed: Optional[Seed] = None,
                 use_Pareto_smoothing: bool = False
                 ):
      self._list_proposal_distrib = list_proposal_distrib
      self._N = list_proposal_distrib.shape[0]
      self._rng = np.random.default_rng(seed)

    def sample() -> VectorType:
       
      #TODO
      #use rng to resample from proposal distributions
      


    def importance_resample(
          
        ) -> VectorType:
        #TODO
      weights = np.exp(
      np.apply_along_axis(lp, axis=1, arr=thetas)
        - np.apply_along_axis(lpminus1, axis=1, arr=thetas)
        )
        M = thetas.shape[0]
      # TODO(bward): should use random Generator object
      idxs = np.random.choice(M, size=M, replace=True, p=weights / weights.sum())

      return thetas[idxs]  # type: ignore
