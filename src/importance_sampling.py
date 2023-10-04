import numpy as np
from numpy.typing import NDArray
from typing import Iterator, Optional
from typing import List, Optional, Tuple

from .costum_typing import LogDensityModel
from .costum_typing import DrawAndLogP, GradModel, Seed, VectorType

class ImportanceSampling:
     

    def __init__(self,
                 target_distrib, 
                 list_proposal_distrib: List,
                 seed: Optional[Seed] = None,
                 use_Pareto_smoothing: bool = False
                 ):
      self._list_proposal_distrib = list_proposal_distrib
      self._N = list_proposal_distrib.shape[0]
      self._rng = np.random.default_rng(seed)

    def sample() -> VectorType:
       pass
      #TODO
      #use rng to resample from proposal distributions
      


    def importance_resample(
          
        ) -> VectorType:
        #TODO
        pass
      # weights = np.exp(
      # np.apply_along_axis(lp, axis=1, arr=thetas)
      #   - np.apply_along_axis(lpminus1, axis=1, arr=thetas)
      #   )
      #   M = thetas.shape[0]
      # # TODO(bward): should use random Generator object
      # idxs = np.random.choice(M, size=M, replace=True, p=weights / weights.sum())

      #return thetas[idxs]  # type: ignore
