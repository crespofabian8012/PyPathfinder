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

    def sample(self) -> VectorType:
       pass
      #TODO
      #use rng to resample from proposal distributions
      


    def importance_resample(self) -> VectorType:
        pass
