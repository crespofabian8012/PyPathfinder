import numpy as np
from typing import List, Optional, Tuple
from costum_typing import DrawAndLogP
from costum_typing import  GradModel, Seed, VectorType
from approximation_model import ApproximationModel
from numpy.random import choice
from random import choices
class Mixture:
    # mixture of  approximate distributions models with associated weights

    def __init__(self,
                 list_approxs: List[ApproximationModel],
                 list_weights: VectorType = None
                 ):

        self._list_approxs = list_approxs
        self._num_distrib = len(list_approxs)

        if (list_weights != None):
            if ((sum(list_weights)-1.0) > 0.01):
                #normalize weights
                self._list_weights =  self._list_weights / sum(list_weights)
            else:
                self._list_weights = list_weights
        else:
            self._list_weights = np.full(self._num_distrib, 1.0 / len(self._list_weights))

    def dims(self) -> int:
        if (self._num_distrib >0):
            return  self._list_approxs[0].dims()
        else:
            return 0

    def sample(self,  n: int,
                seed: Optional[Seed] = None) -> VectorType[DrawAndLogP]:

        approximation_draws = choices(population =self._list_approxs, n,
                      woights=self._list_weights)
        samples = [approx.sample() for approx in approximation_draws]

        return samples