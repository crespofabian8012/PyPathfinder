import numpy as np
from typing import List, Optional, Tuple
class Mixture:
    # mixture of Gaussian Approximate distributions with associated weights

    def __init__(self,
                 list_distrib: List,
                 list_weights: List = None
                 ):

        self._list_qs = list_distrib
        self._num_distrib = len(list_distrib)
        if (list_weights != None):
          self._list_weights = list_weights
        else:
          self._list_weights = np.full(self._num_distrib, 1.0 / len(self._list_weights))



    def sample(self, m):
        pass
        # TODO