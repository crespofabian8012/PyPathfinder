import numpy as np

from typing import List, Optional, Tuple

from costum_typing import LogDensityModel
from costum_typing import DrawAndLogP, GradModel, Seed, VectorType
from approximation_model import ApproximationModel

from  psis import sumlogs, psislw
from random import choices
def importance_sample( n_approx_draws: int,
                       n_draws: int,
                       log_density_grad: GradModel,
                       approx_model: ApproximationModel,
                       seed: Optional[Seed] = None,
                       pareto_smoothed = False):

    rng = np.random.default_rng(seed)
    draws = [approx_model.sample(n_approx_draws, seed) for i in range(n_approx_draws)]

    log_weights  = [log_density_grad.log_density(x)-approx_model.log_density(x) for x in draws["samples"]]

    if (pareto_smoothed):
        weights = psislw(log_weights)

    weights = np.exp(log_weights) / np.exp(sumlogs(log_weights))
    idxs = choices(population=draws, n_draws,
                      woights=weights)

    return draws[idxs]


     
