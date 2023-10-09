from mixture import Mixture
import importance_sampling
from optimization_path import OptimPath
from costum_typing import  GradModel, Seed, VectorType
from typing import Iterator, Optional
import numpy as np
from importance_sampling import importance_sample
def pathfinder(num_paths: int,
               n_dim: int,
               init_bound: float,
               log_density_grad: GradModel,
               n_approx_draws: int,
               n_draws: int,
               seed: Optional[Seed] = None
                ):

    init_points = [-1*init_bound + 2*init_bound* np.random.random(n_dim) for i in range(num_paths) ]
    optim_paths = [OptimPath(init_points[i], init_bound=init_bound,
                             log_density_grad=log_density_grad, seed=seed) for i in range(num_paths)]

    path_trajectories = [ path.optim_path() for path in optim_paths]
    objective_fun = optim_paths[0].get_objective_function_grad()

    path_approximations = []
    # path_approximations = [PathApproximation(n_dim=n_dim,
    #                                 x_centers=X,
    #                                 minus_log_density_grad=objective_fun,
    #                                 Ykts=Ykt_history,
    #                                 Skts=Skt_history,
    #                                 list_flags=list_flags) for zip() ]

    aprox_model = Mixture( list_approxs=path_approximations)
    imp_sample = importance_sample(n_approx_draws=n_approx_draws,
                      n_draws=n_draws,
                      log_density_grad=log_density_grad,
                      approx_model=aprox_model,
                      seed=seed,
                      pareto_smoothed = False)
    return imp_sample

