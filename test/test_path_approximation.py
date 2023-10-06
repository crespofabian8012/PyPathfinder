import os
import numpy as np

import pytest
from cmdstanpy import cmdstan_path, CmdStanModel
import bridgestan as bs
from pathlib import Path

from typing import Any

import sys
sys.path.append(os.path.join(Path(__file__).parent.parent, "src"))

from typing import Any
from  optimization_path import OptimPath
from  path_approximation import PathApproximation

CURRENT_DIR = os.path.realpath(os.path.dirname(__file__))
class TestPathApproximation:


    def test_path_approx_funnel100(self):
        n_dim = 100
        init_bound = 10.0
        init_theta = np.ones(n_dim)

        parent_path = Path(__file__).parent
        funnel100_dir = os.path.join(parent_path.parent, 'example', 'funnel100')
        stan_file = os.path.join(funnel100_dir, 'funnel100.stan')

        bs_model = bs.StanModel.from_stan_file(stan_file, model_data=None)
        path = OptimPath(
            init_point=init_theta,
            init_bound=init_bound,
            log_density_grad=bs_model.log_density_gradient,
            number_iter=1000000,
            explore_hmc_from_initial=False,
            seed=12345)
        X, G, F, Ykt_history, Skt_history, list_flags  = path.optim_path()
        objective_fun = path.get_objective_function_grad()

        path_approx = PathApproximation(n_dim= n_dim,
                                        x_centers = X,
                                        minus_log_density_grad= objective_fun,
                                        Ykts = Ykt_history,
                                        Skts = Skt_history,
                                        list_flags = list_flags)
