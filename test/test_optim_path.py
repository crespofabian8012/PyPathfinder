import functools

import os
import numpy as np

import pytest
from cmdstanpy import cmdstan_path, CmdStanModel
import bridgestan as bs
from pathlib import Path

import sys
sys.path.append(os.path.join(Path(__file__).parent.parent, "src"))

from typing import Any
from  optimization_path import OptimPath

CURRENT_DIR = os.path.realpath(os.path.dirname(__file__))

# class TestOptimPath:
#
#
#   def test_creation_optim_path_funnel100(self):
#     #bs.set_bridgestan_path("C:\\Users\\FC7458\\.bridgestan\\bridgestan-2.2.0")
#     #set BRIDGESTAN="C:\Users\FC7458\Downloads\bridgestan\"
#     init_bound = 10.0
#     init_theta =  -1*init_bound + 2*init_bound* np.random.random(100)
#
#
#     parent_path =Path(__file__).parent
#     funnel100_dir = os.path.join(parent_path.parent , 'example', 'funnel100')
#     stan_file = os.path.join(funnel100_dir, 'funnel100.stan')
#
#     bs_model = bs.StanModel.from_stan_file(stan_file, model_data = None)
#     num_param = bs_model.param_num()
#
#     np.testing.assert_equal(num_param, 100)
#
#     theta_unc = init_theta
#     lp, grad = bs_model.log_density_gradient(theta_unc, jacobian=False)
#
#
#     pathfinder = OptimPath(
#                  init_point = init_theta,
#                  init_bound = init_bound,
#                  log_density_grad = bs_model.log_density_gradient,
#                  number_iter = 1000,
#                  explore_hmc_from_initial = False,
#                  seed = 12345)
#
#     np.testing.assert_array_equal(pathfinder._init_point, init_theta)
#     np.testing.assert_equal(pathfinder._init_bound, init_bound)
#
#
#   def test_optim_path(self):
#     init_bound = 10.0
#     #init_theta =  -1*init_bound + 2*init_bound* np.random.random(100)
#     init_theta = np.ones(100)
#
#     parent_path =Path(__file__).parent
#     funnel100_dir = os.path.join(parent_path.parent , 'example', 'funnel100')
#     stan_file = os.path.join(funnel100_dir, 'funnel100.stan')
#
#     my_file = Path(stan_file)
#
#     try:
#       bs_model = bs.StanModel.from_stan_file(stan_file, model_data = None)
#
#     except Exception as e:
#       print(e)
#       print("Error opening the stan file or it doesnt exist!")
#       stan_code ="""
#                parameters {
#                   real log_sigma;
#                   vector[99] alpha;
#                  }
#                model {
#                  log_sigma ~ normal(0, 3);
#                  alpha ~ normal(0, exp(log_sigma / 2));
#                 }
#                 """
#       stan_file_path = os.path.join(funnel100_dir, 'temp_stan.stan')
#       temp_file = open(stan_file_path, "w")
#       temp_file.write( stan_code + '\n' )
#       temp_file.close()
#       bs_model = bs.StanModel.from_stan_file(stan_file_path, model_data = None)
#
#     num_param = bs_model.param_num()
#
#     theta_unc = init_theta
#     lp, grad = bs_model.log_density_gradient(theta_unc, jacobian=False)
#
#     grad1 = np.full(shape=num_param, fill_value=-0.3678794, dtype=np.float16)
#     grad1 = np.append(-31.4010788, grad1)
#
#     #np.testing.assert_array_almost_equal(grad, grad1)
#
#     pathfinder = OptimPath(
#                  init_point = init_theta,
#                  init_bound = init_bound,
#                  log_density_grad = bs_model.log_density_gradient,
#                  number_iter = 1000000,
#                  explore_hmc_from_initial = False,
#                  seed = 12345)
#
#     X, G, F, Ykt_history, Skt_history,list_true_cond = pathfinder.optim_path()
#     print("X=")
#     print(X)
#
#     print("G=")
#     print(G)
#
#     print("F=")
#     print(F)

