import numpy as np
import os
import pyro
import pyro.distributions as dist
import torch
from arviz import InferenceData
from pyro.distributions import Distribution

import gempy as gp
from gempy_engine.core.backend_tensor import BackendTensor
import gempy_probability as gpp
from gempy_engine.core.data.interpolation_input import InterpolationInput
from pyro.infer import Predictive
import pyro
import arviz as az
import matplotlib.pyplot as plt

from pyro.infer import NUTS
from pyro.infer import MCMC
from pyro.infer.autoguide import init_to_mean


def run_predictive(prob_model: gpp.GemPyPyroModel, geo_model: gp.data.GeoModel, 
                   y_obs_list: torch.Tensor, n_samples: int, plot_trace:bool=False) -> az.InferenceData:
    predictive = Predictive(
        model=prob_model,
        num_samples=n_samples
    )
    prior: dict[str, torch.Tensor] = predictive(geo_model, y_obs_list)
    # print("Number of interpolations: ", geo_model.counter)

    data: az.InferenceData = az.from_pyro(prior=prior)
    if plot_trace:
        az.plot_trace(data.prior)
        plt.show()
        
    return data
