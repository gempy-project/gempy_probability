import pyro
import torch
from typing import Callable, Dict
from pyro.distributions import Distribution

from gempy_probability.modules.forwards import run_gempy_forward
import gempy as gp


def make_gempy_pyro_model(
        priors: Dict[str, Distribution],
        set_interp_input_fn: Callable[..., gp.data.Solutions],
        likelihood_fn: Callable[[gp.data.Solutions], Distribution],
        obs_name: str = "obs"
):

    def model(geo_model, obs_data):
        # 1) Sample each prior
        samples = {}
        for name, dist in priors.items():
            samples[name] = pyro.sample(name, dist)

        # 3) Run your forward geological model
        simulated: gp.data.Solutions = run_gempy_forward(
            interp_input=set_interp_input_fn(samples, geo_model), 
            geo_model=geo_model
        )

        # 4) Build likelihood and observe
        lik_dist = likelihood_fn(simulated)
        pyro.sample(obs_name, lik_dist, obs=obs_data)

    return model


def make_generic_pyro_model(
        priors: Dict[str, Distribution],
        forward_fn: Callable[..., torch.Tensor],
        likelihood_fn: Callable[[torch.Tensor], Distribution],
        obs_name: str = "obs"
):
    """
    Build a Pyro model function with:
      - priors:  map parameter names -> pyro.distributions objects
      - forward_fn: fn(samples_dict, *args)-> simulated quantity (torch.Tensor)
      - likelihood_fn: fn(simulated)-> a Pyro Distribution over data
      - obs_name: name of the observed site
    """
    def model(geo_model, obs_data):
        # 1) Sample each prior
        samples = {}
        for name, dist in priors.items():
            samples[name] = pyro.sample(name, dist)

        # 2) Run your forward geological model
        simulated = forward_fn(samples, geo_model)

        # 3) Build likelihood and observe
        lik_dist = likelihood_fn(simulated)
        pyro.sample(obs_name, lik_dist, obs=obs_data)

    return model
