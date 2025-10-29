import pyro
import torch
from pyro.distributions import Distribution
from typing import Callable, Dict, Optional

import gempy as gp
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.modules.forwards import run_gempy_forward

GemPyPyroModel = Callable[[gp.data.GeoModel, torch.Tensor], None]


def make_gempy_pyro_model(
        *,
        priors: Dict[str, Distribution],
        set_interp_input_fn: Callable[
            [Dict[str, Distribution], gp.data.GeoModel],
            gp.data.InterpolationInput
        ],
        likelihood_fn: Optional[Callable[[gp.data.Solutions], Distribution]],
        obs_name: str = "obs"
) -> GemPyPyroModel:
    """
    Factory to produce a Pyro model for GemPy forward simulations.

    This returns a `model(geo_model, obs_data)` function that you can hand
    directly to Pyro's inference APIs (`Predictive`, `MCMC`, …).  Internally
    the generated model does:

      1. Samples each key/value in `priors` via `pyro.sample(name, dist)`.  
      2. Calls `set_interp_input_fn(samples, geo_model)` to inject those samples
         into a new `InterpolationInput`.  
      3. Runs the GemPy forward solver with `run_gempy_forward(...)`.  
      4. Wraps the resulting `Solutions` in `likelihood_fn(...)` to get a Pyro
         distribution, then observes `obs_data` under that distribution.

    Parameters
    ----------
    priors : Dict[str, Distribution]
        A mapping from Pyro sample‐site names to Pyro Distribution objects
        defining your prior beliefs over each parameter.
    set_interp_input_fn : Callable[[samples, geo_model], InterpolationInput]
        A user function which receives:
          
          * `samples` (Dict[str, Tensor]): the values drawn from your priors  
          * `geo_model` (gempy.core.data.GeoModel): the base geological model  
          
        and must return a ready‐to‐use `InterpolationInput` for GemPy.
    likelihood_fn : Callable[[Solutions], Distribution]
        A function mapping the GemPy forward‐model output (`Solutions`)
        to a Pyro Distribution representing the likelihood of the observed data.
    obs_name : str, optional
        The Pyro site name under which `obs_data` will be observed.
        Defaults to `"obs"`.

    Returns
    -------
    model : Callable[[GeoModel, Tensor], None]
        A Pyro‐compatible model taking:

          * `geo_model`: your GemPy GeoModel object  
          * `obs_data`: a torch.Tensor of observations  

        This function has no return value; it registers its random draws
        via `pyro.sample`.

    Example
    -------
    >>> import pyro.distributions as dist
    >>> from gempy_probability import make_gempy_pyro_model
    >>> from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
    >>> import gempy_probability
    >>>
    >>> # 1) Define a simple Normal prior on μ
    >>> priors = {"μ": dist.Normal(0., 1.)}
    >>>
    >>> # 2) Function to inject μ into your interp-input
    >>> def set_input(samples, gm):
    ...     inp = interpolation_input_from_structural_frame(gm)
    ...     inp.surface_points.sp_coords =  torch.index_put(
    ...          input=inp.surface_points.sp_coords,
    ...          indices=(torch.tensor([0]), torch.tensor([2])), # * This has to be Tensors
    ...          values=(samples["μ"])
    ...          )
    ...     return inp
    >>>
    >>>
    >>> # 4) Build the model
    >>> pyro_model = make_gempy_pyro_model(
    ...                 priors=priors,
    ...                 set_interp_input_fn=set_input, 
    ...                 likelihood_fn=gempy_probability.likelihoods.thickness_likelihood, 
    ...                 obs_name="y")
    >>>
    >>>
    >>> # Now this can be used with Predictive or MCMC directly:
    >>> #   Predictive(pyro_model, num_samples=100)(geo_model, obs_tensor)
    """

    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    def model(geo_model: gp.data.GeoModel, obs_data: torch.Tensor):
        # 1) Sample from the user‐supplied priors
        samples: Dict[str, torch.Tensor] = {}
        for name, dist in priors.items():
            samples[name] = pyro.sample(name, dist)

        # 2) Build the new interpolation input
        interp_input = set_interp_input_fn(samples, geo_model)

        # 3) Run GemPy forward simulation
        simulated: gp.data.Solutions = run_gempy_forward(
            interp_input=interp_input,
            geo_model=geo_model
        )

        # 4) Wrap in likelihood & observe
        if likelihood_fn is None:
            return
        
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
