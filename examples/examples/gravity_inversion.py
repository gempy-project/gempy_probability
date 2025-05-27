"""
Probabilistic Inversion Example: Geological Model
--------------------------------------------------

This example demonstrates a probabilistic inversion of a geological model using Bayesian methods.
"""
import numpy as np
import os

import arviz as az
import dotenv
import gempy_engine
import gempy_viewer as gpv
import pyro
import pyro.distributions as dist
import torch
from gempy_engine.core.backend_tensor import BackendTensor
from matplotlib import pyplot as plt
from pyro.infer import MCMC, NUTS, Predictive

import gempy as gp
from examples.examples._aux_func_gravity_inversion import setup_geophysics
from gempy_probability.modules.likelihoods.gravity_likelihoods import calculate_scale_shift, gaussian_kernel
from gempy_probability.modules.plot.plot_posterior import default_red, default_blue

dotenv.load_dotenv()

# %%
# Config
seed = 123456
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

model_path = os.getenv("PATH_TO_NUGGET_TEST_MODEL")
model_file = os.path.join(model_path, "nugget_effect_optimization.gempy")
geo_model = gp.API.load_model(model_file)

# %%
# Forward Gravity
# ---------------
# Setup geophysics configuration for the model
geophysics_input = setup_geophysics(
    env_path="PATH_TO_MODEL_1_BOUGUER",
    geo_model=geo_model,
    densities=[2.61, 2.92, 3.1, 2.92, 2.61, 2.61, 2.61]
)

# %%
# Adjust interpolation options for geological modeling
interpolation_options = geo_model.interpolation_options
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True

# %%
# Compute the geological model
sol: gp.data.Solutions = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    ),
    validate_serialization=False # TODO: [ ] Validate this serialization
)

# %%
# Visualize the computed geological model in 3D
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)

# %%
# Calculate and adapt the observed gravity data for model comparison
scale_factor, shift_term = calculate_scale_shift(
    measured=geophysics_input["Bouguer_267_complete"].values,
    simulated=sol.gravity.detach().numpy()
)
adapted_observed_grav: np.dtype[float] = scale_factor * geophysics_input["Bouguer_267_complete"] + shift_term

# %%
# Plot the 2D gravity data for visualization
plot2d = gpv.plot_2d(geo_model, show_topography=True, section_names=["topography"], show=False)
sc = plot2d.axes[0].scatter(
    geophysics_input['X'],
    geophysics_input['Y'],
    c=adapted_observed_grav,
    cmap='viridis',
    s=100,
    zorder=10000
)
plt.colorbar(sc, label="mGal")
plt.show()


# %%
# Define Probabilistic model
# --------------------------
# Define hyperparameters for the Bayesian geological model
length_scale_prior = torch.tensor(1_000.0)
variance_prior = torch.tensor(25.0 ** 2)
covariance_matrix = gaussian_kernel(geophysics_input[['X', 'Y']], length_scale_prior, variance_prior)

# %%
# Configure the Pyro model for geological data
prior_tensor = BackendTensor.t.array([2.61, 2.92, 3.1, 2.92, 2.61, 2.61]).to(torch.float64)
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=geo_model.geophysics_input.tz,
    densities=prior_tensor,
)

bar

# %%
# Define the Pyro probabilistic model for inversion
def model(y_obs_list, interpolation_input):
    """
    Pyro model representing the probabilistic aspects of the geological model.
    """
    prior_mean = 2.62
    mu_density = pyro.sample(
        name=r'$\mu_{density}$',
        fn=dist.Normal(prior_mean, torch.tensor(0.5, dtype=torch.float64))
    )
    geo_model.geophysics_input.densities = torch.index_put(
        input=prior_tensor,
        indices=(torch.tensor([3]),),
        values=mu_density
    )
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input
    )
    simulated_geophysics = geo_model.solutions.gravity
    pyro.deterministic(r'$\mu_{gravity}$', simulated_geophysics)
    pyro.sample(
        name="obs",
        fn=dist.MultivariateNormal(simulated_geophysics, covariance_matrix),
        obs=y_obs_list
    )


# %%
# Prepare observed data for Pyro model and optimize mesh settings
y_obs_list = torch.tensor(adapted_observed_grav.values).view(1, 17)
interpolation_options.mesh_extraction = False
interpolation_options.number_octree_levels = 1
geo_model.grid.active_grids ^= gp.data.Grid.GridTypes.TOPOGRAPHY
geo_model.grid.active_grids ^= gp.data.Grid.GridTypes.DENSE

# %%
# Perform prior sampling and visualize the results
# raise NotImplementedError("From this point we need to optimize the code again.")
if True:
    prior = Predictive(model, num_samples=50)(y_obs_list, interpolation_input=geo_model.interpolation_input_copy)
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    plt.show()

# %%
# Run Markov Chain Monte Carlo (MCMC) using the NUTS algorithm for probabilistic inversion
pyro.primitives.enable_validation(is_validate=True)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=300)
mcmc.run(y_obs_list, interpolation_input=geo_model.interpolation_input_copy)

# %%
# Analyze posterior samples and predictives, and visualize the results
posterior_samples = mcmc.get_samples(50)
posterior_predictive = Predictive(model, posterior_samples)(y_obs_list, interpolation_input=geo_model.interpolation_input_copy)
data = az.from_pyro(
    posterior=mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)
az.plot_trace(data)
plt.show()

# %%
# Create density plots for posterior and prior distributions
# These plots provide insights into the parameter distributions and their changes.
az.plot_density(
    data=[data, data.prior],
    shade=.9,
    hdi_prob=.99,
    data_labels=["Posterior", "Prior"],
    colors=[default_red, default_blue],
)
plt.show()

# %%
az.plot_density(
    data=[data.posterior_predictive, data.prior_predictive],
    shade=.9,
    var_names=[r'$\mu_{gravity}$'],
    data_labels=["Posterior Predictive", "Prior Predictive"],
    colors=[default_red, default_blue],
)
plt.show()

# sphinx_gallery_thumbnail_number = -1
