"""
2.2 - Including GemPy
=====================

Complex probabilistic model
---------------------------

"""
from pyro.infer.autoguide import init_to_mean

from pyro.infer.inspect import get_dependencies

import gempy as gp
import gempy_viewer as gpv
import os
import numpy as np
import matplotlib.pyplot as plt

import arviz as az
from gempy_engine.core.backend_tensor import BackendTensor
import gempy_engine

# %%
data_path = os.path.abspath('../')


# %%
def plot_geo_setting_well(geo_model):
    device_loc = np.array([[6e3, 0, 3700]])
    p2d = gpv.plot_2d(geo_model, show_topography=False, legend=False, show=False)

    well_1 = 3.41e3
    well_2 = 3.6e3
    p2d.axes[0].scatter([3e3], [well_1], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter([9e3], [well_2], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter(device_loc[:, 0], device_loc[:, 2], marker='x', s=400,
                        c='#DA8886', zorder=10)

    p2d.axes[0].vlines(3e3, .5e3, well_1, linewidth=4, color='gray')
    p2d.axes[0].vlines(9e3, .5e3, well_2, linewidth=4, color='gray')
    p2d.axes[0].vlines(3e3, .5e3, well_1)
    p2d.axes[0].vlines(9e3, .5e3, well_2)

    p2d.fig.show()


# %%
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Wells',
    extent=[0, 12000, -500, 500, 0, 4000],
    refinement=3,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
        path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
    )
)

geo_model.interpolation_options.uni_degree = 0
geo_model.interpolation_options.mesh_extraction = False
geo_model.interpolation_options.sigmoid_slope = 1100.

x_loc = 6000
y_loc = 0
z_loc = np.linspace(0, 4000, 100)
xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])

gp.set_custom_grid(
    geo_model.grid,
    xyz_coord=xyz_coord
)

# %%
# Input setting
plot_geo_setting_well(geo_model=geo_model)

# %%
# Interpolate initial guess
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy
    )
)

plot_geo_setting_well(geo_model=geo_model)

# %%
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, Predictive, HMC

from gempy_probability.plot_posterior import PlotPosterior, default_red, default_blue

sp_coords_copy = geo_model.interpolation_input.surface_points.sp_coords.copy()
coords = torch.as_tensor(sp_coords_copy[1:, :])
val = torch.as_tensor(sp_coords_copy[0, :2])

detached_coords = torch.as_tensor(sp_coords_copy)

BackendTensor.change_backend_gempy(
    engine_backend=gp.data.AvailableBackends.PYTORCH,
)


def model(y_obs_list):
    # Pyro models use the 'sample' function to define random variables
    prioir_mean = sp_coords_copy[0, 2]
    mu_top = pyro.sample(r'$\mu_{top}$', dist.Normal(prioir_mean, torch.tensor(0.02, dtype=torch.float64)))

    interpolation_input = geo_model.interpolation_input  # ! If we pull this out the model it breaks. It seems all the graph needs to be in the function!

    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([0]), torch.tensor([2])),
        mu_top
    )

    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )

    simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    thickness = simulated_well.sum()  # * We would need to segment this better

    pyro.deterministic(r'$\mu_{thickness}$', thickness.detach())  # Deterministic transformation
    y_thickness = pyro.sample(r'$y_{thickness}$', dist.Normal(thickness, 50), obs=y_obs_list)

    if False:
        gpv.plot_2d(geo_model)


y_obs_list = torch.tensor([200, 210, 190])

a = get_dependencies(model, (y_obs_list[:1]))
import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(a)

# 1. Prior Sampling
if True:
    prior = Predictive(model, num_samples=50)(y_obs_list)
    data = az.from_pyro(
        prior=prior,
    )

    az.plot_trace(data.prior)
    plt.show()

# Now you can run MCMC using NUTS to sample from the posterior
if True:
    pyro.primitives.enable_validation(is_validate=True)
    # # Set up the NUTS sampler
    nuts_kernel = NUTS(
        model,
        step_size=0.0085,  # Example of custom step size
        adapt_step_size=True,  # Let NUTS adapt the step size
        target_accept_prob=0.9,  # Example of target acceptance rate
        max_tree_depth=10,
        init_strategy=init_to_mean,
    )  # Example of maximum tree depth

    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=50, disable_validation=False)

    mcmc.run(y_obs_list)

    # 3. Sample from Posterior Predictive

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(y_obs_list)

    data = az.from_pyro(
        posterior=mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )

    az.plot_trace(data)
    plt.show()
    plt.show()

    # %%
    az.plot_density(
        data=[data.posterior_predictive, data.prior_predictive],
        shade=.9,
        var_names=[
            r'$\mu_{thickness}$'
        ],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )

    plt.show()
