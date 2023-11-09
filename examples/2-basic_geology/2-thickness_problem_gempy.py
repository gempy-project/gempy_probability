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
    refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
        path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
    )
)

geo_model.interpolation_options.uni_degree = 0
geo_model.interpolation_options.mesh_extraction = False
geo_model.interpolation_options.sigmoid_slope = 1000.

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
gp.compute_model(geo_model)

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



# val = geo_model.interpolation_input.surface_points.sp_coords[0, 2]

BackendTensor.change_backend_gempy(
    engine_backend=gp.data.AvailableBackends.PYTORCH,
)


def model(y_obs_list):
    # Pyro models use the 'sample' function to define random variables
    prioir_mean = sp_coords_copy[0, 2]
    mu_top = pyro.sample(r'$\mu_{top}$', dist.Normal(prioir_mean, 0.02))

    # sample = 0.0487 + (mu_top + geo_model.transform.position[2]) / geo_model.transform.isometric_scale
    if False:
        thickness=sample
    else:
        interpolation_input = geo_model.interpolation_input
        # foo = torch.cat(
        #     (interpolation_input.surface_points.sp_coords,
        #      torch.tensor([[0, 0, sample]]))
        # )

        
        params = torch.concat((val.view(2,1), mu_top.view(1,1)))
        combined = torch.concat((params.view(1,3), coords,), dim=0).view(4,3)
        interpolation_input.surface_points.sp_coords = combined
        # thickness = interpolation_input.surface_points.sp_coords.sum()

        geo_model.solutions = gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=geo_model.interpolation_options,
            data_descriptor=geo_model.input_data_descriptor,
            geophysics_input=geo_model.geophysics_input,
        )

        if False:
            gpv.plot_2d(geo_model)

        simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
        thickness = simulated_well.sum()

        # interpolation_input.surface_points.sp_coords[0, 2] =  0.0487 + (mu_top + geo_model.transform.position[2]) / geo_model.transform.isometric_scale

    # mu_thickness = pyro.deterministic(r'$\mu_{thickness}$', thickness)  # Deterministic transformation
    mu_thickness = thickness
    y_thickness = pyro.sample(r'$y_{thickness}$', dist.Normal(mu_thickness, 50), obs=y_obs_list)

    # sigma_top = pyro.sample(r"$\sigma_{top}$", dist.Gamma(0.3, 3.0))
    # y_top = pyro.sample(r"y_{top}", dist.Normal(mu_top, sigma_top), obs=torch.tensor([3.02]))

    # mu_bottom = pyro.sample(r'$\mu_{bottom}$', dist.Normal(1000, 500))
    # sigma_bottom = pyro.sample(r'$\sigma_{bottom}$', dist.Gamma(0.3, 3.0))
    # y_bottom = pyro.sample(r'y_{bottom}', dist.Normal(mu_bottom, sigma_bottom), obs=torch.tensor([1.02]))

    # TODO: To decide what to do with this.
    # interpolation_input = geo_model.interpolation_input
    # geo_model.taped_interpolation_input = interpolation_input
    # 
    # sample =  0.0487 + (mu_top + geo_model.transform.position[2]) / geo_model.transform.isometric_scale
    # foo = torch.cat(
    #    (  interpolation_input.surface_points.sp_coords,
    #     torch.tensor([[0, 0, sample]] ))
    # )
    # 
    # interpolation_input.surface_points.sp_coords = foo
    # # interpolation_input.surface_points.sp_coords[0, 2] =  0.0487 + (mu_top + geo_model.transform.position[2]) / geo_model.transform.isometric_scale
    # 
    # # original_surface_points = interpolation_input.surface_points.sp_coords.detach().clone()
    # # coords[0:1, 2] = val + (mu_top + geo_model.transform.position[2]) / geo_model.transform.isometric_scale
    # # coords[2:3, 2] = (mu_bottom + geo_model.transform.position[2]) / geo_model.transform.isometric_scale
    # 
    # if False:
    #     mu_top.register_hook(lambda x: print("I am here!", x))
    # interpolation_input.surface_points.sp_coords = coords
    # 
    # geo_model.solutions = gempy_engine.compute_model(
    #     interpolation_input=interpolation_input,
    #     options=geo_model.interpolation_options,
    #     data_descriptor=geo_model.input_data_descriptor,
    #     geophysics_input=geo_model.geophysics_input,
    # )
    # 
    # 
    # simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    # thickness = simulated_well.sum()
    # 
    # # # Count how many values are between 1.5 an 2.5 in simulated_well
    # # n_values = torch.sum((simulated_well > 1.5) & (simulated_well < 2.5), axis=0, keepdim=True)
    # # # thickness = torch.tensor(n_values * 40)
    # # 
    # # random_noise = pyro.sample(r'noise', dist.Normal(10, 3))
    # # thickness = n_values * 40 + random_noise
    # 
    # mu_thickness = pyro.deterministic(r'$\mu_{thickness}$', thickness)  # Deterministic transformation
    # # sigma_thickness = pyro.sample(r'$\sigma_{thickness}$', dist.Gamma(2000, 3.0))
    pass


y_obs_list = torch.tensor([
    # 2.12, 2.06, 2.08, 2.05, 2.08, 2.09, 2.19, 2.07, 2.16, 2.11, 2.13,
    200, 210, 190])

a = get_dependencies(model, (y_obs_list[:1]))
import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(a)

# 1. Prior Sampling
if True:
    prior = Predictive(model, num_samples=10)(y_obs_list)
    data = az.from_pyro(
        prior=prior,
    )

    az.plot_trace(data.prior)
    plt.show()

# Now you can run MCMC using NUTS to sample from the posterior
if True:
    pyro.primitives.enable_validation(is_validate=True)
    # # Set up the NUTS sampler
    nuts_kernel = NUTS(model,
                       step_size=0.0085,  # Example of custom step size
                       adapt_step_size=True,  # Let NUTS adapt the step size
                       target_accept_prob=0.9,  # Example of target acceptance rate
                       max_tree_depth=10,
                       init_strategy=init_to_mean,
                       )  # Example of maximum tree depth
    
    mcmc = MCMC(nuts_kernel, num_samples=30, warmup_steps=20, disable_validation=False)

    mcmc.run(y_obs_list)

    # 3. Sample from Posterior Predictive
    posterior_samples = mcmc.get_samples()
    data = az.from_pyro(
        posterior=mcmc,
        # prior=prior,
    )

    az.plot_trace(data)
    plt.show()
