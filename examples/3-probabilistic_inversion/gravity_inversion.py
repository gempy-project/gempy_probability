'''
# GemPy 3: gravity inversion for normal fault model

Based on `GemPy3_Tutorial_XX_fault_gravity.ipynb`

For installation, see the first notebook - here only repeated if running on Google Colab.
'''
# %%
# !pip install gempy --pre
# %%
# For 3D plots
# !sudo apt install libgl1-mesa-glx xvfb
# !pip install pyvista

import pyvista as pv
# pv.start_xvfb()
# %% md
# GemPy is now separated into different modules to keep the dependencies low. This means that the base version `gempy-engine` is very lean. To enable possibilities to view the created models, we also need to install the `gempy-viewer` module:
# %%
# !pip install gempy_viewer
# %% md
# With these two modules installed (only required the first time on each system), we can import the modules into the notebook, as usual:
# %%
# Importing GemPy and viewer
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
# %% md
# And for some additional steps in this notebook:
# %%
# Auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
# %% md
# Packages for inversion
# %%
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az

# %% md
## Step 1: Model setup

# In a first step, we define the model domain. In the standard setting, this as simple as defining model extent
# and grid resolution (i.e.: grid elements in each axis direction). We also need to define a structural frame
# (more on that later) - for now, simply filled with a default structure:

# %%
resolution = [150, 10, 150]
extent = [0, 200, -100, 100, -100, 0]

# %%
# Configure GemPy for geological modeling with PyTorch backend
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Fault model',
    extent=extent,
    resolution=resolution,
    structural_frame=gp.data.StructuralFrame.initialize_default_structure()
)

# %%
interpolation_options = geo_model.interpolation_options
interpolation_options.mesh_extraction = True
interpolation_options.kernel_options.range = .7
interpolation_options.kernel_options.c_o = 3
interpolation_options.kernel_options.compute_condition_number = True
# %% md
## Step 2: Add geological data


# %% md
### Add surface points
# %%
gp.add_surface_points(
    geo_model=geo_model,
    x=[40, 60, 120, 140],
    y=[0, 0, 0, 0],
    z=[-50, -50, -60, -60],
    elements_names=['surface1', 'surface1', 'surface1', 'surface1']
)

gp.add_orientations(
    geo_model=geo_model,
    x=[130],
    y=[0],
    z=[-50],
    elements_names=['surface1'],
    pole_vector=[[0, 0, 1.]]
)

# Define second element
element2 = gp.data.StructuralElement(
    name='surface2',
    color=next(geo_model.structural_frame.color_generator),
    surface_points=gp.data.SurfacePointsTable.from_arrays(
        x=np.array([120]),
        y=np.array([0]),
        z=np.array([-40]),
        names='surface2'
    ),
    orientations=gp.data.OrientationsTable.initialize_empty()
)

# Add second element to structural frame
geo_model.structural_frame.structural_groups[0].append_element(element2)

# add fault
# Calculate orientation from point values
fault_point_1 = (80, -20)
fault_point_2 = (110, -80)

# calculate angle
angle = np.arctan((fault_point_2[0] - fault_point_1[0]) / (fault_point_2[1] - fault_point_1[1]))

x = np.cos(angle)
z = - np.sin(angle)

element_fault = gp.data.StructuralElement(
    name='fault1',
    color=next(geo_model.structural_frame.color_generator),
    surface_points=gp.data.SurfacePointsTable.from_arrays(
        x=np.array([fault_point_1[0], fault_point_2[0]]),
        y=np.array([0, 0]),
        z=np.array([fault_point_1[1], fault_point_2[1]]),
        names='fault1'
    ),
    orientations=gp.data.OrientationsTable.from_arrays(
        x=np.array([fault_point_1[0]]),
        y=np.array([0]),
        z=np.array([fault_point_1[1]]),
        G_x=np.array([x]),
        G_y=np.array([0]),
        G_z=np.array([z]),
        names='fault1'
    )
)

group_fault = gp.data.StructuralGroup(
    name='Fault1',
    elements=[element_fault],
    structural_relation=gp.data.StackRelationType.FAULT,
    fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
)

# Insert the fault group into the structural frame:
geo_model.structural_frame.insert_group(0, group_fault)
# %% md
## Compute model
# %%
geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)
gp.compute_model(geo_model)
# %%

# %%
# Visualize the computed geological model in 3D
gempy_vista = gpv.plot_3d(
    model=geo_model,
    show=True,
    kwargs_plot_structured_grid={'opacity': 0.8},
    image=True
)

# %%
# Preview the model's input data:
p2d = gpv.plot_2d(geo_model, show=False)
plt.grid()
plt.show()

# %% md
## Calculate gravity
# %%
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")
# %% md
# Set device positions

# %%
interesting_columns = pd.DataFrame()
x_vals = np.arange(20, 191, 10)
interesting_columns['X'] = x_vals
interesting_columns['Y'] = np.zeros_like(x_vals)

# Configuring the data correctly is key for accurate gravity calculations.
device_location = interesting_columns[['X', 'Y']]
device_location['Z'] = 0  # Add a Z-coordinate

# Set up a centered grid for geophysical calculations
# This grid will be used for gravity gradient calculations.
gp.set_centered_grid(
    grid=geo_model.grid,
    centers=device_location,
    resolution=np.array([75, 5, 150]),
    radius=np.array([150, 10, 300])
)

# Calculate the gravity gradient using GemPy
# Gravity gradient data is critical for geophysical modeling and inversion.
gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)

densities_tensor = BackendTensor.t.array([2., 2., 3., 2.])
densities_tensor.requires_grad = True

# Set geophysics input for the GemPy model
# Configuring this input is crucial for the forward gravity calculation.
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=BackendTensor.t.array(gravity_gradient),
    densities=densities_tensor
)

# %%
# Compute the geological model with geophysical data
# This computation integrates the geological model with gravity data.
sol = gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.PYTORCH,
        dtype='float64'
    )
)
grav = - sol.gravity
grav[0].backward()
# %%
plt.plot(x_vals, grav.detach().numpy(), '.-')
plt.xlim([0, 200])
plt.show()

# %% md
## Plot model and gravity solution
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

blocks = geo_model.solutions.raw_arrays.lith_block.reshape(resolution)


def plot_model_and_grav(blocks, grav, **kwds):
    # Assuming the setup variables x_vals, grav, and blocks are defined.
    x_min, x_max = 0, 200
    y_min, y_max = -100, 0

    vmin = kwds.get("v_min", 1)
    vmax = kwds.get("v_max", 5)

    grav_min = kwds.get("grav_min", grav.min())
    grav_max = kwds.get("grav_max", grav.max())

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    try:
        ax1.plot(x_vals, grav.detach().numpy(), '.-')
    except AttributeError:
        ax1.plot(x_vals, grav, '.-')

    if grav_min:
        ax1.set_ylim([grav_min, grav_max])

    ax1.set_ylabel("Gravity anomaly $\Delta g_z$ [m/s$^2$]")

    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((0, grav_min), width=50, height=grav_max-grav_min, facecolor='gray', alpha=0.2)
    # ax1.add_patch(rect)
    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((150, grav_min), width=50, height=grav_max-grav_min, facecolor='gray', alpha=0.2)
    # ax1.add_patch(rect)

    ax2 = fig.add_subplot(212, sharex=ax1)

    # Correctly define your grid edges
    y_edges = np.linspace(y_min, y_max, blocks.shape[2] + 1)
    x_edges = np.linspace(x_min, x_max, blocks.shape[0] + 1)  # This should be corrected if it was incorrect

    # Making sure x_edges and y_edges are correctly sized
    assert len(x_edges) == blocks.shape[0] + 1, "x_edges does not match expected length"
    assert len(y_edges) == blocks.shape[2] + 1, "y_edges does not match expected length"

    # Ensure blocks[:, 5, :].T is correctly shaped relative to x_edges and y_edges
    c = ax2.pcolor(x_edges, y_edges, blocks[:, 5, :].T, cmap='viridis', vmin=vmin, vmax=vmax)
    # c = ax2.pcolor(blocks[:, 5, :].T, cmap='RdYlBu_r') # , vmin=1, vmax=2)

    # plot input data if given in kwds:
    if "input_data" in kwds.keys():
        plt.scatter(kwds["input_data"]['X'], kwds["input_data"]['Z'], color='#CCCCCC', edgecolors='black', s=100)

    # remove ticks from upper plot
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")

    # Add a transparent gray rectangle to the left side of the plot
    # rect = patches.Rectangle((0, -100), width=50, height=100, facecolor='white', alpha=0.2)
    # ax2.add_patch(rect)
    # rect = patches.Rectangle((150, -100), width=50, height=100, facecolor='white', alpha=0.2)
    # ax2.add_patch(rect)

    # Adjust the subplot layout to prevent the subplots from overlapping
    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(left=0.1, right=.95, bottom=0.1, top=0.95)

    plt.close()

    return fig


input_data = geo_model.surface_points_copy.df
fig = plot_model_and_grav(blocks, grav.detach().numpy(), input_data=input_data)
fig.show()
# %% md
## Set up Pyro model

# %%
# Define hyperparameters for the Bayesian geological model
# Use first: lateral position of fault only
fault_1_x = torch.tensor(80.)
fault_2_x = torch.tensor(110.)


# %% md
### Probabilistic model

# @Miguel: how to adjust for input points as stochastic variables?


def gaussian_kernel(locations, length_scale, variance):
    import torch
    # Compute the squared Euclidean distance between each pair of points
    locations = torch.tensor(locations.values, dtype=torch.float32)
    distance_squared = torch.cdist(locations, locations, p=2.).pow(2.)
    # Compute the covariance matrix using the Gaussian kernel
    covariance_matrix = variance * torch.exp(-0.5 * distance_squared / length_scale ** 2)
    return covariance_matrix


grav = sol.gravity

# Configure the Pyro model for geological data
# * These are density values for the geological model
prior_tensor = densities_tensor  # * This is the prior tensor

covariance_matrix = gaussian_kernel(  # * This is the likelihood function
    locations=device_location,
    length_scale=torch.tensor(1_000.0, dtype=torch.float32),
    variance=torch.tensor(25.0 ** 2, dtype=torch.float32)
)

# * This is the observed gravity data
adapted_observed_grav = grav

# Placing the tensor pointer in the rest of the model
geo_model.geophysics_input = gp.data.GeophysicsInput(
    tz=geo_model.geophysics_input.tz,
    densities=prior_tensor,
)


# %%
# Define the Pyro probabilistic model for inversion


def model(y_obs_list, interpolation_input):
    """
    Pyro model representing the probabilistic aspects of the geological model.
    """
    import gempy_engine

    # * Prior definition
    prior_mean = 2.62
    mu_density = pyro.sample(
        name=r'$\mu_{density}$',
        fn=dist.Normal(
            loc=prior_mean,
            scale=torch.tensor(0.5, dtype=torch.float32))
    )

    # Changing the density of the first formation
    geo_model.geophysics_input.densities = torch.index_put(
        input=prior_tensor,
        indices=(torch.tensor([0]),),
        values=mu_density
    )

    # * Deterministic computation of the geological model
    # GemPy does not have API for this yet so we need to compute
    # the model directly by calling the engine
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input
    )

    simulated_geophysics = geo_model.solutions.gravity
    pyro.deterministic(r'$\mu_{gravity}$', simulated_geophysics)

    # * Likelihood definition
    pyro.sample(
        name="obs",
        fn=dist.MultivariateNormal(simulated_geophysics, covariance_matrix),
        obs=y_obs_list
    )


# %%
# Prepare observed data for Pyro model and optimize mesh settings

# TODO: This is going to be a problem, that 17 should be number of observations
n_devices = device_location.values.shape[0]
y_obs_list = torch.tensor(adapted_observed_grav).view(1, n_devices)

# Optimize for speed
interpolation_options.mesh_extraction = False
interpolation_options.number_octree_levels = 1
geo_model.grid.set_inactive("topography")
geo_model.grid.set_inactive("octree")

# %%
# Perform prior sampling and visualize the results
if PRIOR_PREDICTIVE := True:
    predictive_model = Predictive(model=model, num_samples=50)
    prior = predictive_model(
        y_obs_list=y_obs_list,
        interpolation_input=geo_model.interpolation_input_copy
    )

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
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictive = posterior_predictive(y_obs_list, interpolation_input=geo_model.interpolation_input_copy)

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
