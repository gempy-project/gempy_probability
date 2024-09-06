"""
# GemPy 3: gravity inversion for normal fault model

Based on `GemPy3_Tutorial_XX_fault_gravity.ipynb`

For installation, see the first notebook - here only repeated if running on Google Colab.
"""

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
import pandas as pd

from .._aux_functions.aux_funcs_probabilistic_inversion import plot_model_and_grav

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

input_data = geo_model.surface_points_copy.df
fig = plot_model_and_grav(
    blocks=(geo_model.solutions.raw_arrays.lith_block.reshape(resolution)),
    grav=grav.detach().numpy(),
    x_vals=x_vals,
    input_data=input_data
)
fig.show()
