"""
Probabilistic Inversion Example: Geological Model: Making the model stable
--------------------------------------------------------------------------


"""


import os
import time

import arviz as az
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import xarray as xr
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from pyro.infer import MCMC, NUTS, Predictive

import gempy as gp
import gempy_engine
import gempy_viewer as gpv

from examples.examples._aux_func_gravity_inversion import calculate_scale_shift, gaussian_kernel, initialize_geo_model, setup_geophysics
from examples.examples._aux_func_gravity_inversion_II import process_file

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.modules.plot.plot_posterior import default_red, default_blue

# %%
# Config
seed = 123456
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

# %%
# Start the timer for benchmarking purposes
start_time = time.time()

# %%
# Load necessary configuration and paths from environment variables
config = dotenv_values()
path = config.get("PATH_TO_MODEL_1_Subsurface")

# %%
# Initialize lists to store structural elements for the geological model
structural_elements = []
global_extent = None
color_gen = gp.data.ColorsGenerator()

# %%
# Process each .nc file in the specified directory for model construction
for filename in os.listdir(path):
    base, ext = os.path.splitext(filename)
    if ext == '.nc':
        structural_element, global_extent = process_file(os.path.join(path, filename), global_extent, color_gen)
        structural_elements.append(structural_element)

# %%
# Configure GemPy for geological modeling with PyTorch backend
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH, dtype="float64")

geo_model = initialize_geo_model(
    structural_elements=structural_elements,
    extent=(np.array(global_extent)),
    topography=(xr.open_dataset(os.path.join(path, "Topography.nc"))),
    load_nuggets=False
)

