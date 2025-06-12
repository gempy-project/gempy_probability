"""
Tests for gradient computation in GemPy models.

This module contains tests for computing gradients using both numerical
differentiation and automatic differentiation (PyTorch).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
from gempy_engine.core.backend_tensor import BackendTensor

# Constants
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_file_dir, '..', '..', 'examples', 'tutorials')
DATA_PATH = os.path.abspath(data_path)
SIGMOID_SLOPE = 1000
MODEL_EXTENT = [0, 12000, -500, 500, 0, 4000]
MODEL_RESOLUTION = np.array([10, 1, 10])


def create_test_model(sigmoid_slope=100, plot=False):
    """
    Create a GemPy geological model for testing gradients.
    
    Args:
        sigmoid_slope (float): Sigmoid slope parameter for interpolation
        plot (bool): Whether to display the model plot
        
    Returns:
        gp.data.GeoModel: Configured geological model
    """
    # Create the base model
    gempy_model = gp.create_geomodel(
        project_name='Wells',
        extent=MODEL_EXTENT,
        resolution=MODEL_RESOLUTION,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=f"{DATA_PATH}/data/2-layers/2-layers_orientations.csv",
            path_to_surface_points=f"{DATA_PATH}/data/2-layers/2-layers_surface_points.csv"
        )
    )
    
    # Configure interpolation options
    gempy_model.interpolation_options.uni_degree = 0
    gempy_model.interpolation_options.mesh_extraction = False
    gempy_model.interpolation_options.sigmoid_slope = sigmoid_slope
    
    # Setup PyTorch backend for gradient computation
    engine_config = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH)
    BackendTensor.change_backend_gempy(
        engine_backend=engine_config.backend,
        use_gpu=engine_config.use_gpu,
        dtype=engine_config.dtype
    )
    
    # Create interpolation input and enable gradients
    interpolation_input = interpolation_input_from_structural_frame(gempy_model)
    gempy_model.taped_interpolation_input = interpolation_input
    
    sp_coords_tensor = gempy_model.taped_interpolation_input.surface_points.sp_coords
    sp_coords_tensor.requires_grad = True
    
    # Compute the model
    gempy_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=gempy_model.interpolation_options,
        data_descriptor=gempy_model.input_data_descriptor,
        geophysics_input=gempy_model.geophysics_input,
    )
    
    if plot:
        gpv.plot_2d(gempy_model, show_scalar=False, kwargs_lithology={"plot_grid": True})
        plt.show()
        
    return gempy_model


def plot_gradient_grid(geo_model, gradient_values, title="Gradient Visualization"):
    """
    Plot gradient values on a 2D grid.
    
    Args:
        geo_model: GemPy geological model
        gradient_values: Array of gradient values to visualize
        title: Plot title
    """
    max_abs_val = np.max(np.abs(gradient_values))
    
    gpv.plot_2d(
        geo_model,
        show_topography=False,
        legend=False,
        show=True,
        override_regular_grid=gradient_values[:100],
        kwargs_lithology={
            'cmap': 'seismic',
            'norm': None,
            'plot_grid': True,
            'vmin': -max_abs_val,
            'vmax': max_abs_val
        }
    )
    plt.title(title)


def test_gradients_numerical():
    """
    Test gradient computation using numerical differentiation.
    
    This test computes gradients by finite differences, modifying surface
    points and observing changes in the geological model output.
    """
    geo_model = create_test_model(SIGMOID_SLOPE, plot=True)
    
    # Get initial parameter value
    par_val = geo_model.surface_points_copy.data['Z'][0]
    var = 50
    point_n = 0
    
    # Compute model for different parameter values
    values_to_compute = np.linspace(par_val - var, par_val + var, 30)
    arrays = []
    
    for val in values_to_compute:
        gp.modify_surface_points(geo_model, slice=point_n, Z=val)
        sol = gp.compute_model(geo_model)
        block_values = sol.octrees_output[0].last_output_center.final_block[:100]
        arrays.append(block_values)
    
    arrays = np.array(arrays)
    
    # Plot the parameter sensitivity
    _plot_parameter_sensitivity(values_to_compute, arrays, par_val)
    
    # Compute numerical gradients
    grads = np.diff(arrays.reshape(-1, 10, 10), axis=0)
    
    # Reset model to original state
    gp.modify_surface_points(geo_model, slice=point_n, Z=par_val)
    gp.compute_model(geo_model)
    
    # Extract gradient at middle point
    gradient_z_sp_1 = grads[15] / (2 * var / 30)
    
    plot_gradient_grid(geo_model, gradient_z_sp_1, "Numerical Gradients")


def _plot_parameter_sensitivity(values, arrays, par_val):
    """Plot parameter sensitivity grid."""
    iter_a = arrays.reshape(-1, 10, 10)
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    
    for i in range(10):
        for j in range(10):
            ax = plt.subplot(10, 10, (9 - j) * 10 + i + 1)
            ax.plot(values, iter_a[:, i, j], '.')
            ax.axvline(par_val, ymax=3, color='r')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_ylim(0, 3)
    
    plt.suptitle("Parameter Sensitivity Analysis")
    plt.show()


def test_gradients_automatic():
    """
    Test gradient computation using automatic differentiation.
    
    This test uses PyTorch's automatic differentiation to compute
    gradients of the geological model output with respect to surface points.
    """
    geo_model = create_test_model(sigmoid_slope=SIGMOID_SLOPE, plot=True)
    
    # Get the model output block
    block = geo_model.solutions.octrees_output[0].last_output_center.final_block
    sp_coords_tensor = geo_model.taped_interpolation_input.surface_points.sp_coords
    
    # Initialize Jacobian matrix
    jacobian = torch.zeros((
        sp_coords_tensor.shape[0],  # Number of surface points
        sp_coords_tensor.shape[1],  # Number of coordinates (x, y, z)
        block.shape[0]              # Number of model output elements
    ))
    
    # Compute gradients for each element in the block
    for e, element in enumerate(block):
        # Clear previous gradients
        if sp_coords_tensor.grad is not None:
            sp_coords_tensor.grad.zero_()
        
        # Compute gradients
        element.backward(retain_graph=True, create_graph=True)
        jacobian[:, :, e] = sp_coords_tensor.grad
    
    print(f"Gradients computed - Shape: {jacobian.shape}")
    print(f"Gradient range: [{jacobian.min():.6f}, {jacobian.max():.6f}]")
    
    # Visualize gradients for the first surface point (Z-coordinate)
    for surface_point_idx in range(1):  # Only first surface point
        gradient_z = jacobian[surface_point_idx, 2, :].detach().numpy()
        
        plot_gradient_grid(
            geo_model, 
            gradient_z, 
            f"Automatic Gradients - Surface Point {surface_point_idx} (Z)"
        )
