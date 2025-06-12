import numpy as np

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
import os
import matplotlib.pyplot as plt
import pytest

from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
from gempy_engine.core.backend_tensor import BackendTensor

data_path = os.path.abspath('../../examples/tutorials/')
SIGMOID_SLOPE = 1000


def _model(sigoid_slope=100, plot=False):
    gempy_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        resolution=np.array([10, 1, 10]) * 1,
        # refinement=1,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
            path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
        )
    )
    gempy_model.interpolation_options.uni_degree = 0
    gempy_model.interpolation_options.mesh_extraction = False
    gempy_model.interpolation_options.sigmoid_slope = sigoid_slope
    engine_config = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH)
    
    BackendTensor.change_backend_gempy(
        engine_backend=engine_config.backend,
        use_gpu=engine_config.use_gpu,
        dtype=engine_config.dtype
    )

    interpolation_input = interpolation_input_from_structural_frame(gempy_model)

    gempy_model.taped_interpolation_input = interpolation_input  # * This is used for gradient tape

    sp_coords_tensor = gempy_model.taped_interpolation_input.surface_points.sp_coords
    sp_coords_tensor.requires_grad = True

    gempy_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=gempy_model.interpolation_options,
        data_descriptor=gempy_model.input_data_descriptor,
        geophysics_input=gempy_model.geophysics_input,
    )

    if plot:
        gpv.plot_2d(gempy_model, show_scalar=False, kwargs_lithology={
            "plot_grid": True,
        })
        plt.show()
    return gempy_model


def test_gradients_numpy():
    geo_model = _model(SIGMOID_SLOPE, plot=True)

    gpv.plot_2d(geo_model, kwargs_lithology={"plot_grid": True})

    par_val = geo_model.surface_points_copy.data['Z'][0]
    var = 50
    point_n = 0

    values_to_compute = np.linspace(par_val - var, par_val + var, 30)
    arrays = np.array([])
    for i, val in enumerate(values_to_compute):
        gp.modify_surface_points(
            geo_model,
            slice=point_n,
            Z=val
        )
        sol = gp.compute_model(geo_model)
        arrays = np.append(arrays, sol.octrees_output[0].last_output_center.final_block[:100])


    # Plot values
    iter_a = arrays.reshape(-1, 10, 10)
    fig = plt.figure()
    plt.axis('off')

    for i in range(10):
        for j in range(10):
            ax = plt.subplot(10, 10, (9 - j) * 10 + i + 1)  # ((1+j)*10)-i)
            ax.plot(values_to_compute, iter_a[:, i, j], '.')
            ax.axvline(par_val, ymax=3, color='r')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # ax.sharex()
            ax.set_ylim(0, 3)
    
    plt.show()
    
    # TODO:
    grads = np.diff(arrays.reshape(-1, 10, 10), axis=0)

    gp.modify_surface_points(
        geo_model,
        slice=point_n,
        Z=par_val
    )
    gp.compute_model(geo_model)
    gradient_z_sp_1 = grads[15]/(2*var/30)

    max_abs_val = np.max(np.abs(gradient_z_sp_1))
    p = gpv.plot_2d(
        geo_model,
        show_topography=False,
        legend=True,
        show=True,
        override_regular_grid=gradient_z_sp_1,
        kwargs_lithology={
            'cmap': 'seismic',
            'norm': None,
            "plot_grid": True,
            "vmin": -max_abs_val,
            "vmax": max_abs_val
        }
    )



def test_gradients_I():
    geo_model = _model(sigoid_slope=SIGMOID_SLOPE, plot=True)
    # * This is the activated block
    block = geo_model.solutions.octrees_output[0].last_output_center.final_block

    # * This is the scalar field
    # block = geo_model.solutions.octrees_output[0].last_output_center.exported_fields.scalar_field

    sp_coords_tensor = geo_model.taped_interpolation_input.surface_points.sp_coords

    import torch
    jacobian = torch.zeros((
        sp_coords_tensor.shape[0],
        sp_coords_tensor.shape[1],
        block.shape[0])
    )

    if False:
        sp_coords_tensor.register_hook(lambda x: print("I am here!", x))

    for e, element in enumerate(block):
        if sp_coords_tensor.grad is not None:
            sp_coords_tensor.grad.zero_()
        element.backward(retain_graph=True, create_graph=True)
        jacobian[:, :, e] = sp_coords_tensor.grad


    print("Gradients:", jacobian)
    print("Max min:", jacobian.max(), jacobian.min())

    for i in range(0, 1):
        gradient_z_sp_1 = jacobian[i, 2, :].detach().numpy()
        
        max_abs_val = np.max(np.abs(gradient_z_sp_1))
        p = gpv.plot_2d(
            geo_model,
            show_topography=False,
            legend=False,
            show=True,
            override_regular_grid=gradient_z_sp_1[:100],
            kwargs_lithology={
                'cmap': 'seismic',
                "norm": None,
                "plot_grid": True,
                "vmin": -max_abs_val,
                "vmax": max_abs_val
            }
        )


