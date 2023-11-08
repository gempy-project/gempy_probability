import numpy as np

import gempy as gp
import gempy_viewer as gpv
import os
import matplotlib.pyplot as plt

data_path = os.path.abspath('../../examples/')


def model():
    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        resolution=[10, 1, 10],
        refinement=1,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
            path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
        )
    )
    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False
    geo_model.interpolation_options.sigmoid_slope = 100.
    gp.compute_model(
        gempy_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH
        )
    )
    return geo_model


def test_gradients_numpy():
    geo_model = model()

    gpv.plot_2d(geo_model, kwargs_lithology={"plot_grid": True})

    par_val = geo_model.surface_points.data['Z'][0]
    var = 500

    values_to_compute = np.linspace(par_val - var, par_val + var, 30)
    arrays = np.array([])
    for i, val in enumerate(values_to_compute):
        gp.modify_surface_points(
            geo_model,
            slice=0,
            Z=val
        )
        sol = gp.compute_model(geo_model)
        arrays = np.append(arrays, sol.octrees_output[0].last_output_center.final_block)


    # Plot values
    iter_a = arrays.reshape(-1, 10, 10)
    fig = plt.figure()
    plt.axis('off')

    for i in range(10):
        for j in range(10):
            # print(i,j, (9-j)*10+i+1)
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
        slice=0,
        Z=par_val
    )
    gp.compute_model(geo_model)
    gradient_z_sp_1 = grads[4]
    gpv.plot_2d(
        geo_model,
        show_topography=False,
        legend=True,
        show=True,
        override_regular_grid=gradient_z_sp_1,
        kwargs_lithology={
            'cmap': 'viridis',
            "plot_grid": True
        }
    )


def test_gradients_I():
    geo_model = model()

    block = geo_model.solutions.octrees_output[0].last_output_center.final_block
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
        element.backward(retain_graph=True)
        jacobian[:, :, e] = sp_coords_tensor.grad

    print("Gradients:", jacobian)

    for i in range(1):
        gradient_z_sp_1 = jacobian[i, 2, :].detach().numpy()
        gpv.plot_2d(
            geo_model,
            show_topography=False,
            legend=True,
            show=True,
            override_regular_grid=gradient_z_sp_1,
            kwargs_lithology={
                'cmap': 'viridis',
                "plot_grid": True
            }
        )
