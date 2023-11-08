import numpy as np

import gempy as gp
import gempy_viewer as gpv
import os

data_path = os.path.abspath('../../examples/')


def test_gradients_I():
    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        resolution=[20, 2, 10],
        refinement=1,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
            path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
        )
    )
    
    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False
    geo_model.interpolation_options.sigmoid_slope = 1000.
    
    gp.compute_model(
        gempy_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH
        )
    )

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
        # jacobian[:, e] = sp_coords_tensor.grad[:, 2]

    print("Gradients:", jacobian)

    for i in range(4):
        gradient_z_sp_1 = jacobian[i, 2, :].detach().numpy()
        # gradient_z_sp_1 = jacobian[i, :].detach().numpy()
        gpv.plot_2d(
            geo_model,
            show_topography=False,
            legend=False,
            show=True,
            override_regular_grid=gradient_z_sp_1,
            kwargs_lithology={
                'cmap': 'viridis',
                "plot_grid": True
            }
        )
