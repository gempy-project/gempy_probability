import numpy as np
import os

import gempy as gp
import gempy_viewer as gpv
from gempy.core.data.grid_modules import CustomGrid
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.modules.likelihoods import apparent_thickness_likelihood_II


def test_gempy_posterior_I():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'examples', 'tutorials', 'data'))
    geo_model = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        refinement=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=os.path.join(data_path, "2-layers", "2-layers_orientations.csv"),
            path_to_surface_points=os.path.join(data_path, "2-layers", "2-layers_surface_points.csv")
        )
    )

    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False

    x_loc = 6000
    y_loc = 0
    z_loc = np.linspace(0, 4000, 100)
    dz = (4000 - 0)/100
    xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])

    custom_grid = CustomGrid(xyz_coord)
    geo_model.grid.custom_grid = custom_grid
    gp.set_active_grid(
        grid=geo_model.grid,
        grid_type=[gp.data.Grid.GridTypes.CUSTOM]
    )
    gp.compute_model(gempy_model=geo_model)
    p2d = gpv.plot_2d(
        model=geo_model,
        show_topography=False,
        legend=False,
        show_lith=False,
        show=False
    )
    
    element_lith_id = 1.5
    thickness = apparent_thickness_likelihood_II(
        model_solutions=geo_model.solutions,
        distance=dz,
        element_lith_id=element_lith_id
    )
    
    print(thickness)

    p2d = gpv.plot_2d(geo_model, show_topography=False, legend=False, show=False)
    # --- New code: plot all custom grid points with lithology 1 ---
    #
    # Extract the simulated well values from the solution obtained.
    simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    #
    # Create a boolean mask for all grid points whose value is between 1 and 2.
    mask = (simulated_well >= element_lith_id) & (simulated_well < (element_lith_id + 1))
    #
    # Retrieve the custom grid coordinates.
    grid_coords = geo_model.grid.custom_grid.values
    #
    # Use the mask to get only the grid points corresponding to lith id 1.
    # Note: Convert the boolean mask to a NumPy array if necessary.
    coords_lith1 = grid_coords[mask]
    #
    # Plot these points on the figure (using the x and z columns for a 2D image).
    p2d.axes[0].scatter(
        coords_lith1[:, 0],
        coords_lith1[:, 2],
        marker='o',
        s=80,
        c='green',
        label='Lith id = 1',
        zorder=9
    )
    p2d.axes[0].legend()
    # --------------------------------------------------------------

    # Plot device location for reference.
    device_loc = np.array([[6e3, 0, 3700]])
    p2d.axes[0].scatter(
        device_loc[:, 0],
        device_loc[:, 2],
        marker='x',
        s=400,
        c='#DA8886',
        zorder=10
    )
    p2d.fig.show()
