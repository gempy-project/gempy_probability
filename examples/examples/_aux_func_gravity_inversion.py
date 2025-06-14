
import numpy as np

import gempy as gp
import xarray as xr
from vector_geology.model_building_functions import optimize_nuggets_for_group




def initialize_geo_model(structural_elements: list[gp.data.StructuralElement], extent: list[float],
                         topography: xr.Dataset, load_nuggets: bool = False
                         ) -> gp.data.GeoModel:
    structural_group_red = gp.data.StructuralGroup(
        name="Red",
        elements=[structural_elements[i] for i in [0, 4, 8]],
        structural_relation=gp.data.StackRelationType.ERODE
    )

    # Any, Probably we can decimize this an extra notch
    structural_group_green = gp.data.StructuralGroup(
        name="Green",
        elements=[structural_elements[i] for i in [5]],
        structural_relation=gp.data.StackRelationType.ERODE
    )

    # Blue range 2 cov 4
    structural_group_blue = gp.data.StructuralGroup(
        name="Blue",
        elements=[structural_elements[i] for i in [2, 3]],
        structural_relation=gp.data.StackRelationType.ERODE
    )

    structural_group_intrusion = gp.data.StructuralGroup(
        name="Intrusion",
        elements=[structural_elements[i] for i in [1]],
        structural_relation=gp.data.StackRelationType.ERODE
    )

    structural_groups = [structural_group_intrusion, structural_group_green, structural_group_blue, structural_group_red]
    structural_frame = gp.data.StructuralFrame(
        structural_groups=structural_groups[2:],
        color_gen=gp.data.ColorsGenerator()
    )
    # TODO: If elements do not have color maybe loop them on structural frame constructor?

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Tutorial_ch1_1_Basics',
        extent=extent,
        resolution=[20, 10, 20],
        refinement=5,  # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
        structural_frame=structural_frame
    )

    if topography is not None:
        gp.set_topography_from_arrays(
            grid=geo_model.grid,
            xyz_vertices=topography.vertex.values
        )

    if load_nuggets:
        import os

        project_root = os.getcwd()
        path_to_temp = os.path.join(project_root, "../temp")
        apply_optimized_nuggets(
            geo_model=geo_model,
            loaded_nuggets_red=(np.load(path_to_temp + "/nuggets_Red.npy")),
            loaded_nuggets_blue=(np.load(path_to_temp + "/nuggets_Blue.npy")),
            loaded_nuggets_green=(np.load(path_to_temp + "/nuggets_Green.npy"))
        )

    geo_model.structural_frame.get_element_by_name("KKR").color = "#A46283"
    geo_model.structural_frame.get_element_by_name("LGR").color = "#6394A4"
    geo_model.structural_frame.get_element_by_name("WAL").color = "#72A473"
    geo_model.structural_frame.get_element_by_name("ABL").color = "#1D3943"
    geo_model.structural_frame.basement_color = "#8B4220"

    geo_model.update_transform()

    return geo_model

def setup_geophysics(env_path: str, geo_model: gp.data.GeoModel, densities: list[float]):
    import pandas as pd
    from dotenv import dotenv_values
    from gempy_engine.core.backend_tensor import BackendTensor
    
    config = dotenv_values()

    df = pd.read_csv(
        filepath_or_buffer=config.get(env_path),
        sep=',',
        header=0
    )

    # Remove the items that have X > 5650000
    df = df[df['X'] < 565000]

    interesting_columns = df[['X', 'Y', 'Bouguer_267_complete']]
    # %%
    device_location = interesting_columns[['X', 'Y']]
    device_location['Z'] = 0 # * stack 0 to the z axis

    gp.set_centered_grid(
        grid=geo_model.grid,
        centers=device_location,
        resolution=np.array([10, 10, 15]),
        radius=np.array([5000, 5000, 5000])
    )

    gravity_gradient = gp.calculate_gravity_gradient(geo_model.grid.centered_grid)

    # %%
    geo_model.geophysics_input = gp.data.GeophysicsInput(
        tz=BackendTensor.t.array(gravity_gradient),
        densities=BackendTensor.t.array(densities),
    )

    return interesting_columns
