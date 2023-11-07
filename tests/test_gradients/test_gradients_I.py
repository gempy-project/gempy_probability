import gempy as gp
import gempy_viewer as gpv
import os

data_path = os.path.abspath('../../examples/')


def test_gradients_I():
    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, 0, 12000, 0, 4000],
        refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
            path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
        )
    )
    
    gp.compute_model(
        gempy_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH
        )
    )
    
    gpv.plot_2d(geo_model, show_topography=False, legend=False, show=True)
