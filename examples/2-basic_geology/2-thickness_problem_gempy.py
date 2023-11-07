import gempy as gp
import gempy_viewer as gpv
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
data_path = os.path.abspath('../')


# %%
def plot_geo_setting_well(geo_model):
    device_loc = np.array([[6e3, 0, 3700]])
    p2d = gpv.plot_2d(geo_model, show_topography=False, legend=False, show=False)

    well_1 = 3.41e3
    well_2 = 3.6e3
    p2d.axes[0].scatter([3e3], [well_1], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter([9e3], [well_2], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter(device_loc[:, 0], device_loc[:, 2], marker='x', s=400,
                        c='#DA8886', zorder=10)

    p2d.axes[0].vlines(3e3, .5e3, well_1, linewidth=4, color='gray')
    p2d.axes[0].vlines(9e3, .5e3, well_2, linewidth=4, color='gray')
    p2d.axes[0].vlines(3e3, .5e3, well_1)
    p2d.axes[0].vlines(9e3, .5e3, well_2)

    p2d.fig.show()


# %%
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Wells',
    extent=[0, 12000, 0, 12000, 0, 4000],
    refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/2-layers/2-layers_orientations.csv",
        path_to_surface_points=data_path + "/data/2-layers/2-layers_surface_points.csv"
    )
)


plot_geo_setting_well(geo_model=geo_model)