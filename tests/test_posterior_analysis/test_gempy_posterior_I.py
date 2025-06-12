import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os

import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_viewer.API._plot_2d_sections_api import plot_sections
from gempy_viewer.core.data_to_show import DataToShow


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

    data = az.from_netcdf("../arviz_data.nc")

    if PLOT := False:
        _plot_posterior(data)

    gp.compute_model(gempy_model=geo_model)
    p2d = gpv.plot_2d(
        model=geo_model,
        show_topography=False,
        legend=False,
        show_lith=False,
        show=False
    )

    posterior_top_mean_z: np.ndarray = (data.posterior[r'$\mu_{top}$'].values[0, :])
    xyz = np.zeros((posterior_top_mean_z.shape[0], 3))
    xyz[:, 2] = posterior_top_mean_z
    world_coord = geo_model.input_transform.apply_inverse(xyz)
    i = 0
    for i in range(0, 200, 5):
        gp.modify_surface_points(
            geo_model=geo_model,
            slice=0,
            Z=world_coord[i, 2]
        )
        gp.compute_model(gempy_model=geo_model)

        plot_sections(
            gempy_model=geo_model,
            sections_data=p2d.section_data_list,
            data_to_show=DataToShow(
                n_axis=1,
                show_data=True,
                show_surfaces=True,
                show_lith=False
            ),
            kwargs_boundaries={
                    "linewidth": 0.5,
                    "alpha"    : 0.1,
            },
        )

    p2d.fig.show()

    pass


def _plot_posterior(data):
    az.plot_trace(data)
    plt.show()
    from gempy_probability.modules.plot.plot_posterior import default_red, default_blue
    az.plot_density(
        data=[data.posterior_predictive, data.prior_predictive],
        shade=.9,
        var_names=[r'$\mu_{thickness}$'],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
    )
    plt.show()
    az.plot_density(
        data=[data, data.prior],
        shade=.9,
        hdi_prob=.99,
        data_labels=["Posterior", "Prior"],
        colors=[default_red, default_blue],
    )
    plt.show()
