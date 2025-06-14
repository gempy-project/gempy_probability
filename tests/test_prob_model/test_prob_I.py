import numpy as np
import os
import pyro.distributions as dist
import torch

import gempy as gp
from gempy_engine.core.backend_tensor import BackendTensor


def test_basic_gempy_I() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'examples', 'tutorials', 'data'))
    geo_model = gp.create_geomodel(
        project_name='Wells',
        extent=[0, 12000, -500, 500, 0, 4000],
        refinement=3,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=os.path.join(data_path, "2-layers", "2-layers_orientations.csv"),
            path_to_surface_points=os.path.join(data_path, "2-layers", "2-layers_surface_points.csv")
        )
    )

    # TODO: Make this a more elegant way 
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    # TODO: Convert this into an options preset
    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False
    geo_model.interpolation_options.sigmoid_slope = 1100.

    # region Minimal grid for the specific likelihood function
    x_loc = 6000
    y_loc = 0
    z_loc = np.linspace(0, 4000, 100)
    xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])
    gp.set_custom_grid(geo_model.grid, xyz_coord=xyz_coord)
    # endregion

    geo_model.grid.active_grids = gp.data.Grid.GridTypes.CUSTOM
    assert geo_model.grid.values.shape[0] == 100, "Custom grid should have 100 cells"
    gp.compute_model(gempy_model=geo_model, validate_serialization=False)
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    normal = dist.Normal(
        loc=(geo_model.surface_points_copy_transformed.xyz[0, 2]),
        scale=torch.tensor(0.1, dtype=torch.float64)
    )

    from gempy_probability.modules.model_definition.model_examples import two_wells_prob_model_I
    _prob_run(
        geo_model=geo_model,
        prob_model=two_wells_prob_model_I,
        normal=normal,
        y_obs_list=torch.tensor([200, 210, 190])
    )


def _prob_run(geo_model: gp.data.GeoModel, prob_model: callable,
              normal: torch.distributions.Distribution, y_obs_list: torch.Tensor) -> None:
    # Run prior sampling and visualization
    from pyro.infer import Predictive
    import pyro
    import arviz as az
    import matplotlib.pyplot as plt

    from pyro.infer import NUTS
    from pyro.infer import MCMC
    from pyro.infer.autoguide import init_to_mean

    # region prior sampling
    predictive = Predictive(
        model=prob_model,
        num_samples=50
    )
    prior = predictive(geo_model, normal, y_obs_list)
    # print("Number of interpolations: ", geo_model.counter)

    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    plt.show()

    # endregion

    # region inference
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(
        prob_model,
        step_size=0.0085,
        adapt_step_size=True,
        target_accept_prob=0.9,
        max_tree_depth=10,
        init_strategy=init_to_mean
    )
    mcmc = MCMC(
        kernel=nuts_kernel,
        num_samples=200,
        warmup_steps=50,
        disable_validation=False
    )
    mcmc.run(geo_model, normal, y_obs_list)
    posterior_samples = mcmc.get_samples()
    posterior_predictive_fn = Predictive(
        model=prob_model,
        posterior_samples=posterior_samples
    )
    posterior_predictive = posterior_predictive_fn(geo_model, normal, y_obs_list)

    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    # endregion

    # print("Number of interpolations: ", geo_model.counter)

    if True: # * Save the arviz data
        data.to_netcdf("arviz_data.nc")
        
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

    # Test posterior mean values
    posterior_top_mean = float(data.posterior[r'$\mu_{top}$'].mean())
    target_top = 0.00875
    target_thickness = 223
    assert abs(posterior_top_mean - target_top) < 0.0200, f"Top layer mean {posterior_top_mean} outside expected range"
    posterior_thickness_mean = float(data.posterior_predictive[r'$\mu_{thickness}$'].mean())
    assert abs(posterior_thickness_mean - target_thickness) < 5, f"Thickness mean {posterior_thickness_mean} outside expected range"
    # Test convergence diagnostics
    assert float(data.sample_stats.diverging.sum()) == 0, "MCMC sampling has divergences"

    print("Posterior mean values:")
    print(f"Top layer mean: {posterior_top_mean}")
    print(f"Thickness mean: {posterior_thickness_mean}")
    print("MCMC convergence diagnostics:")
    print(f"Divergences: {float(data.sample_stats.diverging.sum())}")
