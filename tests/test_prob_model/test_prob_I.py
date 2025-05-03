import os
import gempy as gp
import gempy_engine
import numpy as np


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

    # TODO: Convert this into an options preset
    geo_model.interpolation_options.uni_degree = 0
    geo_model.interpolation_options.mesh_extraction = False
    geo_model.interpolation_options.sigmoid_slope = 1100.

    x_loc = 6000
    y_loc = 0
    z_loc = np.linspace(0, 4000, 100)
    xyz_coord = np.array([[x_loc, y_loc, z] for z in z_loc])
    gp.set_custom_grid(geo_model.grid, xyz_coord=xyz_coord)

    # TODO: Make sure only the custom grid ins active

    gp.compute_model(
        gempy_model=geo_model,
        engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy)
    )

    # TODO: This is the part that has to go to a function no question
    # Probabilistic Geomodeling with Pyro
    # -----------------------------------
    # In this section, we introduce a probabilistic approach to geological modeling.
    # By using Pyro, a probabilistic programming language, we define a model that integrates
    # geological data with uncertainty quantification.

    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy.modules.data_manipulation.engine_factory import interpolation_input_from_structural_frame

    interpolation_input_copy: InterpolationInput = interpolation_input_from_structural_frame(geo_model)
    sp_coords_copy = interpolation_input_copy.surface_points.sp_coords
    # Change the backend to PyTorch for probabilistic modeling
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

    # %%
    # Running Prior Sampling and Visualization
    # ----------------------------------------
    # Prior sampling is an essential step in probabilistic modeling. 
    # It helps in understanding the distribution of our prior assumptions before observing any data.

    # %%
    # Prepare observation data
    import torch
    y_obs_list = torch.tensor([200, 210, 190])

    # %%
    # Run prior sampling and visualization
    from pyro.infer import Predictive
    import pyro
    import arviz as az
    import matplotlib.pyplot as plt

    from gempy_probability.modules.model_definition.model_examples import model
    predictive = Predictive(
        model=model,
        num_samples=50
    )

    prior = predictive(geo_model, sp_coords_copy, y_obs_list)

    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    plt.show()

    from pyro.infer import NUTS
    from pyro.infer import MCMC
    from pyro.infer.autoguide import init_to_mean

    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(
        model,
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
    mcmc.run(geo_model, sp_coords_copy, y_obs_list)

    posterior_samples = mcmc.get_samples()
    
    posterior_predictive_fn = Predictive(
        model=model,
        posterior_samples=posterior_samples
    )

    posterior_predictive = posterior_predictive_fn(geo_model, sp_coords_copy, y_obs_list)

    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
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
