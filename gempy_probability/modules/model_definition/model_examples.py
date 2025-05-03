import gempy.core.data
import gempy_engine
from gempy.modules.data_manipulation.engine_factory import interpolation_input_from_structural_frame


def model(geo_model: gempy.core.data.GeoModel, sp_coords_copy, y_obs_list):
    """
    This Pyro model represents the probabilistic aspects of the geological model.
    It defines a prior distribution for the top layer's location and 
    computes the thickness of the geological layer as an observed variable.
    """
    # Define prior for the top layer's location:
    prior_mean = sp_coords_copy[0, 2]
    import pyro
    import pyro.distributions as dist
    import torch
    mu_top = pyro.sample(r'$\mu_{top}$', dist.Normal(prior_mean, torch.tensor(0.02, dtype=torch.float64)))

    # Update the model with the new top layer's location
    interpolation_input = interpolation_input_from_structural_frame(geo_model)
    interpolation_input.surface_points.sp_coords = torch.index_put(
        interpolation_input.surface_points.sp_coords,
        (torch.tensor([0]), torch.tensor([2])),
        mu_top
    )

    # Compute the geological model
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )

    # Compute and observe the thickness of the geological layer
    simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    thickness = simulated_well.sum()
    pyro.deterministic(r'$\mu_{thickness}$', thickness.detach())
    y_thickness = pyro.sample(r'$y_{thickness}$', dist.Normal(thickness, 50), obs=y_obs_list)
