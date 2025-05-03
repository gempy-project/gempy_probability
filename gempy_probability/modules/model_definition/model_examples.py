import gempy.core.data
import gempy_engine
from gempy.modules.data_manipulation.engine_factory import interpolation_input_from_structural_frame

import pyro
import pyro.distributions as dist
import torch


def model(geo_model: gempy.core.data.GeoModel, normal, y_obs_list):
    """
    This Pyro model represents the probabilistic aspects of the geological model.
    It defines a prior distribution for the top layer's location and 
    computes the thickness of the geological layer as an observed variable.
    """
    # Define prior for the top layer's location:
    # region Prior definition

    mu_top = pyro.sample(
        name=r'$\mu_{top}$',
        fn=normal
    )
    # endregion

    # region Prior injection into the gempy model

    # * Update the model with the new top layer's location
    interpolation_input = interpolation_input_from_structural_frame(geo_model)
    interpolation_input.surface_points.sp_coords = torch.index_put(
        input=interpolation_input.surface_points.sp_coords,
        indices=(torch.tensor([0]), torch.tensor([2])),
        values=mu_top
    )
    # interpolation_input.surface_points.sp_coords[0, 2] = mu_top

    # endregion

    # region Forward model computation

    # * Compute the geological model
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )

    # Compute and observe the thickness of the geological layer
    simulated_well = geo_model.solutions.octrees_output[0].last_output_center.custom_grid_values
    thickness = simulated_well.sum()
    pyro.deterministic(
        name=r'$\mu_{thickness}$',
        value=thickness.detach()
    )

    # endregion

    y_thickness = pyro.sample(
        name=r'$y_{thickness}$',
        fn=dist.Normal(thickness, 25),
        obs=y_obs_list
    )
