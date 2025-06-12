import gempy as gp
import gempy.core.data
import gempy_engine
from gempy.modules.data_manipulation.engine_factory import interpolation_input_from_structural_frame
from gempy_probability.modules.likelihoods import apparent_thickness_likelihood

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
    
    if True: # ?? I need to figure out if we need the index_put or not
        indices__ = (torch.tensor([0]), torch.tensor([2]))  # * This has to be Tensors
        new_tensor: torch.Tensor = torch.index_put(
            input=interpolation_input.surface_points.sp_coords,
            indices=indices__,
            values=mu_top
        )
        interpolation_input.surface_points.sp_coords = new_tensor
    else:
        interpolation_input.surface_points.sp_coords[0, 2] = mu_top

    # endregion

    # region Forward model computation
    
    geo_model.counter +=1

    # * Compute the geological model
    geo_model.solutions = gempy_engine.compute_model(
        interpolation_input=interpolation_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input,
    )
    # if i does not exist init
    
    
    # Compute and observe the thickness of the geological layer
    model_solutions: gp.data.Solutions = geo_model.solutions
    thickness = apparent_thickness_likelihood(model_solutions)

    # endregion

    posterior_dist_normal = dist.Normal(thickness, 25)
    
    # * This is used automagically by pyro to compute the log-probability
    y_thickness = pyro.sample(
        name=r'$y_{thickness}$',
        fn=posterior_dist_normal,
        obs=y_obs_list
    )


