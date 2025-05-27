import gempy as gp
import gempy.core.data
import gempy_engine
from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
from ..likelihoods import apparent_thickness_likelihood

import pyro
import torch

from gempy_probability.modules.model_definition.prob_model_factory import make_pyro_model


def two_wells_prob_model_I(geo_model: gempy.core.data.GeoModel, normal, y_obs_list):
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



import pyro.distributions as dist

# 1) Define your priors
gravity_priors = {
        # e.g. density contrast of layer index 3
        "mu_density": dist.Normal(2.62, 0.5)
}

# 2) Wrap your forward pass into a small fn
def run_gempy_forward(samples, geo_model):
    
    mu_top = samples["mu_density"]

    interp_input = interpolation_input_from_structural_frame(geo_model)
    
    indices__ = (torch.tensor([0]), torch.tensor([2]))  # * This has to be Tensors
    new_tensor: torch.Tensor = torch.index_put(
        input=interp_input.surface_points.sp_coords,
        indices=indices__,
        values=mu_top
    )
    interp_input.surface_points.sp_coords = new_tensor

    # compute model
    sol = gempy_engine.compute_model(
        interpolation_input=interp_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input
    )

    thickness = apparent_thickness_likelihood(sol)
    return thickness

# 3) Define your likelihood factory
def gravity_likelihood(simulated_gravity):
    # e.g. multivariate normal with precomputed cov matrix
    return dist.MultivariateNormal(simulated_gravity, covariance_matrix)

def thickness_likelihood(simulated_thickness):
    # e.g. multivariate normal with precomputed cov matrix
    return dist.Normal(simulated_thickness, 25)

# 4) Build the model
pyro_gravity_model = make_pyro_model(
    priors=gravity_priors,
    forward_fn=run_gempy_forward,
    likelihood_fn=thickness_likelihood,
    obs_name="obs_gravity"
)

