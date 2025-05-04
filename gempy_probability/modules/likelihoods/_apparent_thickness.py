import gempy as gp
import torch
import pyro

def apparent_thickness_likelihood(model_solutions: gp.data.Solutions) -> torch.Tensor:
    """
    This function computes the thickness of the geological layer.
    
    Notes: This is not completed
    """
    simulated_well = model_solutions.octrees_output[0].last_output_center.custom_grid_values
    thickness = simulated_well.sum()
    pyro.deterministic(
        name=r'$\mu_{thickness}$',
        value=thickness.detach()  # * This is only for az to track progress
    )
    return thickness

def apparent_thickness_likelihood_II(model_solutions: gp.data.Solutions, distance: float, element_lith_id: float) -> torch.Tensor:
    # TODO: element_lith_id should be an structured element  
    simulated_well = model_solutions.octrees_output[0].last_output_center.custom_grid_values

    # Create a boolean mask for all values between element_lith_id and element_lith_id+1
    mask = (simulated_well >= element_lith_id) & (simulated_well < (element_lith_id + 1))

    # Count these values and compute thickness as the product of the count and the distance
    thickness = mask.sum() * distance
    return thickness
