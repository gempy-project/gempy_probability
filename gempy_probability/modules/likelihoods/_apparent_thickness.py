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
