import pyro
import torch
import gempy as gp

def gravity_inversion_likelihood(model_solutions: gp.data.Solutions) -> torch.Tensor:
    simulated_geophysics = model_solutions.gravity
    pyro.deterministic(r'$\mu_{gravity}$', simulated_geophysics)
    
    return simulated_geophysics