import torch
from pyro.distributions import Distribution

import gempy_engine
from gempy.modules.data_manipulation import interpolation_input_from_structural_frame
import gempy as gp
from gempy_engine.core.data.interpolation_input import InterpolationInput


def run_gempy_forward(interp_input: InterpolationInput, geo_model: gp.data.GeoModel) -> gp.data.Solutions:

    # compute model
    sol = gempy_engine.compute_model(
        interpolation_input=interp_input,
        options=geo_model.interpolation_options,
        data_descriptor=geo_model.input_data_descriptor,
        geophysics_input=geo_model.geophysics_input
    )
    
    return sol


