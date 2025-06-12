import pyro.distributions as dist
import gempy as gp
from ._apparent_thickness import apparent_thickness_likelihood


def gravity_likelihood(geo_model: gp.data.Solutions):
    # e.g. multivariate normal with precomputed cov matrix
    raise NotImplementedError("This function is not yet implemented")
    return dist.MultivariateNormal(simulated_gravity, covariance_matrix)

def thickness_likelihood(solutions: gp.data.Solutions) -> dist:
    # e.g. multivariate normal with precomputed cov matrix
    simulated_thickness = apparent_thickness_likelihood(solutions)
    return dist.Normal(simulated_thickness, 25)
