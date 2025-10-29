from .modules.model_definition.prob_model_factory import make_gempy_pyro_model, GemPyPyroModel
from .modules import likelihoods
from .api.model_runner import run_predictive, run_mcmc_for_NUTS, run_nuts_inference

"""
Module initialisation for GemPy Probability
"""
import sys

# * Assert at least python 3.10
assert sys.version_info[0] >= 3 and sys.version_info[1] >= 10, "GemPy Probability requires Python 3.10 or higher"

# Import version, with fallback if not generated yet
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# =================== CORE ===================
# Import your core modules here

# =================== API ===================

if __name__ == '__main__':
    pass
