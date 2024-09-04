"""
Normal Prior, several observations
==================================


"""
# sphinx_gallery_thumbnail_number = -1

import arviz as az
import matplotlib.pyplot as plt
import pyro
import torch
from matplotlib.ticker import StrMethodFormatter
from pyro.infer import Predictive, NUTS, MCMC
import pyro.distributions as dist

from gempy_probability.plot_posterior import PlotPosterior

pyro.set_rng_seed(4003)


# %% md
# <a id="3.1"></a>
# ### Model definition
# Generally, models represent an abstraction of reality to answer a specific question, to fulfill a certain purpose, or to _simulate_ (mimic) a proces or multiple processes. What models share is the aspiration to be as realistic as possible, so they can be used for prognoses and to better understand a real-world system.
#
# Fitting of these models to acquired measurements or observations is called calibration and a standard procedure for improving a models reliability (**to answer the question it was designed for**).
#
# Models can also be seen as a general descriptor of correlation of observations in multiple dimensions. Complex systems with generally sparse data coverage (e.g. the subsurface) are difficult to reliably encode from the real-world in the numerical abstraction, i.e. a computational model.
#
# In a probabilistic framework, a model is a framework of different input distributions, which, as an output, has another probability distribution.
#

# %%
def model(distributions_family, data):
    if distributions_family == "normal_distribution":
        mu = pyro.sample('$\mu$', dist.Normal(2.07, 0.07))
    elif distributions_family in "uniform_distribution":
        mu = pyro.sample('$\mu$', dist.Uniform(0, 10))
    else:
        raise ValueError("distributions_family must be either 'normal_distribution' or 'uniform_distribution'")
    sigma = pyro.sample('$\sigma$', dist.Gamma(0.3, 3))
    y = pyro.sample('$y$', dist.Normal(mu, sigma), obs=data)
    return y


# %%
y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])

# %%
pyro.render_model(model, model_args=("normal_distribution", y_obs_list,))
# %% md
# Those observations are used for generating Distributions (~ probabilistic models) in [PyMC3](https://docs.pymc.io/en/v3/) which we encapsulate in the following function:
# %% md
# <a id="3.2"></a>
# ### Simplest probabilistic modeling
#
# Consider the simplest probabilistic model where the output $y$ of a model is a distribution. Let's assume, $y$ is a normal distribution, described by a mean $\mu$ and a standard deviation $\sigma$. Usually, those are considered scalar values, but they themselves can be distributions. This will yield a change of the width and position of the normal distribution $y$ with each iteration.
#
# As a reminder, a normal distribution is defined as:
#
# $$ y = \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2 \sigma ^2}} $$
#
# * $\mu$ mean (Normal distribution)
# * $\sigma$ standard deviation (Gamma distribution, Gamma log-likelihood)
# * $y$ Normal distribution
#
# With this constructed model, we are able to infer which model parameters will fit observations better by _optimizing_ for regions with high density mass. In addition (or even substituting) to data observations, informative values like prior simulations or expert knowledge can pour into the construction of the first $y$ distribution, the _prior_.
#
# There isn't a limitation about how "informative" a prior can or must be. Depending on the variance of the model's parameters and on the number of observations, a model will be more _prior driven_ or _data driven_.
# %% md
# Let's set up a `pymc3` model using the `thickness_observation` from above as observations and with $\mu$ and $\sigma$ being:
# * $\mu$ = Normal distribution with mean 2.08 and standard deviation 0.07
# * $\sigma$ = Gamma distribution with $\alpha$ (shape parameter) 0.3 and $\beta$ (rate parameter) 3
# * $y$ = Normal distribution with $\mu$, $\sigma$ and `thickness_observation_list` as observations
#
# A [Gamma distribution](https://docs.pymc.io/en/latest/api/distributions/generated/pymc.Gamma.html) can also be expressed by mean and standard deviation with $\alpha = \frac{\mu^2}{\sigma^2}$ and $\beta = \frac{\mu}{\sigma^2}$
#


#%% md
# <a id='3.3'></a>
# ###  One Observation: (:doc:`1.1_Intro_to_Bayesian_Inference`)
# ### Several Observations: (:doc:`1.2_Intro_to_Bayesian_Inference`)

# %%
# License
# =======
# The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:
# 
# https://opensource.org/licenses/BSD-3-Clause
# 
# The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:
# 
# https://creativecommons.org/licenses/by-nc/4.0/
# Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
