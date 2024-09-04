"""
2.1 - Only Pyro
===============


Model definition
----------------

Same problem as before, let’s assume the observations are layer
thickness measurements taken on an outcrop. Now, in the previous example
we chose a prior for the mean arbitrarily:
:math:`𝜇∼Normal(mu=10.0, sigma=5.0)`–something that made sense for these
specific data set. If we do not have any further information, keeping
an uninformative prior and let the data to dictate the final value of
the inference is the sensible way forward. However, this also enable to
add information to the system by setting informative priors.

Imagine we get a borehole with the tops of the two interfaces of
interest. Each of this data point will be a random variable itself since
the accuracy of the exact 3D location will be always limited. Notice
that this two data points refer to depth not to thickness–the unit of
the rest of the observations. Therefore, the first step would be to
perform a transformation of the parameters into the observations space.
Naturally in this example a simple subtraction will suffice.

Now we can define the probabilistic models:

"""
# sphinx_gallery_thumbnail_number = -2

# %%
# Importing Necessary Libraries
# -----------------------------

import os
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.inspect import get_dependencies
from gempy_probability.plot_posterior import PlotPosterior, default_red, default_blue
import arviz as az

# %%
# Introduction to the Problem
# ---------------------------
# In this example, we are considering layer thickness measurements taken on an outcrop as our observations.
# We use a probabilistic approach to model these observations, allowing for the inclusion of prior knowledge
# and uncertainty quantification.

# Setting the working directory for sphinx gallery
cwd = os.getcwd()
if 'examples' not in cwd:
    path_dir = os.getcwd() + '/examples/tutorials/ch5_probabilistic_modeling'
else:
    path_dir = cwd

# %%
# Defining the Observations and Model
# -----------------------------------
# The observations are layer thickness measurements. We define a Pyro probabilistic model
# that uses Normal and Gamma distributions to model the top and bottom interfaces of the layers
# and their respective uncertainties.

# Defining observed data
y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09, 2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)

# Defining the probabilistic model
def model(y_obs_list_):
    mu_top = pyro.sample(r'$\mu_{top}$', dist.Normal(3.05, 0.2))
    sigma_top = pyro.sample(r"$\sigma_{top}$", dist.Gamma(0.3, 3.0))
    y_top = pyro.sample(r"y_{top}", dist.Normal(mu_top, sigma_top), obs=torch.tensor([3.02]))

    mu_bottom = pyro.sample(r'$\mu_{bottom}$', dist.Normal(1.02, 0.2))
    sigma_bottom = pyro.sample(r'$\sigma_{bottom}$', dist.Gamma(0.3, 3.0))
    y_bottom = pyro.sample(r'y_{bottom}', dist.Normal(mu_bottom, sigma_bottom), obs=torch.tensor([1.02]))

    mu_thickness = pyro.deterministic(r'$\mu_{thickness}$', mu_top - mu_bottom)
    sigma_thickness = pyro.sample(r'$\sigma_{thickness}$', dist.Gamma(0.3, 3.0))
    y_thickness = pyro.sample(r'y_{thickness}', dist.Normal(mu_thickness, sigma_thickness), obs=y_obs_list_)

# Exploring model dependencies
dependencies = get_dependencies(model, model_args=y_obs_list[:1])

# %%
pyro.render_model(model, model_args=(y_obs_list,))

# %%
# Prior Sampling
# --------------
# Prior sampling is performed to understand the initial distribution of the model parameters
# before considering the observed data.

# Prior sampling
prior = Predictive(model, num_samples=10)(y_obs_list)

# %%
# Running MCMC Sampling
# ---------------------
# Markov Chain Monte Carlo (MCMC) sampling is used to sample from the posterior distribution,
# providing insights into the distribution of model parameters after considering the observed data.

# Running MCMC using the NUTS algorithm
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
mcmc.run(y_obs_list)

# %%
# Posterior Predictive Sampling
# -----------------------------
# After obtaining the posterior samples, we perform posterior predictive sampling.
# This step allows us to make predictions based on the posterior distribution.

# Sampling from the posterior predictive distribution
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(model, posterior_samples)(y_obs_list)

# %%
# Visualizing the Results
# -----------------------
# We use ArviZ, a library for exploratory analysis of Bayesian models, to visualize
# the results of our probabilistic model.

# Creating a data object for ArviZ
data = az.from_pyro(
    posterior=mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)

# Plotting trace of the sampled parameters
az.plot_trace(data)
plt.show()

# %%
# Density Plots of Posterior and Prior
# ------------------------------------
# Density plots provide a visual representation of the distribution of the sampled parameters.
# Comparing the posterior and prior distributions allows us to assess the impact of the observed data.

# Plotting density of posterior and prior distributions
az.plot_density(
    data=[data, data.prior],
    shade=.9,
    data_labels=["Posterior", "Prior"],
    colors=[default_red, default_blue],
)
plt.show()

# %%
# Can we see a correlation between the posteriors of $\mu_{bottom}$ and $\mu_{top}$?
#
# Density Plots of Posterior Predictive and Prior Predictive
# ----------------------------------------------------------
# These plots show the distribution of the posterior predictive and prior predictive checks.
# They help in evaluating the performance and validity of the probabilistic model.

# Plotting density of posterior predictive and prior predictive
az.plot_density(
    data=[data.posterior_predictive, data.prior_predictive],
    shade=.9,
    var_names=[r'$\mu_{thickness}$'],
    data_labels=["Posterior Predictive", "Prior Predictive"],
    colors=[default_red, default_blue],
)
plt.show()

# %%
# Marginal Distribution Plots
# ---------------------------
# Marginal distribution plots provide insights into the distribution of individual parameters.
# These plots help in understanding the uncertainty and variability in the parameter estimates.

# Creating marginal distribution plots
p = PlotPosterior(data)
p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=False)
p.plot_marginal(
    var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'],
    plot_trace=False,
    credible_interval=.70,
    kind='kde',
    marginal_kwargs={"bw": 1}
)
plt.show()

# %%
# Posterior Distribution Visualization
# ------------------------------------
# This section provides a more detailed visualization of the posterior distributions
# of the parameters, integrating different aspects of the probabilistic model.

# Visualizing the posterior distributions
p = PlotPosterior(data)
p.create_figure(figsize=(9, 6), joyplot=True)
iteration = 99
p.plot_posterior(
    prior_var=['$\\mu_{top}$', '$\\mu_{bottom}$'],
    like_var=['$\\mu_{top}$', '$\\mu_{bottom}$'],
    obs='y_{thickness}',
    iteration=iteration,
    marginal_kwargs={
        "credible_interval": 0.94,
        'marginal_kwargs': {"bw": 1},
        'joint_kwargs': {"bw": 1}
    }
)
plt.show()

# %%
# Pair Plot of Key Parameters
# ---------------------------
# Pair plots are useful to visualize the relationships and correlations between different parameters.
# They help in understanding how parameters influence each other in the probabilistic model.

# Creating a pair plot for selected parameters
az.plot_pair(data, divergences=False, var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'])
plt.show()
