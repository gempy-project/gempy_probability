"""
2.1 - Only Pyro
===============

"""
import os

import arviz as az
# Importing auxiliary libraries
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.inspect import get_dependencies

from gempy_probability.plot_posterior import PlotPosterior, default_red, default_blue

# %%
# Model definition
# ----------------
# 
# Same problem as before, let’s assume the observations are layer
# thickness measurements taken on an outcrop. Now, in the previous example
# we chose a prior for the mean arbitrarily:
# :math:`𝜇∼Normal(mu=10.0, sigma=5.0)`–something that made sense for these
# specific data set. If we do not have any further information, keeping
# an uninformative prior and let the data to dictate the final value of
# the inference is the sensible way forward. However, this also enable to
# add information to the system by setting informative priors.
# 
# Imagine we get a borehole with the tops of the two interfaces of
# interest. Each of this data point will be a random variable itself since
# the accuracy of the exact 3D location will be always limited. Notice
# that this two data points refer to depth not to thickness–the unit of
# the rest of the observations. Therefore, the first step would be to
# perform a transformation of the parameters into the observations space.
# Naturally in this example a simple subtraction will suffice.
# 
# Now we can define the probabilistic models:
# 

# %%
# This is to make it work in sphinx gallery
cwd = os.getcwd()
if not 'examples' in cwd:
    path_dir = os.getcwd() + '/examples/tutorials/ch5_probabilistic_modeling'
else:
    path_dir = cwd

# %%

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)


def model(y_obs_list_):
    # Pyro models use the 'sample' function to define random variables
    mu_top = pyro.sample(r'$\mu_{top}$', dist.Normal(3.05, 0.2))
    sigma_top = pyro.sample(r"$\sigma_{top}$", dist.Gamma(0.3, 3.0))
    y_top = pyro.sample(r"y_{top}", dist.Normal(mu_top, sigma_top), obs=torch.tensor([3.02]))

    mu_bottom = pyro.sample(r'$\mu_{bottom}$', dist.Normal(1.02, 0.2))
    sigma_bottom = pyro.sample(r'$\sigma_{bottom}$', dist.Gamma(0.3, 3.0))
    y_bottom = pyro.sample(r'y_{bottom}', dist.Normal(mu_bottom, sigma_bottom), obs=torch.tensor([1.02]))

    mu_thickness = pyro.deterministic(r'$\mu_{thickness}$', mu_top - mu_bottom)  # Deterministic transformation
    sigma_thickness = pyro.sample(r'$\sigma_{thickness}$', dist.Gamma(0.3, 3.0))
    y_thickness = pyro.sample(r'y_{thickness}', dist.Normal(mu_thickness, sigma_thickness), obs=y_obs_list_)


dependencies = get_dependencies(
    model,
    model_args=y_obs_list[:1]
)

# 1. Prior Sampling
prior = Predictive(model, num_samples=10)(y_obs_list)

# Now you can run MCMC using NUTS to sample from the posterior
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
mcmc.run(y_obs_list)

# 3. Sample from Posterior Predictive
posterior_samples = mcmc.get_samples()
posterior_predictive = Predictive(model, posterior_samples)(y_obs_list)

# %%
data = az.from_pyro(
    posterior=mcmc,
    prior=prior,
    posterior_predictive=posterior_predictive
)

data

# %% 

az.plot_trace(data)
plt.show()

# %% 
# sphinx_gallery_thumbnail_number = 3
az.plot_density(
    data=[data, data.prior],
    shade=.9,
    data_labels=["Posterior", "Prior"],
    colors=[default_red, default_blue],
)

plt.show()

# %%
az.plot_density(
    data=[data.posterior_predictive, data.prior_predictive],
    shade=.9,
    var_names=[
        r'$\mu_{thickness}$'
    ],
    data_labels=["Posterior Predictive", "Prior Predictive"],
    colors=[default_red, default_blue],
)

plt.show()

# %%

p = PlotPosterior(data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=False)
p.plot_marginal(
    var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'],
    plot_trace=False,
    credible_interval=.70,
    kind='kde',
    marginal_kwargs={
        "bw": 1
    }
)
plt.show()

# %%
p = PlotPosterior(data)
p.create_figure(figsize=(9, 6), joyplot=True)
iteration = 99
p.plot_posterior(
    prior_var=['$\\mu_{top}$', '$\\mu_{bottom}$'],
    like_var=['$\mu_{thickness}$', '$\sigma_{thickness}$'],
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
az.plot_pair(data, divergences=False, var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'])
plt.show()
