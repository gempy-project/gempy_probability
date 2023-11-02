"""
Intro to Bayesian Inference
===========================


Let's start with the simplest model in structural geology we have been able to come up and trying to be
agonizingly pedantic about it. You want to model the thickness of one layer on one specific outcrop and we
want to be right. To be sure, we will go during 10 years once per month with a tape measure and we will
write down the value on the tape. Finally, 10 years have passed and we are ready to give a conclusive answer.
However out of 120 observations there are 120 distinct numbers. The first thing we need to reconcile is that any
measurements of any phenomenon are isolated. Changes in the environment---maybe some earthquake---or in the way we
gather the data---we changed the measurement device or simply our eyes got tired over the years---will also influence
which observations \( y \) we end up with. No matter the exact source of the variability, all these processes define the
Observation space, \( Y \). Any observational space is going to have some kind of structure that can be modeled for a
probability density function called *data generating process* \( \pi^\dagger \). In other words, there is a latent
process---which is a complex combination of several dynamics and limitations---that every time we perform a measurement
w;ll yield a value following certain probability function. Now, to the question what is the thickness of the layer, the
answer that better describe the 120 measurements will have to be a probability density function instead a single value
but how can we know what probability function is the right one?

"""

import torch
import pyro
import pyro.distributions as dist
from matplotlib.ticker import StrMethodFormatter
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az
import matplotlib.pyplot as plt

from gempy_probability.plot_posterior import PlotPosterior

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)


def model(conf, data):
    if conf in ['n_s', 'n_o']:
        mu = pyro.sample('$\mu$', dist.Normal(2.08, 0.07))
    elif conf in ['u_s', 'u_o']:
        mu = pyro.sample('$\mu$', dist.Uniform(0, 10))
    sigma = pyro.sample('$\sigma$', dist.Gamma(0.3, 3))
    y = pyro.sample('$y$', dist.Normal(mu, sigma), obs=data)
    return y


def infer_model(config, data):
    # 1. Prior Sampling
    prior = Predictive(model, num_samples=100)(config, data)
    # 2. MCMC Sampling
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100)  # Assuming 1000 warmup steps
    mcmc.run(config, data)
    # Get posterior samples
    posterior_samples = mcmc.get_samples()
    # 3. Sample from Posterior Predictive
    posterior_predictive = Predictive(model, posterior_samples)(config, data)
    # %%
    az_data = az.from_pyro(
        posterior=mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )

    return az_data


az_data = infer_model("u_s", y_obs_list)
az.plot_trace(az_data)
plt.show()

# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_bell=True)
p.likelihood_axes.set_xlim(1.90, 2.2)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 9), joyplot=True, marginal=False, likelihood=False, n_samples=31)
p.plot_joy(('$\mu$', '$\sigma$'), '$y$', iteration=14)
plt.show()

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde',
                joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
                joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
plt.show()

# %%
az_data = infer_model("u_o", y_obs)

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde',
                joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
                joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.axjoin.set_xlim(1.96, 2.22)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-6, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
plt.show()

# %%
az_data = infer_model("u_s", y_obs_list)
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde',
                joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
                joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-5, hide_lines=True)
p.axjoin.set_xlim(1.96, 2.22)
p.likelihood_axes.set_xlim(1.70, 2.4)
plt.show()

# %%
# License
# The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:
# 
# https://opensource.org/licenses/BSD-3-Clause
# 
# The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:
# 
# https://creativecommons.org/licenses/by-nc/4.0/
# Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
