"""
Normal Prior, single observation
================================

"""
# sphinx_gallery_thumbnail_number = -1

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch

from gempy_probability.modules.plot.plot_posterior import PlotPosterior
from _aux_func import infer_model

az.style.use("arviz-doc")

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)

# Before diving in sampling, let's look at a model, where we have a single observation to sample the posterior from a prior with a normal
# distribution for :math:`\mu` and a gamma distribution for :math:`\sigma`:

# %%
az_data = infer_model(
    distributions_family="normal_distribution",
    data=y_obs
)
az.plot_trace(az_data)
plt.show()

# %%
# Prior Space
# """"""""""

# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(
    var_names=['$\\mu_{likelihood}$', '$\\sigma_{likelihood}$'],
    plot_trace=False,
    credible_interval=.93,
    kind='kde',
    group="prior",
    joint_kwargs={
            'contour'          : True,
            'pcolormesh_kwargs': {}
    },
    joint_kwargs_prior={
            'contour': False,
            'pcolormesh_kwargs': {}
    }
)

p.axjoin.set_xlim(1.96, 2.22)
p.plot_normal_likelihood(
    mean='$\\mu_{likelihood}$',
    std='$\\sigma_{likelihood}$',
    obs='$y$',
    iteration=-6,
    hide_lines=True
)
p.likelihood_axes.set_xlim(1.70, 2.40)
plt.show()

# %%
# Prior and Posterior Space
# """""""""""""""""""""""""

# %%
def plot_posterior(iteration=-1):
    p = PlotPosterior(az_data)

    p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
    p.plot_marginal(
        var_names=['$\\mu_{likelihood}$', '$\\sigma_{likelihood}$'],
        plot_trace=True,
        credible_interval=.93,
        iteration=iteration,
        kind='kde',
        group="both",
        joint_kwargs={
                'contour'          : True,
                'pcolormesh_kwargs': {}
        },
        joint_kwargs_prior={
                'contour'          : False,
                'pcolormesh_kwargs': {}
        }
    )

    p.axjoin.set_xlim(1.96, 2.22)
    p.plot_normal_likelihood(
        mean='$\\mu_{likelihood}$',
        std='$\\sigma_{likelihood}$',
        obs='$y$',
        iteration=iteration,
        hide_lines=False
    )
    p.likelihood_axes.set_xlim(1.96, 2.22)
    p.fig.set_label(f'Iteration {iteration}')
    plt.show()


for i in np.linspace(50, 800, 5, dtype=int):
    plot_posterior(i)

# %%
# MCMC boils down to be a collection of methods helping to do Bayesian inference, thus based on Bayes Theorem:
# 
# .. math::
#    P(\theta | x) = \frac{P(x|\theta) P(\theta)}{P(x)}
# 
# * :math:`P(\theta | x)` is the Posterior
# * :math:`P(x)` is the Prior
# * :math:`P(x | \theta)` is the Likelihood
# * :math:`P(x)` is the Evidence
# 
# As calculating the posterior in this form is most likely not possible in real-world problems, if one could sample from the posterior,
# one might approximate it with Monte Carlo. But in order to sample directly from the posterior, one would need to invert Bayes Theorem.
# 
# The solution to this problem is, when we cannot draw Monte Carlo (MC) samples from the distribution directly, we let a Markov Chain
# do it for us. [1]

# %%
# What con we do next? 
# """"""""""""""""""""
# Increasing the number of observations - sampling (:doc:`1.2_Intro_to_Bayesian_Inference`)

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
