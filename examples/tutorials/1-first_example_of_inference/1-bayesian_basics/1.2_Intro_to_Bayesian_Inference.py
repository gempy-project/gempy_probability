"""
Normal Prior, several observations
==================================

# We previously mentioned that we can optimize this model to better fit observations. For this, we generate a prior from the model we previously created, the one with multiple observations from `y_obs_list`.
# In the following, samples are iteratively drawn and evaluated to form the posterior. This generates what is called a "trace". A predictive Posterior distribution is then generated from said trace.

"""
# sphinx_gallery_thumbnail_number = -1

import arviz as az
import matplotlib.pyplot as plt
import pyro
import torch
from matplotlib.ticker import StrMethodFormatter

from gempy_probability.plot_posterior import PlotPosterior
from _aux_func import infer_model

y_obs = torch.tensor([2.12])
y_obs_list = torch.tensor([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
                           2.19, 2.07, 2.16, 2.11, 2.13, 1.92])
pyro.set_rng_seed(4003)

# %%
az_data = infer_model(
    distributions_family="normal_distribution",
    data=y_obs_list
)
az.plot_trace(az_data)
plt.show()

#%% md
#
# #### Raw observations:
# The behaviour of this chain is controlled by the observations we fed into the model. Let's have a look again at the observations and how they are spread/distributed:
# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_bell=True)
p.likelihood_axes.set_xlim(1.90, 2.2)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

#%% md
# The bulk of observations is between 2.05 and 2.15, one observation at the lower end with 1.92.
# #### Final inference
#
# Now let's plot the inferred posterior distribution (i.e. the last sample iteration) and the observations:
# %%
p = PlotPosterior(az_data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

#%% md
# The bell-peak is above a cluster of observations, but the one observation at 1.92 seems to be of greater importance, as otherwise, the highest likelihood might be located more around 2.1.
# ### Joyplot
#
# Just plotting the posterior does not convey the underlying process. Attached to GemPy, and used for its stochastic capabilities, visualization methods for drawing distributions were written. More specifically, we use joyplots for showing the change of the distribution (more precisely its mean and its standard deviation) with the progressing chaing:
# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 9), joyplot=True, marginal=False, likelihood=False, n_samples=31)
p.plot_joy(('$\mu$', '$\sigma$'), '$y$', iteration=14)
plt.show()
#%% md
# Below we show a gif of how the curve actually moves ($\mu$ is changed) and changes its width (change in $\sigma$ parameters) with progressive sampling. Dark colors represent an increase in likeliness. However, we see that the colors change with new iterations. That means, that what previously seemed likely becomes less likely as we keep exploring the probability space.
# <hr>
#
# ![joyplot animated](images/joyplot_2.gif "segment")
#
# <hr>
# ### Join probabiity
# %%
p = PlotPosterior(az_data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(
    var_names=['$\mu$', '$\sigma$'],
    plot_trace=False,
    credible_interval=0.95,
    kind='kde',
    joint_kwargs={'contour': True, 'pcolormesh_kwargs': {}},
    joint_kwargs_prior={'contour': False, 'pcolormesh_kwargs': {}})

p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)

plt.show()
#%% md
# The small gif below shows the first 100 samplings (starting from the 10th iteration) of the chain:
#%% md
# <hr>
#
# ![sampling](images/sampling_2.gif "segment")
#
# <hr>
#%% md
# A sped-up version of the full sampling (each 10 steps until we reach the 1000th iteration) provides an impression how we arrive at the posterior distribution...and that the change after the first couple of iterations gets smaller and smaller, as the chain itself gets more and more "educated" about its guesses:
#%% md
# <hr>
#
# ![sampling](images/sampling.gif "segment")
#
# <hr>

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
