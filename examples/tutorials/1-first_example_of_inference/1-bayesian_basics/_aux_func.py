import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az
import torch


# def model(distributions_family, data):
#     if distributions_family == "normal_distribution":
#         mu = pyro.sample('$\\mu$', dist.Normal(2.07, 0.07))
#     elif distributions_family in "uniform_distribution":
#         mu = pyro.sample('$\\mu$', dist.Uniform(0, 10))
#     else:
#         raise ValueError("distributions_family must be either 'normal_distribution' or 'uniform_distribution'")
#     sigma = pyro.sample('$\\sigma$', dist.Gamma(0.3, 3))
#     y = pyro.sample('$y$', dist.Normal(mu, sigma), obs=data)
#     return y


def model(distributions_family, data):
    if distributions_family == "normal_distribution":
        param_mean = pyro.param('$\\mu_{prior}$', torch.tensor(2.07))
        param_std = pyro.param('$\\sigma_{prior}$', torch.tensor(0.07), constraint=dist.constraints.positive)
        mu = pyro.sample('$\\mu_{likelihood}$', dist.Normal(param_mean, param_std))
    elif distributions_family in "uniform_distribution":
        param_low = pyro.param('$\\mu_{prior}$', torch.tensor(0.))
        param_high = pyro.param('$\\sigma_{prior}$', torch.tensor(10.))
        # mu = pyro.sample('$\\mu_{likelihood}$', dist.Uniform(param_low, param_high))

        mu = pyro.sample('$\mu_{likelihood}$', dist.Uniform(0, 10))
    else:
        raise ValueError("distributions_family must be either 'normal_distribution' or 'uniform_distribution'")
    param_concentration = pyro.param('$\\alpha_{prior}$', torch.tensor(0.3))
    param_rate = pyro.param('$\beta_{prior}$', torch.tensor(3.))

    sigma = pyro.sample('$\\sigma_{likelihood}$', dist.Gamma(param_concentration, param_rate))
    y = pyro.sample('$y$', dist.Normal(mu, sigma), obs=data)
    return y


def infer_model(distributions_family, data):
    # 1. Prior Sampling
    prior = Predictive(model, num_samples=100)(distributions_family, data)
    # 2. MCMC Sampling
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100)  # Assuming 1000 warmup steps
    mcmc.run(distributions_family, data)
    # Get posterior samples
    posterior_samples = mcmc.get_samples(1100)
    # 3. Sample from Posterior Predictive
    posterior_predictive = Predictive(model, posterior_samples)(distributions_family, data)
    # %%
    az_data = az.from_pyro(
        posterior=mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )

    return az_data
