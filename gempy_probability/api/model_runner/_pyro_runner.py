import arviz as az
import matplotlib.pyplot as plt
import torch
from pyro.infer import MCMC
from pyro.infer import NUTS
from pyro.infer import Predictive
from pyro.infer.autoguide import init_to_mean

import gempy as gp
import gempy_probability as gpp
from ...core.samplers_data import NUTSConfig


def run_predictive(prob_model: gpp.GemPyPyroModel, geo_model: gp.data.GeoModel,
                   y_obs_list: torch.Tensor, n_samples: int, plot_trace: bool = False) -> az.InferenceData:
    predictive = Predictive(
        model=prob_model,
        num_samples=n_samples
    )
    prior: dict[str, torch.Tensor] = predictive(geo_model, y_obs_list)
    # print("Number of interpolations: ", geo_model.counter)

    data: az.InferenceData = az.from_pyro(prior=prior)
    if plot_trace:
        az.plot_trace(data.prior)
        plt.show()

    return data


def run_nuts_inference(prob_model: gpp.GemPyPyroModel, geo_model: gp.data.GeoModel,
                       y_obs_list: torch.Tensor, config: NUTSConfig, plot_trace: bool = False,
                       run_posterior_predictive: bool = False) -> az.InferenceData:
    nuts_kernel = NUTS(
        prob_model,
        step_size=config.step_size,
        adapt_step_size=config.adapt_step_size,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        init_strategy=init_to_mean
    )
    data = run_mcmc_for_NUTS(
        geo_model=geo_model,
        nuts_kernel=nuts_kernel,
        prob_model=prob_model,
        y_obs_list=y_obs_list,
        num_samples=config.num_samples,
        warmup_steps=config.warmup_steps,
        plot_trace=plot_trace,
        run_posterior_predictive=run_posterior_predictive,
    )

    return data


def run_mcmc_for_NUTS(
        geo_model: gp.data.GeoModel,
        nuts_kernel: NUTS,
        prob_model: gpp.GemPyPyroModel,
        y_obs_list: torch.Tensor,
        num_samples: int,
        warmup_steps: int,
        plot_trace: bool = False,
        run_posterior_predictive: bool = False,
) -> az.InferenceData:
    mcmc = MCMC(
        kernel=nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        disable_validation=False,
    )
    mcmc.run(geo_model, y_obs_list)

    if run_posterior_predictive:
        posterior_predictive = Predictive(
            model=prob_model,
            posterior_samples=mcmc.get_samples(),
        )(geo_model, y_obs_list)
        data: az.InferenceData = az.from_pyro(
            posterior=mcmc,
            posterior_predictive=posterior_predictive,
        )
    else:
        data: az.InferenceData = az.from_pyro(posterior=mcmc)

    if plot_trace:
        az.plot_trace(data)
        plt.show()

    return data
