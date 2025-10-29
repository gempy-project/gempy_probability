from typing import Callable, Optional

import gempy as gp
import gempy_viewer as gpv
import numpy as np
from gempy_viewer.modules.plot_2d.visualization_2d import Plot2D


def plot_gempy(
        geo_model,  # gp.data.GeoModel - avoiding import
        n_samples: int,
        samples: np.ndarray,
        update_model_fn: Callable,
        gempy_plot: Plot2D,
        plot_kwargs: Optional[dict] = None
):
    """
    General function to plot GemPy models with uncertainty from prior/posterior samples.
    
    Parameters
    ----------
    geo_model : gp.data.GeoModel
        The geological model to update and plot
    n_samples : int
        Number of samples to plot
    samples : np.ndarray
        Array of sample values to iterate through
    update_model_fn : Callable
        Function that takes (geo_model, sample_value, sample_idx) and updates the model.
        Should return None and modify geo_model in place.
    gempy_plot : Plot2D
        GemPy Plot2D object containing the figure and section data to plot on
    plot_kwargs : dict, optional
        Additional plotting kwargs for boundaries, surface points, etc.
    
    Examples
    --------
    >>> def update_model_fn(geo_model, sample_value, sample_idx):
    ...     # Transform sample value to world coordinates
    ...     xyz = np.zeros((1, 3))
    ...     xyz[0, 2] = sample_value
    ...     world_coord = geo_model.input_transform.apply_inverse(xyz)
    ...     # Modify surface point
    ...     gp.modify_surface_points(geo_model, slice=0, Z=world_coord[0, 2])
    >>> 
    >>> p2d = gpv.plot_2d(geo_model, show_lith=False, show_data=False, show=False)
    >>> samples = prior_inference_data.prior['$\\mu_{top}$'].values[0, :]
    >>> plot_gempy(geo_model, n_samples=50, samples=samples, 
    ...            update_model_fn=update_model_fn, gempy_plot=p2d)
    """
    # Import here to avoid circular dependencies and to make gempy optional
    import gempy as gp
    from gempy_viewer.API._plot_2d_sections_api import plot_sections
    from gempy_viewer.core.data_to_show import DataToShow
    
    plot_kwargs = plot_kwargs or {}

    # Iterate through samples
    for i in np.linspace(0, n_samples - 1, n_samples).astype(int):
        # Update model using the provided function
        update_model_fn(geo_model, samples[i], i)

        # Compute the model
        gp.compute_model(gempy_model=geo_model)

        # Plot the updated model
        default_plot_kwargs = {
            'kwargs_boundaries': {
                "linewidth": 0.5,
                "alpha": 0.1,
            },
            'kwargs_surface_points': {
                'alpha': 0.1
            },
        }
        # Merge with user-provided kwargs (user kwargs override defaults)
        final_plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

        plot_sections(
            gempy_model=geo_model,
            sections_data=gempy_plot.section_data_list,
            data_to_show=DataToShow(
                n_axis=1,
                show_data=True,
                show_surfaces=True,
                show_lith=False
            ),
            **final_plot_kwargs
        )

    gempy_plot.fig.show()