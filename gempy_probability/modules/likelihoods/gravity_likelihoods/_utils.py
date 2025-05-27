import numpy as np
from typing import Tuple

def calculate_scale_shift(
    measured: np.ndarray,
    simulated: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the best-fit scale (slope) and shift (intercept) so that:
        simulated ≈ scale_factor * measured + shift_term

    Parameters
    ----------
    measured : array-like, shape (n,)
        The observed (independent) values.
    simulated : array-like, shape (n,)
        The simulated or target (dependent) values.

    Returns
    -------
    scale_factor : float
        Multiplicative factor.
    shift_term : float
        Additive offset.

    Raises
    ------
    ValueError
        If lengths differ or fewer than 2 valid points remain.
    """
    # Flatten inputs
    measured = np.asarray(measured).ravel()
    simulated = np.asarray(simulated).ravel()

    # Check same length
    if measured.size != simulated.size:
        raise ValueError(
            f"Length mismatch: measured has {measured.size}, "
            f"simulated has {simulated.size}"
        )

    # Mask out NaN or infinite entries
    valid_mask = np.isfinite(measured) & np.isfinite(simulated)
    measured_valid = measured[valid_mask]
    simulated_valid = simulated[valid_mask]

    if measured_valid.size < 2:
        raise ValueError(
            "Need at least 2 valid (non-NaN/Inf) data points"
        )

    # Build design matrix [measured_valid, 1]
    design_matrix = np.vstack([
        measured_valid,
        np.ones_like(measured_valid)
    ]).T

    # Solve least squares: design_matrix @ [scale_factor, shift_term] ≈ simulated_valid
    (scale_factor, shift_term), *_ = np.linalg.lstsq(design_matrix,
                                                    simulated_valid,
                                                    rcond=None)

    return float(scale_factor), float(shift_term)

def gaussian_kernel(locations, length_scale, variance):
    import torch
    # Compute the squared Euclidean distance between each pair of points
    locations = torch.tensor(locations.values)
    distance_squared = torch.cdist(locations, locations, p=2).pow(2)
    # Compute the covariance matrix using the Gaussian kernel
    covariance_matrix = variance * torch.exp(-0.5 * distance_squared / length_scale ** 2)
    return covariance_matrix
