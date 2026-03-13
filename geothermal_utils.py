"""
Utility functions for geothermal doublet sensitivity analysis.

This module provides:
- Core models: breakthrough_time, calculate_breakthrough_curve
- Vectorized batch processing: calculate_breakthrough_curve_batch
- Unified sensitivity analysis: compute_sobol_indices

References:
- Schulz, R. (1987). Analytical model calculations for heat exchange in a confined aquifer. 
  Journal of Geophysics, 61, 12–20.
- Charbeneau, R. J. (2000). Groundwater Hydraulics and Pollutant Transport. 
  Prentice Hall, Upper Saddle River.
"""

import numpy as np
from scipy.special import erfc

# Constants
SECONDS_PER_YEAR = 365 * 24 * 3600


def breakthrough_time(samples):
    """
    Vectorized model for thermal breakthrough time in a geothermal doublet.
    
    Parameters:
    -----------
    samples : ndarray
        Parameter samples, shape (N, 8)
        Columns: [M, a, Q, phi, rho_F, c_F, rho_S, c_S]
    
    Returns:
    --------
    t_b : ndarray
        Breakthrough times in years, shape (N,)
    """
    M_s, a_s, Q_s, phi_s, rho_F_s, c_F_s, rho_S_s, c_S_s = samples.T
    rho_A_c_A_s = phi_s * rho_F_s * c_F_s + (1 - phi_s) * rho_S_s * c_S_s
    G_s = rho_A_c_A_s / (rho_F_s * c_F_s)
    return G_s * 4 * np.pi * M_s * a_s * a_s / (3 * Q_s) / SECONDS_PER_YEAR


def calculate_breakthrough_curve(time, M, a, Q, phi, rho_F, c_F, rho_S, c_S, lambda_S, 
                                  n_streamlines=360, epsilon=0.001):
    """
    Calculate the thermal breakthrough curve for a geothermal doublet system.
    
    Implements the analytical solution combining Schulz (1987) and Charbeneau (2000)
    for thermal breakthrough in a confined aquifer with a doublet well configuration.
    
    Parameters:
    -----------
    time : array-like
        Time vector (years)
    M : float
        Thickness of the aquifer (m)
    a : float
        Half the distance between injection and extraction wells (m)
    Q : float
        Injection/extraction rate (m³/s)
    phi : float
        Porosity of the aquifer
    rho_F : float
        Fluid density (kg/m³)
    c_F : float
        Fluid heat capacity (J/kg/K)
    rho_S : float
        Density of the solid matrix (kg/m³)
    c_S : float
        Heat capacity of the solid matrix (J/kg/K)
    lambda_S : float
        Thermal conductivity (W/m/K)
    n_streamlines : int, optional
        Number of streamlines to calculate (default: 360)
    epsilon : float, optional
        Edge value offset to prevent division by zero (default: 0.001)
    
    Returns:
    --------
    T : ndarray
        Temperature difference (T - T_0) / (T_i - T_0) over time
    """
    # Calculate derived parameters
    rho_A_c_A = phi * rho_F * c_F + (1 - phi) * rho_S * c_S
    G = rho_A_c_A / (rho_F * c_F)
    H = np.sqrt(lambda_S * rho_S * c_S) / (M * rho_F * c_F)
    
    # Calculate streamline arrival times using Charbeneau formula
    # Avoid edge values (0 and pi) to prevent division by zero
    phi1 = np.linspace(epsilon, np.pi - epsilon, n_streamlines)
    F = 1 - phi1 / np.pi
    
    # Calculate with valid values only
    sin_term = np.sin(np.pi * F)
    tan_term = np.tan(np.pi * F)
    
    # Filter out problematic values (where sin is close to zero)
    valid_mask = np.abs(sin_term) > 1e-10
    
    tf = np.zeros_like(phi1)
    tf[valid_mask] = (4 * np.pi * phi * M * a * a / Q * 
                      (1 - np.pi * F[valid_mask] / tan_term[valid_mask]) / 
                      (sin_term[valid_mask] ** 2) / SECONDS_PER_YEAR)
    
    tau2 = G * tf[valid_mask] / phi
    
    # Calculate temperature breakthrough curve
    T = np.zeros_like(time, dtype=float)
    for tau in tau2:
        U = np.heaviside(time - tau, 1)
        ind = U > 0
        if np.any(ind):
            abc = erfc(H * tau / G * np.sqrt(SECONDS_PER_YEAR) / np.sqrt(time[ind] - tau))
            T[ind] = T[ind] + U[ind] * abc
    
    T = T / len(tau2) if len(tau2) > 0 else T
    return T


def calculate_breakthrough_curve_batch(time, parameters, n_streamlines=360, epsilon=0.001):
    """
    Vectorized calculation of thermal breakthrough curves for multiple parameter sets.

    This optimized serial implementation computes each sample efficiently using
    vectorized streamline contributions and avoids multiprocessing overhead.

    Parameters:
    -----------
    time : array-like
        Time vector (years)
    parameters : ndarray
        Parameter matrix with shape (N_samples, 9)
        Each row: [M, a, Q, phi, rho_F, c_F, rho_S, c_S, lambda_S]
    n_streamlines : int, optional
        Number of streamlines to calculate (default: 360)
    epsilon : float, optional
        Edge value offset to prevent division by zero (default: 0.001)

    Returns:
    --------
    T : ndarray
        Temperature array with shape (N_samples, len(time))
        Each row is the breakthrough curve for one parameter set
    """
    N_samples = len(parameters)
    time = np.asarray(time, dtype=float)
    N_time = len(time)

    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 9:
        raise ValueError(
            "parameters must have shape (N_samples, 9) with columns "
            "[M, a, Q, phi, rho_F, c_F, rho_S, c_S, lambda_S]"
        )

    if N_samples == 0:
        return np.zeros((0, N_time), dtype=float)

    # Extract parameters for all samples
    M_s = parameters[:, 0]
    a_s = parameters[:, 1]
    Q_s = parameters[:, 2]
    phi_s = parameters[:, 3]
    rho_F_s = parameters[:, 4]
    c_F_s = parameters[:, 5]
    rho_S_s = parameters[:, 6]
    c_S_s = parameters[:, 7]
    lambda_S_s = parameters[:, 8]

    # Calculate derived parameters for all samples (vectorized)
    rho_A_c_A_s = phi_s * rho_F_s * c_F_s + (1 - phi_s) * rho_S_s * c_S_s
    G_s = rho_A_c_A_s / (rho_F_s * c_F_s)
    H_s = np.sqrt(lambda_S_s * rho_S_s * c_S_s) / (M_s * rho_F_s * c_F_s)

    # Calculate streamlines once (same for all samples)
    phi1 = np.linspace(epsilon, np.pi - epsilon, n_streamlines)
    F = 1 - phi1 / np.pi
    sin_term = np.sin(np.pi * F)
    tan_term = np.tan(np.pi * F)
    valid_mask = np.abs(sin_term) > 1e-10

    # Initialize output array
    T = np.zeros((N_samples, N_time), dtype=float)

    # Calculate curves for each sample
    for k in range(N_samples):
        M_k = M_s[k]
        a_k = a_s[k]
        Q_k = Q_s[k]
        phi_k = phi_s[k]
        G_k = G_s[k]
        H_k = H_s[k]

        # Calculate tau2 for this sample
        tf_k = np.zeros_like(phi1)
        tf_k[valid_mask] = (
            4
            * np.pi
            * phi_k
            * M_k
            * a_k
            * a_k
            / Q_k
            * (1 - np.pi * F[valid_mask] / tan_term[valid_mask])
            / (sin_term[valid_mask] ** 2)
            / SECONDS_PER_YEAR
        )

        tau2_k = G_k * tf_k[valid_mask] / phi_k
        if len(tau2_k) == 0:
            continue

        # Vectorized contribution of all streamlines for this sample.
        tau_matrix = tau2_k[:, np.newaxis]
        delta_t = time[np.newaxis, :] - tau_matrix
        valid_t = delta_t > 0

        safe_delta_t = np.where(valid_t, delta_t, 1.0)
        arg = H_k * tau_matrix / G_k * np.sqrt(SECONDS_PER_YEAR) / np.sqrt(safe_delta_t)
        contrib = np.where(valid_t, erfc(arg), 0.0)
        T[k, :] = np.mean(contrib, axis=0)

    return T


def compute_sobol_indices(A, B, Y_A, Y_B, Y_ABi, num_params):
    """
    Generic function to compute Sobol indices (First-Order and Total-Effect).
    
    Implements the Saltelli pick-freeze method for variance-based sensitivity analysis.
    This function is flexible and can be used for any type of output (scalar, L2-norm, 
    point values, etc.) as long as the outputs are pre-computed.
    
    Parameters:
    -----------
    A : ndarray
        Parameter samples set A, shape (N, num_params)
    B : ndarray
        Parameter samples set B, shape (N, num_params)
    Y_A : ndarray
        Output for set A, shape (N,) for scalar outputs or (N, n_outputs) for multiple
    Y_B : ndarray
        Output for set B, same shape as Y_A
    Y_ABi : dict or list of ndarray
        Outputs for swapped parameter sets AB_i
        Either dict with keys 0..num_params-1 or list of ndarrays
    num_params : int
        Number of parameters
    
    Returns:
    --------
    S_first : ndarray
        First-order Sobol indices, shape (num_params,)
        Measures direct effect of each parameter
    S_total : ndarray
        Total-effect Sobol indices, shape (num_params,)
        Measures direct effect plus all interactions
    
    Notes:
    ------
    - First-order index S_i captures only the direct effect of parameter i
    - Total-effect index S_T captures direct effect plus all interactions
    - If S_T - S_i ≈ 0, parameter i has little interaction with others
    - If S_T >> S_i, parameter i is involved in strong interactions
    """
    S_first = np.zeros(num_params)
    S_total = np.zeros(num_params)
    
    # Ensure outputs are 1D for scalar analysis
    y_A = np.atleast_1d(Y_A).flatten() if np.ndim(Y_A) == 1 else Y_A
    y_B = np.atleast_1d(Y_B).flatten() if np.ndim(Y_B) == 1 else Y_B
    
    # Variance for normalization
    var_y = np.var(np.concatenate([y_A.flatten(), y_B.flatten()]), ddof=1)
    
    if var_y < 1e-10:
        return S_first, S_total  # Return zeros if no variance
    
    # Compute Sobol indices for each parameter
    for i in range(num_params):
        # Get output for AB_i (swap parameter i)
        y_ABi = np.atleast_1d(Y_ABi[i]).flatten() if np.ndim(Y_ABi[i]) == 1 else Y_ABi[i]
        
        # First-order Sobol index: S_i = E[y_B * (y_ABi - y_A)] / Var(y)
        S_first[i] = np.mean(y_B.flatten() * (y_ABi.flatten() - y_A.flatten())) / var_y
        
        # Total-effect Sobol index: S_T = 0.5 * E[(y_A - y_ABi)²] / Var(y)
        S_total[i] = 0.5 * np.mean((y_A.flatten() - y_ABi.flatten()) ** 2) / var_y
    
    # Numerical safety: clip to valid range
    S_first = np.clip(S_first, 0, 1)
    S_total = np.clip(S_total, 0, 1)
    
    return S_first, S_total
