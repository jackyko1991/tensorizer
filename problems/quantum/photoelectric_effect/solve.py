"""Monte Carlo of photons ejecting electrons, retarding-voltage curves, and the straight-line fit eV_s = hν − φ."""

import torch

from . import equation, tensors


def measured_stopping_potential(K: torch.Tensor) -> torch.Tensor:
    """The retarding voltage that stops even the fastest emitted electron: max K over the last axis (≥ 0)."""
    return torch.clamp(K.max(dim=-1).values, min=0.0)


def simulate(nu, phi, intensities, n_photons, width=tensors.BAND, seed=0, voltages=None) -> dict:
    """Every (intensity, frequency, photon) at once.

    Returns kinetic energies K (I, F, N) (−inf for photons not absorbed in this exposure or not emitted), the
    stopping potentials (I, F) and the current (I, F, V) reaching the collector, in units of n_photons.
    """
    n_int = len(intensities)
    counts = tensors.photon_counts(intensities, n_photons, seed)  # (I,)
    n_max = int(counts.max())
    depth = tensors.electron_depths(len(nu), n_max, width, seed, n_int=n_int)  # (I, F, N)
    K = equation.kinetic_energy(nu, phi, depth)  # (I, F, N): hν_f − φ − ε
    present = torch.arange(n_max)[None, None, :] < counts[:, None, None]  # (I, 1, N) photons of this exposure
    K = torch.where(present & (K > 0), K, torch.tensor(-torch.inf, dtype=K.dtype))
    if voltages is None:
        voltages = tensors.retarding_voltages(-1.0, float(equation.photon_energy(nu).max()) - phi + 0.5)
    K_sorted = K.sort(dim=-1).values
    V = voltages.expand(n_int, len(nu), -1).contiguous()  # (I, F, V)
    # electrons with K > eV pass: N − (number with K ≤ eV)
    passed = n_max - torch.searchsorted(K_sorted, V, right=True)
    return {
        "K": K,
        "V_stop": measured_stopping_potential(K),
        "current": passed.double() / n_photons,
        "voltages": voltages,
        "counts": counts,
    }


def fit_line(nu: torch.Tensor, V: torch.Tensor) -> dict:
    """Least squares V = a ν + b (only points above threshold, V > 0), with standard errors from the residuals."""
    keep = V > 0
    x, y = nu[keep], V[keep]
    A = torch.stack([x, torch.ones_like(x)], dim=-1)  # (M, 2) design matrix
    coef = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]  # (2,)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov = torch.linalg.inv(A.T @ A) * (resid @ resid) / dof  # (2, 2)
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "slope_err": float(cov[0, 0].sqrt()),
        "intercept_err": float(cov[1, 1].sqrt()),
    }
