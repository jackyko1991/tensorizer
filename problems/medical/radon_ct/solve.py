"""Scan (sinogram) and reconstruct (plain and filtered back-projection), plus the error vs number of angles."""

import torch

from problems._common.progress import steps

from . import equation, tensors


def scan(f: torch.Tensor, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sinogram p(θ, s) of the image f: (A, S)."""
    return equation.radon(f, theta, x)


def per_angle_backprojections(p: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, window: str):
    """Smeared projections for each angle, without and with the ramp filter: two (A, Y, X) tensors.

    Summing them over θ (equation.backproject) gives the plain and the filtered reconstruction;
    a running sum shows the image forming while the gantry turns.
    """
    H, pad = equation.ramp_filter(p.shape[-1], float(x[1] - x[0]), window, device=p.device)
    q = equation.filter_projections(p, H, pad)  # (A, S)
    return equation.smear(p, theta, x, x), equation.smear(q, theta, x, x)


def fbp(p: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, window: str = "ram-lak") -> torch.Tensor:
    """Filtered back-projection of a sinogram: (A, S) → (Y, X)."""
    _, filtered = per_angle_backprojections(p, theta, x, window)
    return equation.backproject(filtered, theta)


def nrmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    """Normalised root-mean-square error ‖f̂ − f‖ / ‖f‖."""
    return float(torch.linalg.norm(estimate - truth) / torch.linalg.norm(truth))


def error_vs_angles(f: torch.Tensor, x: torch.Tensor, counts: list[int], window: str) -> list[float]:
    """Reconstruct with A evenly spread angles for every A in counts; return the NRMSE of each."""
    errors = []
    for i in steps(len(counts), "angle counts"):
        theta = tensors.angles(counts[i], x.device)
        errors.append(nrmse(fbp(scan(f, theta, x), theta, x, window), f))
    return errors
