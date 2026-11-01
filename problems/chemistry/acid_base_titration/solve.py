"""Solve the charge balance for the pH by bisection, then find the equivalence points from dpH/dV."""

import torch
import torch.nn.functional as F

from .equation import charge_balance


def solve_ph(
    log_beta: torch.Tensor,
    c_acid: torch.Tensor,
    sodium: torch.Tensor,
    kw: float,
    lo: float = -2.0,
    hi: float = 16.0,
    iterations: int = 60,
) -> torch.Tensor:
    """pH_{an} for every acid a and volume n, shape (A, N), by bisection on pH = -log₁₀ h.

    f(pH) decreases with pH, so f(mid) > 0 means the root lies above mid. Each step halves the bracket for ALL
    (a, n) pairs at once; 60 steps shrink 18 pH units to ~10⁻¹⁷.
    """
    shape = (log_beta.shape[0], c_acid.shape[0])
    low = torch.full(shape, lo, dtype=c_acid.dtype, device=c_acid.device)
    high = torch.full(shape, hi, dtype=c_acid.dtype, device=c_acid.device)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        above = charge_balance(mid, log_beta, c_acid, sodium, kw) > 0
        low = torch.where(above, mid, low)
        high = torch.where(above, high, mid)
    return 0.5 * (low + high)


def slope(pH: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """dpH/dV by central differences along the volume axis.   (A, N) → (A, N) in pH units per mL."""
    return torch.gradient(pH, spacing=(V,), dim=-1)[0]


def equivalence_points(dpdv: torch.Tensor, V: torch.Tensor, window_ml: float = 2.0, threshold: float = 0.3):
    """Volumes where dpH/dV has a peak: a local maximum over ±window_ml that is steeper than threshold pH/mL.

    A max-pool finds every local maximum in one call; the first and last window are skipped (edges, not peaks).
    Returns one list of volumes per acid.
    """
    k = max(1, int(round(window_ml / (V[1] - V[0]).item())))
    pooled = F.max_pool1d(dpdv[:, None], 2 * k + 1, stride=1, padding=k)[:, 0]  # (A, N) local maxima
    peak = (dpdv == pooled) & (dpdv > threshold)
    peak[:, :k] = False
    peak[:, -k:] = False
    return [V[row].tolist() for row in peak]
