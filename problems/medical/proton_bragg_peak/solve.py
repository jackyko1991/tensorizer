"""Slow the protons down (RK4 in depth), turn energy loss into dose, fit the range law and build an SOBP."""

import torch

from problems._common.integrators import integrate
from problems._common.progress import steps

from . import equation

E_CUT = 0.1  # MeV: below this the proton is considered stopped (residual range ≈ 1 µm)
C_CM_PER_NS = 29.9792458


def slow_down(E0: torch.Tensor, I_mev: float, z_max: float, dz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate dE/dz = −S(E) for the whole batch. Returns (z (Z+1,), E (Z+1, B))."""
    n = int(round(z_max / dz))
    E = integrate(lambda e, z: equation.rhs(e, I_mev, E_CUT), E0, dz, n)  # (Z+1, B)
    z = torch.arange(n + 1, device=E0.device) * dz
    return z, E.clamp_min(0.0)


def csda_range(z: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Depth where each proton stops (E ≤ E_CUT), linearly interpolated between steps: (B,)."""
    stopped = E <= E_CUT  # (Z+1, B)
    k = torch.where(stopped.any(0), stopped.int().argmax(0), len(z) - 1).clamp_min(1)  # first stopped step
    b = torch.arange(E.shape[1], device=E.device)
    e_before, e_after = E[k - 1, b], E[k, b]
    frac = ((e_before - E_CUT) / (e_before - e_after).clamp_min(1e-9)).clamp(0, 1)
    return z[k - 1] + frac * (z[1] - z[0])


def depth_dose(
    z: torch.Tensor, E: torch.Tensor, ranges: torch.Tensor, energy_spread: float = 0.0, p: float = 1.75
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dose per proton (MeV/cm) at the bin centres for each beam: (z_mid (Z,), D (B, Z)).

    The energy lost in each depth step, −ΔE/Δz, is the pristine Bragg curve; we then weight it by the
    surviving primary fluence and blur it with range straggling plus the beam's energy spread
    (a relative spread σ_E/E widens the range by σ_R ≈ p (σ_E/E) R, since R ∝ E^p).
    """
    dz = z[1] - z[0]
    z_mid = 0.5 * (z[1:] + z[:-1])
    pristine = (-torch.diff(E, dim=0) / dz).T  # (B, Z)
    pristine = pristine * equation.primary_fluence(z_mid)[None, :]
    sigma = torch.sqrt(equation.straggling_width(ranges) ** 2 + (p * energy_spread * ranges) ** 2)
    return z_mid, equation.straggle(pristine, z_mid, sigma)


def fit_bragg_kleeman(E0: torch.Tensor, R: torch.Tensor) -> tuple[float, float]:
    """Least squares on log R = log α + p log E. Returns (α, p)."""
    A = torch.stack([torch.ones_like(E0), torch.log(E0)], dim=1)  # (B, 2)
    coef = torch.linalg.lstsq(A, torch.log(R)[:, None]).solution[:, 0]
    return float(torch.exp(coef[0])), float(coef[1])


def sobp_weights(layers: torch.Tensor, z: torch.Tensor, z1: float, z2: float, iters: int = 3000) -> torch.Tensor:
    """Non-negative weights w_k making D_{kz} w_k ≈ 1 on the target [z1, z2] (projected gradient NNLS)."""
    target = (z >= z1) & (z <= z2)
    A = layers[:, target].T  # (Z_T, K)
    ones = torch.ones(A.shape[0], device=A.device)
    w = torch.linalg.lstsq(A, ones[:, None]).solution[:, 0].clamp_min(0.0)  # unconstrained start, clipped
    step = 1.0 / torch.linalg.matrix_norm(A, ord=2) ** 2  # 1 / Lipschitz constant of the gradient
    for _ in steps(iters, "NNLS iterations"):
        w = (w - step * A.T @ (A @ w - ones)).clamp_min(0.0)
    return w


def flight_times(z: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Time (ns) for each proton to reach depth z: t = ∫ dz / v.  (Z+1, B); stays constant once stopped."""
    beta = torch.sqrt(equation.kinematics(E.clamp_min(E_CUT))[0])
    moving = (E > E_CUT).float()
    dt = (z[1] - z[0]) / (beta * C_CM_PER_NS) * moving
    return torch.cat([torch.zeros_like(dt[:1]), torch.cumsum(dt[:-1], dim=0)])


def flatness(dose: torch.Tensor, z: torch.Tensor, z1: float, z2: float) -> float:
    """(max − min) / mean of the dose over the target."""
    d = dose[(z >= z1) & (z <= z2)]
    return float((d.max() - d.min()) / d.mean())
