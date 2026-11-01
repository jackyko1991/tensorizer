"""Backward-Euler integration of the stiff Chapman network at every altitude at once, with Newton iterations."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import dcdt, jacobian, k2_rate, k4_rate


def backward_euler_step(c, dt: float, coeffs, n_newton: int = 6):
    """Solve G(c⁺) = c⁺ - c - Δt S r(c⁺) = 0 by Newton: (1 - Δt J) δ = -G, c⁺ ← c⁺ + δ, all altitudes batched."""
    eye = torch.eye(3, dtype=c.dtype, device=c.device)
    new = c.clone()
    for _ in range(n_newton):
        G = new - c - dt * dcdt(new, *coeffs)  # (A, 3)
        delta = torch.linalg.solve(eye - dt * jacobian(new, *coeffs), -G)  # (A, 3)
        new = (new + delta).clamp(min=0.0)
    return new


def solve(z: torch.Tensor, times: torch.Tensor, mu: float):
    """Integrate from sunrise (all O₂) through the log-spaced times (T,).

    j₁, k₂, k₄ and [M] are fixed per altitude; j₃ depends on the ozone column overhead and is updated from the
    previous step (lagged), which couples the levels only through the shielding.
    Returns c (T, A, 3) and the final coefficients (j1, j3, k2, k4, M), each (A,)."""
    M = tensors.air_density(z)
    T = tensors.temperature(z)
    j1, k2, k4 = tensors.photolysis_O2(z, mu), k2_rate(T), k4_rate(T)
    c = tensors.initial_state(z)
    out = [c]
    for k in steps(len(times) - 1, "backward-Euler steps"):
        j3 = tensors.photolysis_O3(c[:, 2], z, mu)
        c = backward_euler_step(c, (times[k + 1] - times[k]).item(), (j1, j3, k2, k4, M))
        out.append(c)
    j3 = tensors.photolysis_O3(c[:, 2], z, mu)
    return torch.stack(out), (j1, j3, k2, k4, M)


def lifetimes(c: torch.Tensor, j1, j3, k2, k4, M):
    """Time scales at the end state, each (A,), in s:
    fast  1 / (k₂[M][O₂] + j₃): how quickly O and O₃ trade places;
    slow  [Oₓ] / (2 k₄[O][O₃]): how long odd oxygen lives before O + O₃ destroys it."""
    o, o2, o3 = c[:, 0], c[:, 1], c[:, 2]  # [O], [O₂], [O₃]
    fast = 1 / (k2 * M * o2 + j3)
    slow = (o + o3) / (2 * k4 * o * o3)
    return fast, slow
