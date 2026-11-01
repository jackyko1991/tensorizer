"""Solve the chain exactly: diagonalise K, then add up the normal-mode cosines."""

import torch

from .equation import mode_amplitudes, normal_modes, superpose


def solve(K: torch.Tensor, m: float, x0: torch.Tensor, t_max: float, n_times: int = 1001) -> dict[str, torch.Tensor]:
    """Return times (T,), positions x and velocities v (T, N), and the modes ω (N,), e (N, N), A (N,).

    No time stepping is needed: once the modes are known, x_i(t) is a closed formula
    that we evaluate at all T times in one einsum.
    """
    omega, e = normal_modes(K, m)
    A, phi = mode_amplitudes(e, omega, x0, torch.zeros_like(x0))  # start at rest
    times = torch.linspace(0.0, t_max, n_times, device=K.device)
    x, v = superpose(e, omega, A, phi, times)
    return {"times": times, "x": x, "v": v, "omega": omega, "e": e, "A": A}
