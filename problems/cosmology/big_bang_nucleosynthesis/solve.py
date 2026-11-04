"""Integrate the stiff network with an implicit BDF2 scheme in u = −ln T, all η values at once.

Each step solves Y_{n+1} − c_n = β h (dt/du) f(Y_{n+1}) by Newton's method; the Newton matrices of the whole η batch
are solved together with one torch.linalg.solve on a (B, N, N) tensor.
"""

import torch

from problems._common.progress import steps

from . import equation, tensors


def evolve(Y0: torch.Tensor, lam_f: torch.Tensor, lam_b: torch.Tensor, rate_u: torch.Tensor, u: torch.Tensor,
           newton: int = 4) -> torch.Tensor:  # fmt: skip
    """Y history (S + 1, B, N) from Y0 (B, N). lam_f/lam_b: (S + 1, B, Rx), rate_u = dt/du: (S + 1,), u: (S + 1,)."""
    R, P = tensors.stoichiometry(Y0.device)
    eye = torch.eye(Y0.shape[1], dtype=Y0.dtype, device=Y0.device)
    Ys = [Y0]
    for n in steps(len(u) - 1, "BDF2 steps"):
        h = float(u[n + 1] - u[n])
        if n == 0:  # first step: backward Euler
            c, beta = Ys[-1], 1.0
        else:  # BDF2: Y_{n+1} − (4/3) Y_n + (1/3) Y_{n−1} = (2/3) h f(Y_{n+1})
            c, beta = (4 * Ys[-1] - Ys[-2]) / 3, 2 / 3
        k = beta * h * float(rate_u[n + 1])
        Y = Ys[-1].clone()
        for _ in range(newton):
            f, J = equation.network(Y, lam_f[n + 1], lam_b[n + 1], R, P)
            G = Y - c - k * f  # residual (B, N)
            Y = (Y - torch.linalg.solve(eye - k * J, G[..., None])[..., 0]).clamp(min=0)
        Ys.append(Y)
    return torch.stack(Ys)


def run_bbn(etas: torch.Tensor, tau_n: float, n_nu: float, n_steps: int, device: torch.device,
            T_start: float = 3.0, T_end: float = 1e-3) -> dict:  # fmt: skip
    """The whole calculation for a batch of η: thermal history, rates, network, final abundances."""
    etas = etas.to(device=device, dtype=tensors.F64)
    T = tensors.temperature_grid(T_start, T_end, n_steps, device)  # (S + 1,) MeV, hot → cold
    lam_f, lam_b = equation.rate_coefficients(T, etas, tau_n)
    Y = evolve(tensors.initial_abundances(T_start, len(etas), device), lam_f, lam_b, equation.dt_du(T, n_nu), -T.log())
    final = equation.observables(Y[-1])
    return {"T": T, "t": equation.time_of(T, n_nu), "Y": Y, "etas": etas, **final}
