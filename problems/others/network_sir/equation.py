r"""Epidemics on a network → tensor translation.

Indices: b = realisation (B run at once, each with its own β_b), i, j = people (nodes).

Stochastic SIR in small time steps Δt (one coin per person per step):

    \text{pressure}_{bi} = A_{ij} I_{bj}                                  (j summed: infected contacts of i)
    P(S \to I)_{bi} = 1 - e^{-\beta_b \Delta t\, \text{pressure}_{bi}},    P(I \to R) = 1 - e^{-\gamma \Delta t}

For SIS the recovered go straight back to S. Individual-based mean field (each person infected independently with
probability x_i, susceptible with probability s_i):

    \dot s_{bi} = -\beta_b\, s_{bi} A_{ij} x_{bj},   \dot x_{bi} = \beta_b\, s_{bi} A_{ij} x_{bj} - \gamma x_{bi},
    \dot r_{bi} = \gamma x_{bi}

Near the disease-free state \dot x = (\beta A - \gamma)\,x, which grows iff β/γ > 1/λ_max(A): the epidemic threshold.
Well mixed (A = all ones), with R_0 = β(N−1)/γ, the fraction ever infected z solves z = 1 − e^{−R_0 z}.
"""

import torch


def pressure(A: torch.Tensor, infected: torch.Tensor) -> torch.Tensor:
    """pressure_{bi} = A_{ij} I_{bj}: the number of infected contacts of each person.  j summed."""
    return torch.einsum("ij,bj->bi", A, infected.to(A.dtype))


def infection_prob(beta: torch.Tensor, dt: float, press: torch.Tensor) -> torch.Tensor:
    """P(S → I in Δt) = 1 − exp(−β_b Δt pressure_{bi}): independent chances from each infected contact."""
    return 1.0 - torch.exp(-beta[:, None] * dt * press)


def mean_field_rhs(y: torch.Tensor, A: torch.Tensor, beta: torch.Tensor, gamma: float) -> torch.Tensor:
    """d/dt of y = (s, x, r) stacked as (3, B, N).  A x is the einsum 'ij,bj->bi' (j summed)."""
    s, x, _ = y
    force = beta[:, None] * s * torch.einsum("ij,bj->bi", A.to(y.dtype), x)
    return torch.stack([-force, force - gamma * x, gamma * x])


def lambda_max(A: torch.Tensor) -> torch.Tensor:
    """Largest eigenvalue of the symmetric adjacency matrix (works on stacks (..., N, N))."""
    return torch.linalg.eigvalsh(A.double())[..., -1]


def lambda_nonbacktracking(A: torch.Tensor) -> torch.Tensor:
    """Leading eigenvalue of the non-backtracking (Hashimoto) matrix, via the Ihara–Bass 2N × 2N matrix
    [[A, I − D], [I, 0]], D = diag(degrees). Message passing puts the SIR threshold at transmissibility 1/λ_NB."""
    n = A.shape[0]
    A = A.double()
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    M = torch.cat([torch.cat([A, eye - torch.diag(A.sum(1))], 1), torch.cat([eye, torch.zeros_like(A)], 1)], 0)
    return torch.linalg.eigvals(M).real.max()


def final_size(r0: torch.Tensor, n_iter: int = 2000) -> torch.Tensor:
    """Positive root of z = 1 − exp(−R0 z) for every R0 (batched); 0 when R0 ≤ 1. Fixed-point iteration from z = 1."""
    z = torch.ones_like(r0)
    for _ in range(n_iter):
        z = 1.0 - torch.exp(-r0 * z)
    return torch.where(r0 > 1, z, torch.zeros_like(z))
