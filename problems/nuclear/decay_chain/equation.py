r"""Physics → tensor translation.

A decay chain 1 → 2 → … → M (the last member stable) obeys the Bateman equations

    dN_1/dt = -λ_1 N_1,      dN_i/dt = λ_{i-1} N_{i-1} - λ_i N_i,

which is one linear system with a constant matrix A:

    \dot N_i = A_{ij} N_j        (j summed),
    A_{ii} = -λ_i,   A_{i+1,i} = +λ_i,   every other entry 0.

Its exact solution is the matrix exponential N_i(t) = [e^{A t}]_{ij} N_j(0). The activity of member i
(decays per unit time) is a_i = λ_i N_i.
"""

import torch


def bateman_matrix(lam: torch.Tensor) -> torch.Tensor:
    r"""A_{ij} from the decay constants λ_i (last one 0 for a stable end).   (M,) → (M, M)"""
    loss = torch.diag(-lam)  # -λ_i on the diagonal: member i decays away
    gain = torch.diag(lam[:-1], diagonal=-1)  # +λ_i one row below: …into member i + 1
    return loss + gain


def rhs(n: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""\dot N_i = A_{ij} N_j.   n: (M,) → (M,)"""
    return torch.einsum("ij,j->i", a, n)


def propagator(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""[e^{A t}]_{ij} for every time at once.   a: (M, M), t: (T,) → (T, M, M)"""
    return torch.linalg.matrix_exp(t[:, None, None] * a)


def activity(n: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r"""a_i = λ_i N_i (no sum).   n: (..., M) → (..., M)"""
    return n * lam
