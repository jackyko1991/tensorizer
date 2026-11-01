r"""Physics → tensor translation.

A particle hops between neighbouring sites i of a 1D lattice with energies U_i, kicked by a bath at
temperature T_b. The probability vector p_i(t) obeys the master equation

    \dot p_i = W_{ij} p_j                                    (j summed)

with Metropolis rates for a jump j → i between neighbours (A_ij = 1):

    W_{ij} = A_{ij} \min(1, e^{-(U_i - U_j)/T_b})   (i ≠ j),        W_{jj} = -\sum_{k} W_{kj}

The diagonal makes every column sum to zero, so total probability is conserved. The rates obey detailed
balance, W_ij π_j = W_ji π_i, so the stationary state is the Boltzmann distribution π_i ∝ e^{-U_i/T_b}.

Expanding in the right/left eigenvectors of W (W r_k = λ_k r_k, ℓ_k^T W = λ_k ℓ_k^T):

    p_{bi}(t) = π_i + \sum_{k ≥ 2} a_{bk} e^{λ_k t} r_{ik},       a_{bk} = ℓ_{ik} p_{bi}(0)     (i summed)

b labels the batch of initial temperatures. At long times only the slowest mode k = 2 survives, so the
smaller |a_2|, the closer to equilibrium. A hot start can beat a warm one (the Mpemba effect).
"""

import torch


def metropolis_rates(u: torch.Tensor, A: torch.Tensor, T_bath: float) -> torch.Tensor:
    r"""W_{ij} = A_{ij} min(1, e^{-(U_i-U_j)/T_b}) off the diagonal, columns summing to zero.   → (N, N)"""
    uphill = u[:, None] - u[None, :]  # (N, N)  U_i - U_j: energy gained jumping j → i
    W = A * torch.clamp(torch.exp(-uphill / T_bath), max=1.0)  # (N, N)
    return W - torch.diag(W.sum(dim=0))  # W_jj = -Σ_k W_kj  (escape rate out of j)


def master_rhs(p: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    r"""\dot p_{bi} = W_{ij} p_{bj}   (j summed).   p: (B, N) → (B, N)"""
    return torch.einsum("ij,bj->bi", W, p)


def mode_amplitudes(p0: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    r"""a_{bk} = ℓ_{ik} p_{bi}(0)   (i summed): every start projected on every left eigenvector.   → (B, N)"""
    return torch.einsum("ik,bi->bk", left, p0)


def evolve(a: torch.Tensor, lam: torch.Tensor, right: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    r"""p_{bi}(t) = \sum_k r_{ik} a_{bk} e^{λ_k t}   (k summed, k = 1 is the stationary state).   → (T, B, N)"""
    decay = torch.exp(times[:, None] * lam[None, :])  # (T, N)  e^{λ_k t}
    return torch.einsum("ik,bk,tk->tbi", right, a, decay)


def kl_divergence(p: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    r"""D(p ‖ π) = \sum_i p_i ln(p_i / π_i): distance to equilibrium (≥ 0, zero only at p = π).   (..., N) → (...)"""
    p = p.clamp_min(1e-300)
    return (p * (p.log() - pi.log())).sum(dim=-1).clamp_min(0.0)  # round-off can give -1e-12 once p ≈ π
