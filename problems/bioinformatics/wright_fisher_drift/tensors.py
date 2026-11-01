"""Build the concrete tensors: starting allele frequencies of the replicate populations and the selection grid."""

import torch

# selection strengths for the fixation experiment, as the population-scaled 4Ns
SCALED_S = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]


def initial_frequencies(n_replicates: int, p0: float, device: torch.device) -> torch.Tensor:
    """p_{r,0} = p0 for every replicate population r.   (R,)"""
    return torch.full((n_replicates,), p0, device=device, dtype=torch.float64)


def new_mutants(n_s: int, n_replicates: int, N: int, device: torch.device) -> torch.Tensor:
    """A single new copy of A in each population: p = 1/2N, for every selection value and replicate.   (S, R)"""
    return torch.full((n_s, n_replicates), 1 / (2 * N), device=device, dtype=torch.float64)


def selection_grid(N: int, s_user: float, device: torch.device) -> torch.Tensor:
    """s values (S,) with 4Ns from SCALED_S, plus the slider value s if it is not already there."""
    s = [x / (4 * N) for x in SCALED_S]
    if all(abs(s_user - v) > 1e-12 for v in s):
        s.append(s_user)
    return torch.tensor(sorted(s), device=device, dtype=torch.float64)
