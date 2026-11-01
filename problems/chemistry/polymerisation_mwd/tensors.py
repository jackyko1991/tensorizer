"""Build the concrete tensors: chain lengths, the shift (Toeplitz) index and the initial populations.

The state is one (S, L) tensor: row s = 0 holds step-growth chains, row s = 1 living chain-growth chains, and
column x - 1 the number of chains of length x (x = 1 … L monomer units), per initial monomer (step growth) or per
initiator (living). Both rows start as unimers: P_x(0) = δ_{x,1}.
"""

import torch

STEP, LIVING = 0, 1


def lengths(L: int, device: torch.device) -> torch.Tensor:
    """x = 1, 2, …, L as float64, shape (L,)."""
    return torch.arange(1, L + 1, dtype=torch.float64, device=device)


def shift_index(L: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Index tensor for T_{xy} = P_{x-y}: idx (L, L) and a mask that is 1 only where 1 ≤ x - y ≤ L.

    With columns 0 … L-1 standing for lengths 1 … L, a chain of length x+1 forms from lengths y+1 and x-y,
    i.e. column x - y - 1 (needs x - y ≥ 1).
    """
    x = torch.arange(L, device=device)
    diff = x[:, None] - x[None, :] - 1  # column of the partner chain
    mask = (diff >= 0).to(torch.float64)
    return diff.clamp_min(0), mask


def initial_state(L: int, device: torch.device) -> torch.Tensor:
    """P_{s,x}(0) = δ_{x,1} for both mechanisms: shape (2, L)."""
    P = torch.zeros(2, L, dtype=torch.float64, device=device)
    P[:, 0] = 1.0
    return P


def chain_length_cap(p_max: float) -> int:
    """Enough columns L that the Flory tail p^L at the final conversion is negligible (≲ 1e-7 of the chains)."""
    xn = 1.0 / (1.0 - p_max)
    return int(min(2000, max(200, 16 * xn + 100)))
