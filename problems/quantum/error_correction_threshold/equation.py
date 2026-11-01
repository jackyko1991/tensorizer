r"""Physics → tensor translation for the bit-flip repetition code.

A distance-d repetition code stores one logical bit in d data qubits (|0_L⟩ = |00…0⟩). Each round t,
every data qubit flips with probability p, and the d − 1 parity checks Z_i Z_{i+1} are measured,
each reading wrong with probability q. With s = shot, t = round, i = data qubit, j = check:

    x_{sti} = ⊕_{t' ≤ t} e_{st'i}                      accumulated flips (cumulative sum mod 2)
    σ_{stj} = H_{ji} x_{sti} + m_{stj}   (mod 2)          i summed: measured syndrome
    D_{stj} = σ_{stj} ⊕ σ_{s,t-1,j}                     detection events (what the decoder looks at)

H_{ji} is the (d − 1) × d parity-check matrix, H_{j,j} = H_{j,j+1} = 1. Without measurement errors the
best decoder is a majority vote, and the logical error probability is the binomial tail

    P_L = Σ_{k > d/2} C(d, k) p^k (1 − p)^{d − k}.

Per-round logical error ε from P_L after R rounds:  1 − 2 P_L = (1 − 2ε)^R.
Error-suppression factor:  Λ_{d/d+2} = ε_d / ε_{d+2}.
"""

import math

import torch


def parity_check(d: int, device: torch.device | None = None) -> torch.Tensor:
    """H_{ji}: (d − 1, d) with ones on the diagonal and the superdiagonal."""
    h = torch.zeros(d - 1, d, dtype=torch.long, device=device)
    j = torch.arange(d - 1, device=device)
    h[j, j] = 1
    h[j, j + 1] = 1
    return h


def accumulate(flips: torch.Tensor) -> torch.Tensor:
    r"""x_{sti} = ⊕_{t' ≤ t} e_{st'i}.   flips (S, R, d) bool → (S, R, d) long in {0, 1}."""
    return torch.cumsum(flips.long(), dim=1) % 2


def syndrome(x: torch.Tensor, h: torch.Tensor, meas_flips: torch.Tensor | None = None) -> torch.Tensor:
    r"""σ_{stj} = H_{ji} x_{sti} + m_{stj} (mod 2) — repeated i summed.   (S, R, d) → (S, R, d − 1)."""
    s = torch.einsum("ji,sti->stj", h, x) % 2
    if meas_flips is not None:
        s = (s + meas_flips.long()) % 2
    return s


def detection_events(s: torch.Tensor) -> torch.Tensor:
    r"""D_{stj} = σ_{stj} ⊕ σ_{s,t-1,j} with σ_{s,0,j} = 0.   (S, R, d − 1) → (S, R, d − 1)."""
    previous = torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], dim=1)
    return (s + previous) % 2


def majority_vote_fails(x_final: torch.Tensor) -> torch.Tensor:
    r"""Logical error ⇔ more than half the data qubits are flipped: Σ_i x_{si} > d/2.   (S, d) → (S,) bool."""
    d = x_final.shape[-1]
    return torch.einsum("si->s", x_final) > d / 2


def binomial_logical_error(p: torch.Tensor, d: int) -> torch.Tensor:
    r"""Exact majority-vote failure: P_L = Σ_{k > d/2} C(d, k) p^k (1 − p)^{d − k}."""
    p = torch.as_tensor(p, dtype=torch.float64)
    return sum(math.comb(d, k) * p**k * (1 - p) ** (d - k) for k in range(d // 2 + 1, d + 1))


def per_round(p_logical: torch.Tensor, rounds: int) -> torch.Tensor:
    r"""ε from 1 − 2 P_L = (1 − 2ε)^R (an error per round; for R = 1, ε = P_L)."""
    x = (1 - 2 * p_logical).clamp(min=0.0)
    return (1 - x ** (1.0 / rounds)) / 2


def log_weight(p: float) -> float:
    """Matching weight of one error with probability p: w = ln((1 − p)/p) (minus its log-likelihood ratio)."""
    p = min(max(p, 1e-12), 0.5 - 1e-12)
    return math.log((1 - p) / p)
