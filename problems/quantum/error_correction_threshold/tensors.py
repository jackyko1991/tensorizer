"""Build the concrete tensors: the noise samples (all shots at once) and the decoder's lookup tables."""

import torch


def distances(d_max: int) -> list[int]:
    """Odd code distances 3, 5, …, d_max."""
    return list(range(3, d_max + 1, 2))


def error_rates(n: int, p_min: float, p_max: float) -> torch.Tensor:
    """n physical error rates p, log-spaced: (P,)."""
    return torch.logspace(torch.log10(torch.tensor(p_min)), torch.log10(torch.tensor(p_max)), n, dtype=torch.float64)


def sample_noise(
    shots: int, rounds: int, d: int, p: float, q: float, gen: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent errors for every shot in one tensor.

    e_{sti}: (S, R, d) bool, data qubit i flips in round t with probability p
    m_{stj}: (S, R, d − 1) bool, check j reads wrong in round t with probability q
    """
    e = torch.rand(shots, rounds, d, generator=gen).to(device) < p
    m = torch.rand(shots, rounds, d - 1, generator=gen).to(device) < q
    return e, m


def bit_table(d: int, device: torch.device | None = None) -> torch.Tensor:
    """bits[x, i] = bit i of the integer x: every one of the 2^d error patterns, (2^d, d) long."""
    x = torch.arange(2**d, device=device)
    return (x[:, None] >> torch.arange(d, device=device)) & 1


def pack_bits(s: torch.Tensor) -> torch.Tensor:
    """Pack a bit string (…, n) of 0/1 (a syndrome or an error pattern) into one integer Σ_j s_j 2^j: (…,)."""
    powers = 2 ** torch.arange(s.shape[-1], device=s.device)
    return torch.einsum("...j,j->...", s.long(), powers)


def mismatch_table(d: int, device: torch.device | None = None) -> torch.Tensor:
    """K[σ, x] = number of checks where the observed syndrome σ disagrees with the syndrome of pattern x.

    Shape (2^{d−1}, 2^d). K[σ, x] measurement errors explain seeing σ while the data error is x.
    """
    bits = bit_table(d, device)  # (2^d, d)
    syn = (bits[:, :-1] + bits[:, 1:]) % 2  # (2^d, d − 1) syndrome of each pattern
    obs = bit_table(d - 1, device)  # (2^{d−1}, d − 1) every possible observed syndrome
    return (obs[:, None, :] != syn[None, :, :]).sum(-1)  # (2^{d−1}, 2^d)
