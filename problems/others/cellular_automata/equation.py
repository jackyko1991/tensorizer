r"""Rules → tensor translation.

Elementary cellular automata (Wolfram 1983): a ring of cells x_i ∈ {0, 1}; each new cell depends only on itself and
its two neighbours. The three cells form a 3-bit code

    k_{ri} = 4\,x_{r,i-1} + 2\,x_{r,i} + x_{r,i+1} = P_{rij}\,w_j,\qquad w = (4, 2, 1)   (j summed),

and the rule number r ∈ {0, …, 255} written in binary is the lookup table: bit k of r is the new state,

    x'_{ri} = T_{r,\,k_{ri}},\qquad T_{rk} = \lfloor r / 2^k \rfloor \bmod 2.

Putting all 256 rules in one (256, W) tensor, one `gather` advances every rule at once. Mirror symmetry
(left ↔ right) swaps codes k ↔ k̃ with the outer bits exchanged; complement symmetry (0 ↔ 1) maps
T_k → 1 − T_{7−k}. Together they cut 256 rules to 88 distinct behaviours.

Lenia (Chan 2019) is a continuous cellular automaton: states A ∈ [0, 1], a smooth ring kernel K of radius R
instead of the 3-cell neighbourhood, and a smooth growth function instead of a table,

    U = K * A,\qquad G(U) = 2\,e^{-(U-\mu)^2 / 2\sigma^2} - 1,\qquad A' = \operatorname{clip}(A + G(U)/T,\ 0,\ 1).

The convolution is a product of Fourier transforms on the periodic grid.
"""

import torch

W = torch.tensor([4, 2, 1])  # weights of left, centre, right in the neighbourhood code


def neighbourhood_code(x: torch.Tensor) -> torch.Tensor:
    r"""k_{ri} = P_{rij} w_j   (j summed over left, centre, right; periodic ring).   (R, W) → (R, W) int64"""
    patch = torch.stack([torch.roll(x, 1, -1), x, torch.roll(x, -1, -1)], dim=-1).long()  # (R, W, 3)
    return torch.einsum("rij,j->ri", patch, W.to(x.device))


def eca_step(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    r"""x'_{ri} = T_{r, k_{ri}}: one generation of every rule at once.   (R, W), (R, 8) → (R, W) uint8"""
    return table.gather(1, neighbourhood_code(x))


def mirror_rule(rules: torch.Tensor) -> torch.Tensor:
    """Rule number of the left-right mirror image: bit k̃ of the new rule = bit k of the old, k̃ = k with bits 0 and 2
    swapped.   (R,) → (R,)"""
    k = torch.arange(8)
    k_mirror = ((k & 1) << 2) | (k & 2) | ((k >> 2) & 1)  # (8,)
    bits = (rules[:, None] >> k_mirror) & 1  # (R, 8) bit k of the mirror = bit k̃ of the rule
    return (bits << k).sum(-1)


def complement_rule(rules: torch.Tensor) -> torch.Tensor:
    """Rule number with 0 and 1 exchanged: T'_k = 1 − T_{7−k}.   (R,) → (R,)"""
    k = torch.arange(8)
    bits = 1 - ((rules[:, None] >> (7 - k)) & 1)  # (R, 8)
    return (bits << k).sum(-1)


def growth(u: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    r"""G(U) = 2 exp(−(U − μ)²/2σ²) − 1 ∈ [−1, 1]: grow where the neighbourhood is just right.   any shape"""
    return 2 * torch.exp(-((u - mu) ** 2) / (2 * sigma**2)) - 1


def lenia_step(a: torch.Tensor, k_hat: torch.Tensor, mu: float, sigma: float, t: float) -> torch.Tensor:
    r"""A' = clip(A + G(K * A)/T, 0, 1) with K * A = IFFT(FFT(A)·FFT(K)).   (..., S, S) → (..., S, S)"""
    u = torch.fft.ifft2(torch.fft.fft2(a) * k_hat).real
    return (a + growth(u, mu, sigma) / t).clamp(0, 1)
