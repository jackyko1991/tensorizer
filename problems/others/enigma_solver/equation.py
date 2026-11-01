r"""Machine → tensor translation.

The Enigma is not physics, but it is a beautiful product of permutation matrices. A letter x is a one-hot vector
e_x (or just its index). A wiring W is a 26×26 permutation matrix, W e_x = e_{w[x]}. A rotor turned o steps away
from its rest position is the same wiring seen through a cyclic shift ρ (ρ e_x = e_{x+1}):

    R = ρ^{-o} W ρ^{o}        index form:  R_{ij} = (ρ^{-o})_{ia} W_{ab} (ρ^{o})_{bj}     (a, b summed)
                              as indices:  x → (w[(x + o) mod 26] − o) mod 26

One key press sends the current through the plugboard S, the fast right rotor R₁, the middle R₂, the slow left R₃,
the reflector U and back again:

    E = S⁻¹ R₁⁻¹ R₂⁻¹ R₃⁻¹ U R₃ R₂ R₁ S
    E_{ij} = S_{ai} R₁_{ba} R₂_{cb} R₃_{dc} U_{de} R₃_{ef} R₂_{fg} R₁_{gh} S_{hj}     (a … h summed)

using R⁻¹ = Rᵀ for a permutation matrix. U is a symmetric involution without fixed points (U² = I, U_ii = 0), so
E = Q⁻¹ U Q is too: E² = I (encryption = decryption) and E_ii = 0 (no letter is ever encrypted to itself).
Before each key press the rotors step: the right rotor always; the middle one when the right rotor sits at its
notch; the left one, and the middle one again, when the middle rotor sits at its own notch (the double step).
"""

import torch


def rotor_matrix(w: torch.Tensor, rho_minus: torch.Tensor, rho_plus: torch.Tensor) -> torch.Tensor:
    r"""R_{tij} = (ρ^{-o_t})_{ia} W_{ab} (ρ^{o_t})_{bj}   (a, b summed).   W (26, 26), ρ (T, 26, 26) → (T, 26, 26)"""
    return torch.einsum("tia,ab,tbj->tij", rho_minus, w, rho_plus)


def keypress_matrix(s: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, u: torch.Tensor):
    r"""E_{tij} = S_{ai} R1_{tba} R2_{tcb} R3_{tdc} U_{de} R3_{tef} R2_{tfg} R1_{tgh} S_{hj}   (a … h summed).

    S, U (26, 26); R1 (right), R2 (middle), R3 (left) (T, 26, 26) → (T, 26, 26), one matrix per key press.
    """
    return torch.einsum("ai,tba,tcb,tdc,de,tef,tfg,tgh,hj->tij", s, r1, r2, r3, u, r3, r2, r1, s)


def rotor(x: torch.Tensor, w: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    r"""Index form of R = ρ^{-o} W ρ^{o}: x → (w[(x + o) mod 26] − o) mod 26.   Any shape, x and o broadcast."""
    return (w[(x + o) % 26] - o) % 26


def rotor_table(w: torch.Tensor) -> torch.Tensor:
    r"""F_{ox} = (w[(x + o) mod 26] − o) mod 26: the rotor at every offset o (row o is R for that o).

    (26,) → (26, 26)
    """
    a = torch.arange(26, device=w.device)
    return rotor(a[None, :], w, a[:, None])


def encipher(x: torch.Tensor, offsets: torch.Tensor, m: dict, s: torch.Tensor) -> torch.Tensor:
    r"""y = S⁻¹ R₁⁻¹ R₂⁻¹ R₃⁻¹ U R₃ R₂ R₁ S x for a whole batch of letters.

    x (...,) letter indices; offsets (..., 3) = window position − ring setting of (left, mid, right);
    m holds the rotor tables f, f_inv (3, 26, 26) (see rotor_table) and the reflector u (26,); s is the plugboard.
    Each rotor is one table look-up F[o, x], which is the same as R x but fast for millions of letters.
    """
    y = s[x]
    for k in (2, 1, 0):  # right → middle → left
        y = m["f"][k][offsets[..., k], y]
    y = m["u"][y]
    for k in (0, 1, 2):  # left → middle → right, through the inverse wirings
        y = m["f_inv"][k][offsets[..., k], y]
    return s[y]  # S is its own inverse


def step(pos: torch.Tensor, notch: torch.Tensor) -> torch.Tensor:
    """One key press of the stepping mechanism, double step included.   pos (..., 3) window letters → (..., 3)"""
    mid_at_notch = pos[..., 1] == notch[1]
    right_at_notch = pos[..., 2] == notch[2]
    kick = torch.stack([mid_at_notch, mid_at_notch | right_at_notch, torch.ones_like(mid_at_notch)], dim=-1)
    return (pos + kick.long()) % 26


def letter_counts(y: torch.Tensor) -> torch.Tensor:
    r"""c_{bk} = \sum_t [y_{bt} = k]: how often each letter occurs in every candidate text.   (B, T) → (B, 26)"""
    counts = torch.zeros(y.shape[0], 26, device=y.device)
    return counts.scatter_add_(1, y, torch.ones_like(y, dtype=counts.dtype))


def index_of_coincidence(c: torch.Tensor) -> torch.Tensor:
    r"""IoC_b = c_{bk}(c_{bk} − 1) / (N(N − 1))   (k summed): English ≈ 0.066, random ≈ 0.038.   (B, 26) → (B,)"""
    n = c[0].sum()
    return torch.einsum("bk,bk->b", c, c - 1) / (n * (n - 1))


def unigram_score(c: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    r"""L_b = c_{bk} log p_k / N   (k summed): mean log-likelihood per letter under English.   (B, 26) → (B,)"""
    return torch.einsum("bk,k->b", c, log_p) / c[0].sum()
