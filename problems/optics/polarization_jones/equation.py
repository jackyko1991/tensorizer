r"""Physics → tensor translation.

For light travelling along z the transverse field is \mathcal E_i(t) = \operatorname{Re}(E_i e^{-i\omega t}),
i ∈ {x, y}. The complex 2-vector E_i (the Jones vector) holds the amplitudes and the relative phase
of the two components: in phase ⇒ linear, 90° apart with equal size ⇒ circular, otherwise elliptical.

Linear optical elements act as complex 2×2 matrices. In the element's own axes:

    polarizer:              diag(1, 0)             (keep the field along its axis)
    wave plate, delay Γ:    diag(1, e^{iΓ})        (Γ = π/2 quarter-wave, Γ = π half-wave)

Rotated by an angle θ in the lab:  M(θ) = R(θ)\, M_0\, R(-θ). A sequence of elements is the
matrix product, rightmost first, and the output field and intensity are

    E^{out}_i = P_{ij} W_{jk} E^{in}_k,        I = E^*_i E_i

(repeated j, k and i ⇒ summed). einsum('ij,j->i') applies a matrix, einsum('ij,jk->ik') chains two.
"""

import math

import torch


def rotation(theta: float) -> torch.Tensor:
    """R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]: shape (2, 2), complex so it multiplies Jones matrices."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)


def rotated(m0: torch.Tensor, theta: float) -> torch.Tensor:
    """M(θ) = R(θ) M₀ R(−θ): the element M₀ turned by θ.   (2, 2) → (2, 2)"""
    return chain(rotation(-theta), m0, rotation(theta))


def polarizer(theta: float) -> torch.Tensor:
    """Linear polarizer with transmission axis at angle θ: (2, 2) complex."""
    return rotated(torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64), theta)


def wave_plate(retardance: float, alpha: float) -> torch.Tensor:
    """Wave plate delaying the slow axis by Γ = retardance, fast axis at angle α: (2, 2) complex."""
    delay = complex(math.cos(retardance), math.sin(retardance))  # e^{iΓ}
    return rotated(torch.tensor([[1, 0], [0, delay]], dtype=torch.complex64), alpha)


def chain(*elements: torch.Tensor) -> torch.Tensor:
    r"""Elements in the order the light meets them → one matrix  M = E_n … E_2 E_1.

    Each step is  M_{ik} ← E_{ij} M_{jk}  (repeated j summed).
    """
    m = torch.eye(2, dtype=torch.complex64)
    for e in elements:
        m = torch.einsum("ij,jk->ik", e, m)
    return m


def apply(m: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""E^{out}_i = M_{ij} E_j.   (2, 2), (2,) → (2,)"""
    return torch.einsum("ij,j->i", m, e)


def intensity(e: torch.Tensor) -> torch.Tensor:
    r"""I = E^*_i E_i (a real number).   (2,) → ()"""
    return torch.einsum("i,i->", e.conj(), e).real


def unpolarized_intensity(m: torch.Tensor) -> torch.Tensor:
    """Unpolarized light of intensity 1 = equal, incoherent mix of x- and y-polarized light (½ each)."""
    ex = torch.tensor([1, 0], dtype=torch.complex64)
    ey = torch.tensor([0, 1], dtype=torch.complex64)
    return 0.5 * (intensity(apply(m, ex)) + intensity(apply(m, ey)))


def field_trace(e: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    r"""\mathcal E_i(t) = Re(E_i e^{-iωt}) at the phases ωt: (2,), (T,) → (T, 2). It draws the ellipse."""
    return torch.einsum("i,t->ti", e, torch.exp(-1j * wt)).real
