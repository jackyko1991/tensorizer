r"""Physics → tensor translation.

A neutrino born with flavour α (e, μ, τ) is a superposition of three mass states i = 1, 2, 3 with amplitudes
U^*_{αi}, the elements of the PMNS mixing matrix. In vacuum each mass state picks up the phase
φ_i = m_i^2 L / 2E (natural units), so the amplitude to be detected as flavour β after a distance L is

    A_{βα}(L) = U_{βi} \, U^*_{αi} \, e^{-i m_i^2 L / 2E}        (i summed),
    P(α → β) = |A_{βα}|^2.

With L in km, E in GeV and m² in eV²:  m_i^2 L / 2E = 2.534 · m_i^2 L / E.

The PMNS matrix is a product of three rotations, one of them carrying the CP phase δ:

    U = R_{23}(θ_{23}) \, U_{13}(θ_{13}, δ) \, R_{12}(θ_{12}).
"""

import torch

PHASE = 2.534  # (1/2E) m² L in units eV², km, GeV  (= 2 × 1.267)


def pmns(theta12: float, theta13: float, theta23: float, delta: float) -> torch.Tensor:
    r"""U_{αi} = [R_{23}]_{αβ} [U_{13}]_{βγ} [R_{12}]_{γi}   (β, γ summed).   angles in radians → (3, 3) complex"""
    c12, s12 = _cos_sin(theta12)
    c13, s13 = _cos_sin(theta13)
    c23, s23 = _cos_sin(theta23)
    e = torch.exp(torch.tensor(1j * delta, dtype=torch.complex128))
    zero, one = torch.zeros((), dtype=torch.complex128), torch.ones((), dtype=torch.complex128)
    r23 = torch.stack([one, zero, zero, zero, c23, s23, zero, -s23, c23]).reshape(3, 3)
    u13 = torch.stack([c13, zero, s13 / e, zero, one, zero, -s13 * e, zero, c13]).reshape(3, 3)
    r12 = torch.stack([c12, s12, zero, -s12, c12, zero, zero, zero, one]).reshape(3, 3)
    return torch.einsum("ab,bc,ci->ai", r23, u13, r12)


def _cos_sin(theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor(theta, dtype=torch.float64)
    return torch.cos(t).to(torch.complex128), torch.sin(t).to(torch.complex128)


def phases(m2: torch.Tensor, l_over_e: torch.Tensor) -> torch.Tensor:
    r"""e^{-i m_i^2 L/2E} for every baseline.   m2: (3,) eV², l_over_e: (K,) km/GeV → (K, 3) complex"""
    phi = PHASE * torch.einsum("k,i->ki", l_over_e, m2)
    return torch.exp(-1j * phi)


def amplitude(u: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    r"""A_{kβα} = U_{βi} U^*_{αi} e^{-iφ_{ki}}   (i summed).   u: (3, 3), ph: (K, 3) → (K, 3, 3)"""
    return torch.einsum("bi,ai,ki->kba", u, u.conj(), ph)


def probability(u: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    r"""P_{kβα} = |A_{kβα}|^2: probability α → β at baseline k.   → (K, 3, 3) real"""
    return amplitude(u, ph).abs() ** 2


def two_flavour(theta: float, dm2: float, l_over_e: torch.Tensor) -> torch.Tensor:
    r"""P = \sin^2 2θ \, \sin^2(1.267 Δm^2 L/E): the textbook two-flavour formula."""
    return torch.sin(torch.tensor(2 * theta)) ** 2 * torch.sin(PHASE / 2 * dm2 * l_over_e) ** 2
