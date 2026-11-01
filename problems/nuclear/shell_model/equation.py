r"""Physics → tensor translation.

In the shell model each nucleon moves independently in an average potential: a Woods–Saxon well plus a
spin–orbit term that depends on how the orbital angular momentum l and the spin s are aligned,

    V(r) = -V_0 f(r) + V_{so} \, r_0^2 \, \frac{1}{r}\frac{df}{dr} \, ⟨\mathbf l·\mathbf s⟩,
    f(r) = \frac{1}{1 + e^{(r - R)/a}},    R = r_0 A^{1/3},
    ⟨\mathbf l·\mathbf s⟩ = \tfrac12 [\,j(j+1) - l(l+1) - \tfrac34\,].

Writing ψ = (u(r)/r) × (spin-angle part), each channel k = (l, j) gives a 1D radial equation. On a grid r_i
with the second-difference matrix D_{ij} it is a matrix eigenproblem, one per channel:

    H_{kij} = -\frac{\hbar^2}{2m} D_{ij} + \Big[V_k(r_i) + \frac{\hbar^2 l_k(l_k+1)}{2m r_i^2}\Big] δ_{ij},
    H_{kij} u_{kjn} = E_{kn} u_{kin}        (j summed).

Each level E_{kn} holds 2j + 1 nucleons (the m_j states).
"""

import torch

HBAR2_2M = 20.736  # ħ²/2m for a nucleon in MeV fm²


def woods_saxon_shape(r: torch.Tensor, radius: float, a: float) -> torch.Tensor:
    r"""f(r_i) = 1 / (1 + e^{(r_i − R)/a}).   (N,) → (N,)"""
    return torch.sigmoid((radius - r) / a)


def ls_factor(ell: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    r"""⟨l·s⟩_k = ½ [j(j+1) − l(l+1) − ¾]: +l/2 for j = l + ½, −(l+1)/2 for j = l − ½.   (K,) → (K,)"""
    return 0.5 * (j * (j + 1) - ell * (ell + 1) - 0.75)


def potential(
    r: torch.Tensor, ell: torch.Tensor, j: torch.Tensor, v0: float, v_so: float, radius: float, a: float, r0: float
) -> torch.Tensor:
    r"""V_k(r_i) + ħ² l_k(l_k+1) / 2m r_i²: well, spin–orbit and centrifugal terms.   → (K, N)"""
    f = woods_saxon_shape(r, radius, a)  # (N,)
    dfdr = -f * (1 - f) / a  # (N,) negative: the well edge
    central = -v0 * f  # (N,)
    spin_orbit = v_so * r0**2 * torch.einsum("k,i->ki", ls_factor(ell, j), dfdr / r)  # (K, N)
    centrifugal = HBAR2_2M * torch.einsum("k,i->ki", ell * (ell + 1), 1 / r**2)  # (K, N)
    return central + spin_orbit + centrifugal


def hamiltonian(d2: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""H_{kij} = −(ħ²/2m) D_{ij} + V_{ki} δ_{ij}.   d2: (N, N), v: (K, N) → (K, N, N)"""
    return -HBAR2_2M * d2 + torch.diag_embed(v)


def occupancy(j: torch.Tensor) -> torch.Tensor:
    """2j + 1 nucleons of one kind fit in a level of angular momentum j."""
    return 2 * j + 1
