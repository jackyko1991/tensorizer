r"""Physics → tensor translation.

The Bloch equation for an ensemble of isochromats n (spins that see the same field), in the frame rotating
at the Larmor frequency, with the field written as an angular frequency \Omega = \gamma B:

    dM_{ni}/dt = \varepsilon_{ijk} M_{nj} \Omega_{nk}  -  R_{ij} (M_{nj} - M^0_j)        (j, k summed)

    \Omega_{nk} = (\omega_1 \cos\phi,\ \omega_1 \sin\phi,\ \Delta\omega_n)    RF pulse in x/y, off-resonance along z
    R_{ij}      = diag(1/T_2, 1/T_2, 1/T_1),     M^0_j = (0, 0, M_0)

The first term is the cross product M × Ω (the same \varepsilon_{ijk} contraction as the Lorentz force);
the second pulls M_z back to M_0 with T_1 and kills M_x, M_y with T_2. The coil measures the average
transverse magnetisation of all isochromats, a sum over n:

    s = (1/N) \sum_n (M_{nx} + i M_{ny})
"""

import math

import torch


def omega(dw: torch.Tensor, w1: float, phase: float) -> torch.Tensor:
    r"""\Omega_{nk} = (\omega_1\cos\phi, \omega_1\sin\phi, \Delta\omega_n) in rad/ms.   dw: (N,) → (N, 3)"""
    x = torch.full_like(dw, w1 * math.cos(phase))
    y = torch.full_like(dw, w1 * math.sin(phase))
    return torch.stack([x, y, dw], dim=-1)


def bloch_rhs(M: torch.Tensor, Om: torch.Tensor, eps: torch.Tensor, R: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
    r"""dM_{ni}/dt = \varepsilon_{ijk} M_{nj} \Omega_{nk} - R_{ij}(M_{nj} - M^0_j).   M, Om: (N, 3) → (N, 3)"""
    precession = torch.einsum("ijk,nj,nk->ni", eps, M, Om)  # j, k summed: M × Ω for every isochromat
    relaxation = torch.einsum("ij,nj->ni", R, M - M0)  # j summed: diagonal relaxation rates
    return precession - relaxation


def signal(M: torch.Tensor) -> torch.Tensor:
    r"""s = (1/N) \sum_n (M_{nx} + i M_{ny}).   M: (..., N, 3) → (...,) complex"""
    return torch.complex(M[..., 0], M[..., 1]).mean(dim=-1)
