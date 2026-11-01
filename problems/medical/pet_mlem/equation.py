r"""Physics → tensor translation.

1. Decay and annihilation (for a batch of events e, components i ∈ {x, y}, photons p ∈ {1, 2}):

    a_{ei} = d_{ei} + \sigma_+ \xi_{ei}                       positron range: small random step ξ ~ N(0, 1)
    u_{epi} = \pm (\cos\varphi_e, \sin\varphi_e)               two 511 keV photons, back to back
    h_{epi} = a_{ei} + t_{ep} u_{epi},   |h_{ep}| = R           where each photon hits the detector ring

   The ring hits h_{ep} define the line of response (LOR) with sinogram coordinates (θ, s):
   s = h_{e1,i} n_i(θ) is the signed distance of the LOR from the centre.

2. Reconstruction.  The expected counts in sinogram bin b are  \bar y_b = A_{bj} \lambda_j + r_b
   (repeated j summed: a matrix–vector product), where A_{bj} is the probability that a decay in pixel j
   is detected in bin b and r_b is the random/scatter background.  MLEM maximises the Poisson likelihood
   with the multiplicative update

    \lambda_j^{k+1} = \frac{\lambda_j^k}{A_{bj} 1_b} \; A_{bj} \frac{y_b}{A_{bj'} \lambda_{j'}^k + r_b}

   in which every A_{bj} … _b is summed over the bins b (a back-projection, einsum('bj,b->j')).
"""

import math

import torch

E_PHOTON = 511.0  # keV = electron rest energy m_e c²


def positron_range(decay: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    r"""a_{ei} = d_{ei} + σ ξ_{ei}: the positron travels a short random distance before annihilating.  (E, 2)"""
    xi = torch.randn(decay.shape, generator=generator).to(decay.device)
    return decay + sigma * xi


def annihilation_photons(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Directions u_{epi} (E, 2, 2) and energies (E, 2) of the photon pair.

    The positron and electron are (almost) at rest, so momentum conservation sends the two photons in
    exactly opposite directions, and energy conservation gives each one m_e c² = 511 keV.
    """
    u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (E, 2) photon 1
    directions = torch.stack([u, -u], dim=1)  # (E, 2 photons, 2 components)
    energies = torch.full(phi.shape + (2,), E_PHOTON, device=phi.device)
    return directions, energies


def ring_hits(a: torch.Tensor, u: torch.Tensor, R: float) -> torch.Tensor:
    r"""h_{epi} = a_{ei} + t_{ep} u_{epi} with |h| = R and t > 0.   a: (E, 2), u: (E, 2, 2) → (E, 2, 2).

    |a + t u|² = R²  ⇒  t = -a_i u_i + \sqrt{(a_i u_i)² - a_i a_i + R²}   (repeated i summed)
    """
    au = torch.einsum("ei,epi->ep", a, u)  # (E, 2)
    aa = torch.einsum("ei,ei->e", a, a)  # (E,)
    t = -au + torch.sqrt(au**2 - aa[:, None] + R**2)
    return a[:, None, :] + t[..., None] * u


def lor_coordinates(h1: torch.Tensor, h2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sinogram coordinates of the line through h1 and h2 (each (E, 2)): angle θ ∈ [0, π) of the normal
    n = (cos θ, sin θ), and signed distance s = h1_i n_i from the centre."""
    d = h2 - h1
    theta = torch.remainder(torch.atan2(d[:, 1], d[:, 0]) - math.pi / 2, math.pi)  # normal ⟂ line, folded
    n = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    s = torch.einsum("ei,ei->e", h1, n)
    return theta, s


def forward_project(A: torch.Tensor, lam: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""\bar y_b = A_{bj} λ_j + r_b   (repeated j summed).   A: (B, J), λ: (J,) → (B,)"""
    return torch.einsum("bj,j->b", A, lam) + r


def back_project(A: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""A_{bj} q_b   (repeated b summed): smear every bin value back along its LOR.   → (J,)"""
    return torch.einsum("bj,b->j", A, q)


def mlem_step(lam: torch.Tensor, A: torch.Tensor, y: torch.Tensor, r: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    r"""λ^{k+1}_j = λ^k_j / s_j · A_{bj} (y_b / ȳ_b),   with sensitivity s_j = A_{bj} 1_b."""
    ratio = y / forward_project(A, lam, r).clamp(min=1e-12)  # (B,) measured / expected
    return lam / sens.clamp(min=1e-12) * back_project(A, ratio)


def log_likelihood(y: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
    r"""Poisson log-likelihood (without the constant log y!):  L = y_b \log ȳ_b - ȳ_b, summed over b."""
    return (y * torch.log(y_bar.clamp(min=1e-12)) - y_bar).sum()
