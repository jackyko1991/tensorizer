r"""Physics → tensor translation.

Planck's law: the energy density of black-body radiation per unit wavelength is

    u(\lambda, T) = \frac{8 \pi h c}{\lambda^5} \, \frac{1}{e^{hc / \lambda k T} - 1}

For a batch of temperatures T_a and a grid of wavelengths λ_b the index form is

    u_{ab} = \frac{8 \pi h c}{\lambda_b^5} \, \frac{1}{\exp(hc / \lambda_b k T_a) - 1}

No index is summed: T_a as a column (A, 1) broadcast against λ_b as a row (1, L) gives the (A, L) table.
Two classic laws follow from it:

    Wien:              \lambda_{\max} T = b \approx 2.898 \times 10^{-3}\ \text{m K}
    Stefan–Boltzmann:  U_a = \int u_{ab} \, d\lambda_b = \frac{4 \sigma}{c} T_a^4

The integral sums over the repeated wavelength index b (torch.trapezoid along the last dimension).
"""

import math

import torch

H = 6.62607015e-34  # Planck constant (J s)
C = 2.99792458e8  # speed of light (m/s)
K_B = 1.380649e-23  # Boltzmann constant (J/K)
SIGMA = 2 * math.pi**5 * K_B**4 / (15 * H**3 * C**2)  # Stefan–Boltzmann constant ≈ 5.670e-8 W/(m² K⁴)
WIEN_B = 2.897771955e-3  # Wien's displacement constant (m K)


def planck(lam: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    r"""u_{ab} = 8πhc / λ_b⁵ / (exp(hc / λ_b k T_a) − 1)  in J/m⁴.   lam: (L,), T: (A,) → (A, L)"""
    x = H * C / (lam[None, :] * K_B * T[:, None])  # (A, L)  hc / λkT by broadcasting
    return 8 * math.pi * H * C / lam[None, :] ** 5 / torch.expm1(x)  # expm1(x) = eˣ − 1, accurate for small x


def peak_wavelength(lam: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """λ_max of every spectrum: the grid wavelength with the largest u.   (A, L) → (A,)"""
    return lam[u.argmax(dim=-1)]


def total_energy(lam: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    r"""U_a = Σ_b ½(u_{ab} + u_{a,b+1})(λ_{b+1} − λ_b)  in J/m³.   (A, L) → (A,)"""
    return torch.trapezoid(u, lam, dim=-1)


def stefan_boltzmann(T: torch.Tensor) -> torch.Tensor:
    """The exact total, U = 4σT⁴ / c.   (A,) → (A,)"""
    return 4 * SIGMA / C * T**4


def approximate_color(T: torch.Tensor) -> torch.Tensor:
    """A rough visible colour: Planck at red, green and blue wavelengths, scaled so the brightest is 1.  (A, 3)"""
    rgb_lam = torch.tensor([610e-9, 550e-9, 465e-9], dtype=T.dtype, device=T.device)  # (3,)
    u = planck(rgb_lam, T)  # (A, 3)
    return u / u.max(dim=-1, keepdim=True).values  # (A, 3) each channel in [0, 1]
