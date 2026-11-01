r"""Physics → tensor translation: 1D light in a medium whose permittivity changes in time, ε = ε(t).

Maxwell's equations for E_y, H_z (units with c = 1 in vacuum, μ = 1, so B = H):

    ∂_t D = -∂_x H,     ∂_t B = -∂_x E,     D = ε(t) E

The medium is the same everywhere, so every spatial Fourier mode e^{ikx} evolves on its own (k is conserved,
the frequency is not). For the amplitudes u_{ka} = (D_k, B_k), a = 0, 1:

    \dot u_{ka} = A_{kab}(t) u_{kb},     A_k(t) = \begin{pmatrix} 0 & -ik \\ -ik/ε(t) & 0 \end{pmatrix}

(repeated b ⇒ summed). Eliminating B gives \ddot D_k + \frac{k^2}{ε(t)} D_k = 0: a parametric oscillator per k.

Sudden jump ε1 → ε2 (a temporal interface): D and B are continuous (not E and H), the wave splits into a forward
and a backward ("time-reflected") wave at the new frequency ω2 = ω1 n1/n2, with n = \sqrt{ε}.

Periodic ε(t) with period T = 2π/Ω: Floquet theory. The one-period propagator (monodromy matrix) M_k has
eigenvalues λ = e^{-iω_F T}; |λ| > 1 means Im ω_F > 0, exponential growth: a momentum band gap.
"""

import math

import torch


def generator(k: torch.Tensor, eps: float | torch.Tensor) -> torch.Tensor:
    """A_{kab} for every wavenumber: k (K,) → (K, 2, 2) complex."""
    zero = torch.zeros_like(k)
    row0 = torch.stack([zero, -1j * k], dim=-1)  # dD/dt = -ik B
    row1 = torch.stack([-1j * k / eps, zero], dim=-1)  # dB/dt = -ik E = -ik D / ε
    return torch.stack([row0, row1], dim=-2)


def rhs(u: torch.Tensor, k: torch.Tensor, eps: float | torch.Tensor) -> torch.Tensor:
    r"""\dot u_{k...a} = A_{kab} u_{k...b}  (repeated b ⇒ summed). u: (K, 2) or (K, 2, 2) for the monodromy."""
    A = generator(k, eps)
    return torch.einsum("kab,kb...->ka...", A, u)


def periodic_eps(t: float, eps0: float, alpha: float, omega: float) -> float:
    """ε(t) = ε0 (1 + α cos Ωt)."""
    return eps0 * (1.0 + alpha * math.cos(omega * t))


def temporal_fresnel(n1: float, n2: float) -> tuple[float, float]:
    r"""Amplitudes of the forward and backward D waves after a jump n1 → n2, for an incident forward wave D = 1.

    A forward wave has B = D/n, a backward wave B = -D/n. Continuity of D and of B at the jump:
        D_f + D_b = 1,     (D_f - D_b)/n_2 = 1/n_1
    ⇒  D_f = (1 + n_2/n_1)/2,    D_b = (1 - n_2/n_1)/2.
    """
    return 0.5 * (1 + n2 / n1), 0.5 * (1 - n2 / n1)


def split_directions(D: torch.Tensor, B: torch.Tensor, n: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward part (B = D/n) and backward part (B = -D/n) of a field in a medium of index n."""
    return 0.5 * (D + n * B), 0.5 * (D - n * B)


def floquet_frequencies(eigenvalues: torch.Tensor, period: float) -> torch.Tensor:
    """λ = e^{-iω T}  ⇒  ω = i ln(λ) / T   (complex; Re ω folded into (-Ω/2, Ω/2], Im ω = growth rate)."""
    return 1j * torch.log(eigenvalues) / period


def perturbative_gap(eps0: float, alpha: float, omega: float) -> tuple[float, float]:
    r"""First momentum gap to first order in α: parametric resonance ω_k = k/n_0 ≈ Ω/2.

    \ddot D + ω_k^2 (1 - α\cos Ωt) D ≈ 0 is unstable for |Ω - 2ω_k| < α ω_k / 2, i.e. k ∈ n_0 Ω/2 (1 ∓ α/4).
    """
    k0 = eps0**0.5 * omega / 2
    return k0 * (1 - alpha / 4), k0 * (1 + alpha / 4)
