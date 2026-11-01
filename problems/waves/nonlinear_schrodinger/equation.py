r"""Physics → tensor translation.

Focusing nonlinear Schrödinger equation:   i ψ_t + ½ ψ_xx + |ψ|² ψ = 0.

Split it into two parts that can each be solved exactly over a step Δt:

    dispersion   i ψ_t = −½ ψ_xx   →   \hat ψ_k ← e^{−i k² Δt/2} \hat ψ_k              (in Fourier space)
    nonlinearity i ψ_t = −|ψ|² ψ   →   ψ_j ← e^{i ψ_j^* ψ_j Δt} ψ_j                     (in real space, |ψ| fixed)

Strang splitting does half a dispersion step, a full nonlinear step, then another half dispersion step.
In index form the dispersion half step is ψ_j ← F^{-1}_{jk} e^{−ik²Δt/4} F_{kl} ψ_l   (k, l summed).

Conserved: N = ∫ |ψ|² dx and H = ∫ (½ |ψ_x|² − ½ |ψ|⁴) dx.
"""

import cmath

import torch


def split_step(psi: torch.Tensor, half_linear: torch.Tensor, dt: float) -> torch.Tensor:
    """One Strang step.   psi: (J,) complex, half_linear = e^{−ik²Δt/4}: (K,) → (J,)"""
    psi = torch.fft.ifft(half_linear * torch.fft.fft(psi))  # half a step of dispersion
    psi = psi * torch.exp(1j * psi.abs() ** 2 * dt)  # a full step of nonlinearity
    return torch.fft.ifft(half_linear * torch.fft.fft(psi))  # the other half of dispersion


def norm(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """N = Σ_j ψ_j^* ψ_j Δx.   (…, J) → (…)"""
    return (psi.abs() ** 2).sum(dim=-1) * dx


def hamiltonian(psi: torch.Tensor, k: torch.Tensor, dx: float) -> torch.Tensor:
    """H = Σ_j (½ |ψ_x|² − ½ |ψ|⁴) Δx with the spectral derivative ψ_x = F^{-1}[ik F ψ].   (…, J) → (…)"""
    psi_x = torch.fft.ifft(1j * k * torch.fft.fft(psi, dim=-1), dim=-1)
    return (0.5 * psi_x.abs() ** 2 - 0.5 * psi.abs() ** 4).sum(dim=-1) * dx


def peregrine(x: torch.Tensor, t: float) -> torch.Tensor:
    """Exact Peregrine breather ψ = [1 − 4(1 + 2it) / (1 + 4x² + 4t²)] e^{it}, peak |ψ| = 3 at x = t = 0."""
    return (1 - 4 * (1 + 2j * t) / (1 + 4 * x**2 + 4 * t**2)) * cmath.exp(1j * t)


def bright_soliton(x: torch.Tensor, t: float, eta: float, v: float) -> torch.Tensor:
    """Exact soliton ψ = η sech(η(x − vt)) e^{i(vx + (η² − v²) t / 2)}."""
    phase = v * x + 0.5 * (eta**2 - v**2) * t
    return eta / torch.cosh(eta * (x - v * t)) * torch.exp(1j * phase)
