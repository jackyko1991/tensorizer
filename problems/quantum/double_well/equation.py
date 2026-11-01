r"""Physics → tensor translation.

Time-independent Schrödinger equation in a symmetric double well (units ħ = m = 1):

    \hat H \varphi = -\tfrac12 \varphi'' + V(x) \varphi = E \varphi,      V(x) = V_0 (x^2/a^2 - 1)^2.

On a grid, with the eigenvectors as the columns of \varphi_{xn}:

    H_{xy} = -\tfrac12 D2_{xy} + V_x \delta_{xy},      H_{xy} \varphi_{yn} = E_n \varphi_{xn}      (y summed)

Start in c = (1, 1, 0, …)/\sqrt2, i.e. \psi(0) = (\varphi_0 + \varphi_1)/\sqrt2, and evolve exactly:

    \psi_x(t) = \varphi_{xn} c_n e^{-i E_n t}                   (n summed)

The probability in the left well uses the step function \Theta_x = 1 for x < 0:

    P_L(t) = \psi^*_x \Theta_x \psi_x \, dx                       (x summed)

For two states this gives P_L \approx \tfrac12 (1 + \cos \Delta E t) = \cos^2(\Delta E t / 2), \Delta E = E_1 - E_0.
"""

import torch


def evolve(phi: torch.Tensor, energies: torch.Tensor, c: torch.Tensor, t: float) -> torch.Tensor:
    r"""\psi_x(t) = \varphi_{xn} c_n e^{-i E_n t}.   phi: (N, M), energies: (M,), c: (M,) → (N,) complex"""
    phase = torch.exp(-1j * energies * t)  # (M,)  e^{-i E_n t}
    return torch.einsum("xn,n->x", phi.to(c.dtype), c * phase)


def left_probability(psi: torch.Tensor, x: torch.Tensor, dx: float) -> torch.Tensor:
    r"""P_L = \psi^*_x \Theta_x \psi_x dx  (x summed).   psi: (..., N), x: (N,) → (...)"""
    theta = (x < 0).to(psi.dtype)  # (N,) step function Θ_x
    return torch.einsum("...x,x,...x->...", psi.conj(), theta, psi).real * dx


def two_state_prediction(times: torch.Tensor, delta_E: float) -> torch.Tensor:
    r"""P_L(t) = \cos^2(\Delta E\, t / 2) for an ideal two-level system.   (T,) → (T,)"""
    return torch.cos(delta_E * times / 2) ** 2
