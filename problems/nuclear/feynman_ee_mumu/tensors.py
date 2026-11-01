"""Build the concrete tensors: Dirac matrices, the metric, lepton masses and the four momenta of each event."""

import math

import torch

ALPHA = 1 / 137.035999  # fine-structure constant, e² = 4πα
M_E = 0.000511  # electron mass, GeV

# final state: (label, mass in GeV, symbol)
FINAL = {
    "muon": ("e⁺e⁻ → μ⁺μ⁻", 0.105658, "μ"),
    "tau": ("e⁺e⁻ → τ⁺τ⁻", 1.77686, "τ"),
}


def metric(device: torch.device) -> torch.Tensor:
    """η_{μν} = diag(1, −1, −1, −1): shape (4, 4)."""
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.complex128, device=device))


def gamma_matrices(device: torch.device) -> torch.Tensor:
    """Dirac representation γ^μ_{ab}: γ^0 = diag(1, 1, −1, −1), γ^i = [[0, σ_i], [−σ_i, 0]]: shape (4, 4, 4)."""
    sigma = torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=torch.complex128, device=device
    )  # (3, 2, 2) Pauli matrices
    eye = torch.eye(2, dtype=torch.complex128, device=device)
    zero = torch.zeros(2, 2, dtype=torch.complex128, device=device)
    g = torch.zeros(4, 4, 4, dtype=torch.complex128, device=device)
    g[0] = torch.block_diag(eye, -eye)
    for i in range(3):
        g[i + 1] = torch.cat([torch.cat([zero, sigma[i]], 1), torch.cat([-sigma[i], zero], 1)], 0)
    return g


def angles(n: int, device: torch.device) -> torch.Tensor:
    """Scattering angles θ of the outgoing lepton relative to the incoming electron, 0 … π: shape (B,)."""
    return torch.linspace(0.0, math.pi, n, dtype=torch.float64, device=device)


def energies(threshold: float, sqrt_s_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Centre-of-mass energies √s, log-spaced from just above threshold to sqrt_s_max: shape (E,)."""
    lo = math.log(threshold * 1.002)
    return torch.logspace(lo / math.log(10), math.log10(sqrt_s_max), n, dtype=torch.float64, device=device)


def momenta(sqrt_s: torch.Tensor, theta: torch.Tensor, m_f: float) -> tuple[torch.Tensor, ...]:
    """Centre-of-mass four-momenta p (e⁻), p' (e⁺), k (ℓ⁻), k' (ℓ⁺): four tensors of shape (E, B, 4).

    Beams along ±z; the outgoing pair back to back in the x–z plane at angle θ to the electron.
    """
    e = (sqrt_s / 2)[:, None].expand(-1, theta.shape[0])  # (E, B) beam energy
    p_in = (e**2 - M_E**2).sqrt()
    p_out = (e**2 - m_f**2).clamp(min=0).sqrt()
    zero = torch.zeros_like(e)
    s, c = torch.sin(theta).expand_as(e), torch.cos(theta).expand_as(e)
    p = torch.stack([e, zero, zero, p_in], -1)
    pb = torch.stack([e, zero, zero, -p_in], -1)
    k = torch.stack([e, p_out * s, zero, p_out * c], -1)
    kb = torch.stack([e, -p_out * s, zero, -p_out * c], -1)
    return p, pb, k, kb
