r"""Physics → tensor translation.

A spin-½ in a magnetic field (units ħ = 1):

    i \partial_t |\psi\rangle = \hat H |\psi\rangle,      \hat H = -\tfrac{\gamma}{2} \mathbf B \cdot \boldsymbol\sigma.

Stack the three Pauli matrices into one tensor \sigma_{kab} (k: x, y, z; a, b: spinor indices).
Then the dot product B·σ is a contraction over k:

    H_{ab} = -\tfrac{\gamma}{2} B_k \sigma_{kab}                     (k summed)

Over one short step the field is nearly constant, so

    \psi_a(t + \Delta t) = U_{ab} \psi_b(t),   U = e^{-i H \Delta t}     (b summed)

and the Bloch vector, the expectation of each Pauli matrix, sums over both spinor indices:

    s_k = \psi^*_a \sigma_{kab} \psi_b                                (a, b summed)

Its equation of motion is d\mathbf s/dt = \gamma\, \mathbf s \times \mathbf B: precession at \omega = \gamma B.
"""

import torch


def hamiltonian(B: torch.Tensor, sigma: torch.Tensor, gamma: float) -> torch.Tensor:
    r"""H_{tab} = -(\gamma/2) B_{tk} \sigma_{kab}.   B: (T, 3), sigma: (3, 2, 2) → (T, 2, 2)"""
    return -0.5 * gamma * torch.einsum("tk,kab->tab", B.to(sigma.dtype), sigma)


def propagator(H: torch.Tensor, dt: float) -> torch.Tensor:
    r"""U = e^{-i H \Delta t}, one matrix exponential per time step.   (T, 2, 2) → (T, 2, 2)"""
    return torch.linalg.matrix_exp(-1j * H * dt)


def step(U: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""\psi_a \leftarrow U_{ab} \psi_b.   U: (2, 2), psi: (2,) → (2,)"""
    return torch.einsum("ab,b->a", U, psi)


def bloch_vector(psi: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""s_k = \psi^*_a \sigma_{kab} \psi_b.   psi: (2,) → (3,) real"""
    return torch.einsum("a,kab,b->k", psi.conj(), sigma, psi).real
