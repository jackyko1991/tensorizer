r"""Physics → tensor translation.

A two-qubit state |\psi\rangle = \psi_{ab} |a\rangle |b\rangle is a 2×2 tensor (a: qubit 1, b: qubit 2).

Gates. A two-qubit gate is a rank-4 tensor G_{abcd} (out: a b, in: c d); a single-qubit gate A on qubit 1
is the tensor product (A \otimes I)_{abcd} = A_{ac} \delta_{bd}. Applying a gate sums over the inputs:

    \psi_{ab} \leftarrow G_{abcd} \psi_{cd}                        (c, d summed)

Schmidt decomposition = SVD of the matrix \psi_{ab}:

    \psi_{ab} = U_{ak} s_k V^*_{bk},      S = -s_k^2 \ln s_k^2      (k summed)

Reduced density matrices (partial traces) and Bloch vectors:

    \rho^{(1)}_{ac} = \psi_{ab} \psi^*_{cb}     (b summed: trace out qubit 2)
    \rho^{(2)}_{bd} = \psi_{ab} \psi^*_{ad}     (a summed: trace out qubit 1)
    r_k = \mathrm{Tr}(\rho \sigma_k) = \sigma_{kca} \rho_{ac}

Time evolution with H_{abcd} = J Z_{ac} Z_{bd} + h (X_{ac} \delta_{bd} + \delta_{ac} X_{bd}): reshaped to a 4×4
matrix, H = W \mathrm{diag}(E) W^\dagger and U_{ij}(t) = W_{ik} e^{-i E_k t} W^*_{jk}.
"""

import torch


def apply_gate(G: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""\psi_{ab} \leftarrow G_{abcd} \psi_{cd}.   G: (2, 2, 2, 2), psi: (2, 2) → (2, 2)"""
    return torch.einsum("abcd,cd->ab", G, psi)


def schmidt(psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""\psi_{ab} = U_{ak} s_k V^*_{bk}.   psi: (..., 2, 2) → U (..., 2, 2), s (..., 2), Vh (..., 2, 2)"""
    return torch.linalg.svd(psi)


def entanglement_entropy(psi: torch.Tensor) -> torch.Tensor:
    r"""S = -s_k^2 \ln s_k^2 (k summed), in nats.   psi: (..., 2, 2) → (...)"""
    _, s, _ = schmidt(psi)
    p = s**2  # Schmidt probabilities, sum to 1
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def concurrence(psi: torch.Tensor) -> torch.Tensor:
    r"""C = 2 |\psi_{00}\psi_{11} - \psi_{01}\psi_{10}| = 2|\det\psi|.   (..., 2, 2) → (...)"""
    return 2 * torch.linalg.det(psi).abs()


def reduced_density_1(psi: torch.Tensor) -> torch.Tensor:
    r"""\rho^{(1)}_{ac} = \psi_{ab} \psi^*_{cb}  (trace out qubit 2).   (..., 2, 2) → (..., 2, 2)"""
    return torch.einsum("...ab,...cb->...ac", psi, psi.conj())


def reduced_density_2(psi: torch.Tensor) -> torch.Tensor:
    r"""\rho^{(2)}_{bd} = \psi_{ab} \psi^*_{ad}  (trace out qubit 1).   (..., 2, 2) → (..., 2, 2)"""
    return torch.einsum("...ab,...ad->...bd", psi, psi.conj())


def bloch_vector(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""r_k = \sigma_{kca} \rho_{ac} = Tr(\rho \sigma_k).   rho: (..., 2, 2), sigma: (3, 2, 2) → (..., 3)"""
    return torch.einsum("kca,...ac->...k", sigma, rho).real


def propagators(H: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    r"""U_{ij}(t) = W_{ik} e^{-i E_k t} W^*_{jk}.   H: (2, 2, 2, 2), times: (T,) → (T, 2, 2, 2, 2)"""
    E, W = torch.linalg.eigh(H.reshape(4, 4))  # pair (a, b) → row 2a + b
    phases = torch.exp(-1j * torch.einsum("t,k->tk", times.to(E.dtype), E))  # (T, 4)
    U = torch.einsum("ik,tk,jk->tij", W, phases, W.conj())  # (T, 4, 4)
    return U.reshape(-1, 2, 2, 2, 2)


def evolve(U: torch.Tensor, psi0: torch.Tensor) -> torch.Tensor:
    r"""\psi_{tab} = U_{tabcd} \psi_{cd}(0).   U: (T, 2, 2, 2, 2), psi0: (2, 2) → (T, 2, 2)"""
    return torch.einsum("tabcd,cd->tab", U, psi0)
