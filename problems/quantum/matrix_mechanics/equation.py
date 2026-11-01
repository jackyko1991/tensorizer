r"""Physics → tensor translation.

Heisenberg's matrix mechanics for the oscillator (units ħ = m = ω = 1) in the number basis |0⟩ … |N−1⟩:

    a_{mn} = \sqrt{n}\,\delta_{m,n-1}                       (lowering operator)
    X = (a + a^\dagger)/\sqrt2,     P = i(a^\dagger - a)/\sqrt2
    H_{mn} = \tfrac12 (P_{mk}P_{kn} + X_{mk}X_{kn}) + \lambda X^4_{mn}      (k summed)
    [X, P]_{mn} = X_{mk}P_{kn} - P_{mk}X_{kn} = i\,\delta_{mn}  (except the last entry: truncation)

Time evolution, U = e^{-iHt}, in two equivalent pictures:

    Heisenberg:   X(t)_{mn} = U^*_{km} X_{kl} U_{ln},     ⟨X⟩ = c^*_m X(t)_{mn} c_n       (k, l, m, n summed)
    Schrödinger:  c_m(t) = U_{mn} c_n,                     ⟨X⟩ = c^*_m(t) X_{mn} c_n(t)
"""

import torch


def dagger(A: torch.Tensor) -> torch.Tensor:
    r"""(A^\dagger)_{mn} = A^*_{nm}.   (..., N, N) → (..., N, N)"""
    return A.conj().transpose(-2, -1)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""(AB)_{mn} = A_{mk} B_{kn}  (k summed).   (N, N), (N, N) → (N, N)"""
    return torch.einsum("mk,kn->mn", A, B)


def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""[A, B] = AB - BA.   (N, N) → (N, N)"""
    return matmul(A, B) - matmul(B, A)


def hamiltonian(X: torch.Tensor, P: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    r"""H = \tfrac12(P^2 + X^2) + \lambda X^4.   (N, N) → (N, N) Hermitian"""
    X2 = matmul(X, X)
    return 0.5 * (matmul(P, P) + X2) + lam * matmul(X2, X2)


def number_hamiltonian(a: torch.Tensor) -> torch.Tensor:
    r"""The exact oscillator H = a^\dagger a + \tfrac12 = diag(n + ½).   (N, N) → (N, N)"""
    eye = torch.eye(len(a), dtype=a.dtype, device=a.device)
    return matmul(dagger(a), a) + 0.5 * eye


def heisenberg(U: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    r"""O(t)_{mn} = U^*_{km} O_{kl} U_{ln}  (k, l summed).   U: (T, N, N), op: (N, N) → (T, N, N)"""
    return torch.einsum("tkm,kl,tln->tmn", U.conj(), op, U)


def schrodinger(U: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r"""c_m(t) = U_{mn} c_n  (n summed).   U: (T, N, N), c: (N,) → (T, N)"""
    return torch.einsum("tmn,n->tm", U, c)


def expectation(c: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    r"""⟨O⟩ = c^*_m O_{mn} c_n  (m, n summed).   c: (..., N), op: (..., N, N) → (...) real"""
    return torch.einsum("...m,...mn,...n->...", c.conj(), op, c).real
