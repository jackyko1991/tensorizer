"""Integrals for every bond length, the batched SCF loop, a minimal-basis full CI and the separated atoms."""

import math

import torch

from problems._common.progress import steps

from . import equation as eq


def integrals(alpha: torch.Tensor, coef: torch.Tensor, nuclei: torch.Tensor, Z: torch.Tensor) -> dict:
    """S, T, V, H = T + V (B, M, M) and the two-electron tensor eri (B, M, M, M, M) for every geometry."""
    pairs = eq.gaussian_pairs(alpha, nuclei)  # the basis functions sit on the nuclei
    S = eq.overlap(pairs, coef)
    T = eq.kinetic(pairs, coef)
    V = eq.nuclear_attraction(pairs, coef, nuclei, Z)
    return {"S": S, "T": T, "V": V, "H": T + V, "eri": eq.electron_repulsion(pairs, coef)}


def symmetric_orthogonaliser(S: torch.Tensor) -> torch.Tensor:
    """X = S^{-1/2} = U s^{-1/2} Uᵀ, so that Xᵀ S X = 1.   (B, M, M)"""
    s, U = torch.linalg.eigh(S)
    return U @ torch.diag_embed(s.rsqrt()) @ U.transpose(-1, -2)


def scf(ints: dict, n_electrons: int, tol: float = 1e-10, max_iter: int = 100) -> dict:
    """Roothaan–Hall SCF for every bond length at once, starting from P = 0 (the core-Hamiltonian guess).

    Each pass: F(P) → F' = Xᵀ F X → eigh → C = X C' → new P. Stops when max |ΔP| < tol for all geometries.
    """
    H, eri = ints["H"], ints["eri"]
    X = symmetric_orthogonaliser(ints["S"])
    P = torch.zeros_like(H)
    n_occ = n_electrons // 2
    history = []
    for it in steps(max_iter, "SCF iterations"):
        F = eq.fock(H, P, eri)
        eps, C_prime = torch.linalg.eigh(X.transpose(-1, -2) @ F @ X)  # orbital energies ascending
        C = X @ C_prime  # (B, M, M): column a is MO a in the basis φ_μ
        P_new = eq.density(C, n_occ)
        change = (P_new - P).abs().amax().item()
        P = P_new
        history.append(change)
        n_iter = it + 1
        if change < tol:
            break
    F = eq.fock(H, P, eri)  # Fock matrix of the converged density
    C = _fix_sign(C)
    return {
        "P": P,
        "F": F,
        "C": C,
        "eps": eps,
        "E_el": eq.electronic_energy(P, H, F),
        "n_iter": n_iter,
        "converged": change < tol,
        "history": history,
    }


def _fix_sign(C: torch.Tensor) -> torch.Tensor:
    """An eigenvector's overall sign is arbitrary: make the coefficient on atom 1 positive, for smooth plots."""
    sign = torch.sign(C[:, 1:2, :])
    return C * torch.where(sign == 0, torch.ones_like(sign), sign)


def full_ci(ints: dict, C: torch.Tensor) -> torch.Tensor:
    """Exact (full CI) ground state of 2 electrons in the 2 MOs: lowest eigenvalue of a 3 × 3 singlet matrix.

    Configurations |1²⟩ (RHF), |1¹2¹⟩ singlet, |2²⟩. MO integrals h_ij = C_μi H_μν C_νj and
    (ij|kl) = C_μi C_νj C_λk C_σl (μν|λσ) (μ ν λ σ summed). Returns electronic energies (B,).
    """
    h = torch.einsum("bmi,bmn,bnj->bij", C, ints["H"], C)
    g = torch.einsum("bmi,bnj,bpk,bql,bmnpq->bijkl", C, C, C, C, ints["eri"])
    r2 = math.sqrt(2.0)
    H00 = 2 * h[:, 0, 0] + g[:, 0, 0, 0, 0]
    H11 = h[:, 0, 0] + h[:, 1, 1] + g[:, 0, 0, 1, 1] + g[:, 0, 1, 1, 0]
    H22 = 2 * h[:, 1, 1] + g[:, 1, 1, 1, 1]
    H01 = r2 * (h[:, 0, 1] + g[:, 0, 0, 0, 1])  # = √2 F₁₂ = 0 at convergence (Brillouin's theorem)
    H12 = r2 * (h[:, 0, 1] + g[:, 0, 1, 1, 1])
    H02 = g[:, 0, 1, 0, 1]  # exchange integral K₁₂ couples |1²⟩ and |2²⟩
    M = torch.stack(
        [torch.stack([H00, H01, H02], -1), torch.stack([H01, H11, H12], -1), torch.stack([H02, H12, H22], -1)], -2
    )
    return torch.linalg.eigvalsh(M)[:, 0]


def separated_atoms(alpha: torch.Tensor, coef: torch.Tensor, Z: torch.Tensor, n_atom: tuple) -> float:
    """Energy of the atoms far apart, each with its own 1s function: one electron gives h, two give 2h + (11|11)."""
    total = 0.0
    for k, n in enumerate(n_atom):
        nucleus = torch.zeros(1, 1, 3, dtype=alpha.dtype, device=alpha.device)
        ints = integrals(alpha[k : k + 1], coef[k : k + 1], nucleus, Z[k : k + 1])
        h, J = ints["H"][0, 0, 0].item(), ints["eri"][0, 0, 0, 0, 0].item()
        total += n * h + (J if n == 2 else 0.0)
    return total


def equilibrium(R: torch.Tensor, E: torch.Tensor) -> tuple[float, float]:
    """Bond length and energy at the minimum: a parabola through the lowest grid point and its neighbours."""
    k = int(torch.argmin(E).clamp(1, len(E) - 2))
    x, y = R[k - 1 : k + 2], E[k - 1 : k + 2]
    A = torch.stack([x**2, x, torch.ones_like(x)], -1)
    a, b, c = torch.linalg.solve(A, y).tolist()
    r0 = -b / (2 * a)
    return r0, a * r0**2 + b * r0 + c
