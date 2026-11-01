r"""Physics → tensor translation: integrals over s-type Gaussians and the Roothaan–Hall Fock matrix.

Indices: b bond length, μ ν λ σ (m n l s) contracted basis functions, p q r t primitives, c nucleus, a Cartesian axis.

Two primitives with exponents α, β on centres A, B multiply into one Gaussian on P = (αA + βB)/(α + β):
    S_pq = (π/(α+β))^{3/2} exp(-αβ/(α+β) |A-B|²)
    T_pq = αβ/(α+β) (3 - 2αβ/(α+β) |A-B|²) S_pq
    V_pq = -Σ_c Z_c 2π/(α+β) exp(-αβ/(α+β) |A-B|²) F₀((α+β)|P - C_c|²)                       (c summed)
    (pq|rt) = 2π^{5/2} / (ζ η √(ζ+η)) e^{-…AB} e^{-…CD} F₀(ζη/(ζ+η) |P - Q|²),   ζ = α+β, η = γ+δ
and every contracted integral is a sum over primitives, e.g. S_μν = c_μp c_νq S_μνpq           (p, q summed)

    F_μν = H_μν + P_λσ [(μν|λσ) - ½ (μλ|νσ)]                                                  (λ, σ summed)
    P_λσ = 2 C_λa C_σa                                                                      (a over occupied MOs)
    E = ½ P_μν (H_μν + F_μν) + Z_A Z_B / R
"""

import math

import torch


def boys_f0(t: torch.Tensor) -> torch.Tensor:
    """F₀(t) = ∫₀¹ e^{-t u²} du = ½ √(π/t) erf(√t), with the series 1 - t/3 near t = 0."""
    ts = t.clamp_min(1e-12)
    return torch.where(t < 1e-8, 1 - t / 3, 0.5 * torch.sqrt(math.pi / ts) * torch.erf(torch.sqrt(ts)))


def gaussian_pairs(alpha: torch.Tensor, centres: torch.Tensor) -> dict:
    """Products of every primitive pair (μp, νq).   alpha (M, K), centres (B, M, 3).

    Returns zeta = α+β (M, M, K, K), mu = αβ/(α+β) (M, M, K, K), AB2 = |A-B|² (B, M, M, 1, 1),
    P (B, M, M, K, K, 3) the product centre, and K_ab = exp(-mu AB2) (B, M, M, K, K).
    """
    a = alpha[:, None, :, None]  # (M, 1, K, 1)
    b = alpha[None, :, None, :]  # (1, M, 1, K)
    zeta, mu = a + b, a * b / (a + b)
    d = centres[:, :, None, :] - centres[:, None, :, :]  # (B, M, M, 3)
    AB2 = torch.einsum("bmna,bmna->bmn", d, d)[..., None, None]  # a summed
    A = centres[:, :, None, None, None, :]  # (B, M, 1, 1, 1, 3)
    B = centres[:, None, :, None, None, :]  # (B, 1, M, 1, 1, 3)
    P = (a[..., None] * A + b[..., None] * B) / zeta[..., None]
    return {"zeta": zeta, "mu": mu, "AB2": AB2, "P": P, "K_ab": torch.exp(-mu * AB2)}


def contract2(coef: torch.Tensor, prim: torch.Tensor) -> torch.Tensor:
    """X_bμν = c_μp c_νq X_bμνpq   (p, q summed)."""
    return torch.einsum("mp,nq,bmnpq->bmn", coef, coef, prim)


def overlap(pairs: dict, coef: torch.Tensor) -> torch.Tensor:
    """S_μν, (B, M, M)."""
    prim = (math.pi / pairs["zeta"]) ** 1.5 * pairs["K_ab"]
    return contract2(coef, prim)


def kinetic(pairs: dict, coef: torch.Tensor) -> torch.Tensor:
    """T_μν = ⟨φ_μ| -½∇² |φ_ν⟩, (B, M, M)."""
    s = (math.pi / pairs["zeta"]) ** 1.5 * pairs["K_ab"]
    prim = pairs["mu"] * (3 - 2 * pairs["mu"] * pairs["AB2"]) * s
    return contract2(coef, prim)


def nuclear_attraction(pairs: dict, coef: torch.Tensor, nuclei: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """V_μν = ⟨φ_μ| -Σ_c Z_c/|r - C_c| |φ_ν⟩.   nuclei (B, A, 3), Z (A,) → (B, M, M); c summed."""
    PC = pairs["P"][..., None, :] - nuclei[:, None, None, None, None, :, :]  # (B, M, M, K, K, A, 3)
    PC2 = torch.einsum("bmnpqca,bmnpqca->bmnpqc", PC, PC)
    zeta = pairs["zeta"][..., None]
    per_nucleus = -2 * math.pi / zeta * pairs["K_ab"][..., None] * boys_f0(zeta * PC2)
    return contract2(coef, torch.einsum("c,bmnpqc->bmnpq", Z, per_nucleus))


def electron_repulsion(pairs: dict, coef: torch.Tensor) -> torch.Tensor:
    """(μν|λσ) = ∫∫ φ_μ(1)φ_ν(1) r₁₂⁻¹ φ_λ(2)φ_σ(2), the full (B, M, M, M, M) tensor; p q r t summed.

    The bra pair (μν, pq) and ket pair (λσ, rt) are the same pair tensors broadcast against each other.
    """

    def bra(x):  # (B, M, M, K, K, ...) → (B, M, M, 1, 1, K, K, 1, 1, ...)
        return x[:, :, :, None, None, :, :, None, None]

    def ket(x):  # → (B, 1, 1, M, M, 1, 1, K, K, ...)
        return x[:, None, None, :, :, None, None, :, :]

    zeta, eta = bra(pairs["zeta"][None]), ket(pairs["zeta"][None])
    PQ = bra(pairs["P"]) - ket(pairs["P"])
    PQ2 = (PQ * PQ).sum(-1)  # a summed
    pref = 2 * math.pi**2.5 / (zeta * eta * torch.sqrt(zeta + eta))
    prim = pref * bra(pairs["K_ab"]) * ket(pairs["K_ab"]) * boys_f0(zeta * eta / (zeta + eta) * PQ2)
    return torch.einsum("mp,nq,lr,st,bmnlspqrt->bmnls", coef, coef, coef, coef, prim)


def density(C: torch.Tensor, n_occ: int) -> torch.Tensor:
    """P_bλσ = 2 C_bλa C_bσa over the occupied MOs a (each holds two electrons).   (B, M, M)"""
    occ = C[..., :n_occ]
    return 2 * torch.einsum("bla,bsa->bls", occ, occ)


def fock(H: torch.Tensor, P: torch.Tensor, eri: torch.Tensor) -> torch.Tensor:
    """F_μν = H_μν + P_λσ [(μν|λσ) - ½ (μλ|νσ)]   (λ, σ summed): Coulomb J minus half the exchange K."""
    J = torch.einsum("bls,bmnls->bmn", P, eri)
    K = torch.einsum("bls,bmlns->bmn", P, eri)
    return H + J - 0.5 * K


def electronic_energy(P: torch.Tensor, H: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """E_el = ½ P_μν (H_μν + F_μν)   (μ, ν summed).   (B,)"""
    return 0.5 * torch.einsum("bmn,bmn->b", P, H + F)


def nuclear_repulsion(nuclei: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Σ_{A<B} Z_A Z_B / R_AB for each geometry.   nuclei (B, A, 3) → (B,)"""
    d = nuclei[:, :, None, :] - nuclei[:, None, :, :]
    R = torch.sqrt(torch.einsum("bija,bija->bij", d, d))
    i, j = torch.triu_indices(len(Z), len(Z), offset=1)
    return (Z[i] * Z[j] / R[:, i, j]).sum(-1)


def basis_on_points(alpha: torch.Tensor, coef: torch.Tensor, centres: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """φ_bμ(r_z) = c_μp exp(-α_μp |r_z - A_bμ|²)   (p summed).   r (Z, 3) → (B, Z, M)"""
    d = r[None, :, None, :] - centres[:, None, :, :]  # (B, Z, M, 3)
    d2 = torch.einsum("bzma,bzma->bzm", d, d)
    return torch.einsum("mp,bzmp->bzm", coef, torch.exp(-alpha[None, None] * d2[..., None]))
