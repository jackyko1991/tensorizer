r"""Physics → tensor translation.

Coupled-mode theory of a honeycomb lattice of identical resonators (or waveguides). The field amplitude
ψ_i in resonator i obeys a Schrödinger-like equation, with time (or propagation distance z) and
the frequency detuning playing the roles of t and E:

    i \dot ψ_i = H_{ij} ψ_j                                             (j summed)

H has three kinds of entries (the Haldane model):
    nearest-neighbour coupling        t_1
    next-nearest-neighbour coupling   t_2 e^{-i ν_{ij} φ},  ν_{ij} = ±1 for a left/right turn
    on-site detuning                  ±M on sublattice A/B

The complex t_2 breaks time-reversal symmetry (in photonics: a magneto-optical material or a
modulation). On the infinite lattice a plane wave turns H into a 2×2 Bloch Hamiltonian

    H_{ab}(k) = d_μ(k) σ^μ_{ab}                    (μ = 0…3 summed, σ^0 = identity)
    d_0 = 2 t_2 cos φ Σ_n cos(k·b_n),   d_x = t_1 Σ_n cos(k·δ_n),   d_y = -t_1 Σ_n sin(k·δ_n)
    d_z = M - 2 t_2 sin φ Σ_n sin(k·b_n)

with δ_n the three A → B bonds counted from unit cell to unit cell (δ_n - δ_1), so that H(k) is periodic in k.

The lower band's Chern number C = (1/2π) ∫ Ω d²k is an integer. Fukui–Hatsugai–Suzuki compute it on a
k-grid from the link variables U_μ(k) = ⟨u(k)|u(k + Δk_μ)⟩/|…| (the index a is summed):

    F(k) = arg[ U_1(k) U_2(k + Δk_1) U_1(k + Δk_2)^* U_2(k)^* ],      C = (1/2π) Σ_k F(k)

F/ΔA is the Berry curvature Ω on each plaquette; the gauge-invariant loop product makes C an exact integer
even on a coarse grid. The continuum formula Ω = -½ ε_{abc} d̂_a ∂_x d̂_b ∂_y d̂_c gives the same map.
"""

import math

import torch

# Pauli matrices σ^μ_{ab}, μ = 0 (identity), x, y, z: shape (4, 2, 2)
SIGMA = torch.tensor(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=torch.complex128
)
# Levi-Civita ε_{abc}: shape (3, 3, 3)
EPS = torch.zeros(3, 3, 3, dtype=torch.float64)
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0


def d_vector(
    k: torch.Tensor, delta: torch.Tensor, b: torch.Tensor, t1: float, t2: float, phi: float, M: float
) -> torch.Tensor:
    r"""d_μ(k) for μ = 0, x, y, z.   k: (..., 2), δ_{ni}, b_{ni}: (3, 2) → (..., 4)

    k·δ_n = k_i δ_{ni} is einsum('...i,ni->...n'), then each sum over the three bonds n is .sum(-1).
    """
    kd = torch.einsum("...i,ni->...n", k, delta)  # (..., 3)
    kb = torch.einsum("...i,ni->...n", k, b)  # (..., 3)
    d0 = 2 * t2 * math.cos(phi) * torch.cos(kb).sum(-1)
    dx = t1 * torch.cos(kd).sum(-1)
    dy = -t1 * torch.sin(kd).sum(-1)
    dz = M - 2 * t2 * math.sin(phi) * torch.sin(kb).sum(-1)
    return torch.stack([d0, dx, dy, dz], dim=-1)


def bloch_hamiltonian(d: torch.Tensor) -> torch.Tensor:
    r"""H_{ab}(k) = d_μ(k) σ^μ_{ab}   (μ summed).   d: (..., 4) → (..., 2, 2) complex"""
    return torch.einsum("...m,mab->...ab", d.to(torch.complex128), SIGMA.to(d.device))


def link_variables(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""U_μ(k) = ⟨u(k)|u(k+Δk_μ)⟩ / |…|   (band index a summed; periodic grid).   u: (N1, N2, 2) → 2 × (N1, N2)"""
    U1 = torch.einsum("xya,xya->xy", u.conj(), torch.roll(u, -1, dims=0))
    U2 = torch.einsum("xya,xya->xy", u.conj(), torch.roll(u, -1, dims=1))
    return U1 / U1.abs(), U2 / U2.abs()


def plaquette_flux(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    r"""F(k) = arg[U_1(k) U_2(k+1̂) U_1(k+2̂)^* U_2(k)^*] ∈ (-π, π]: Berry flux through each plaquette.   → (N1, N2)"""
    loop = U1 * torch.roll(U2, -1, dims=0) * torch.roll(U1, -1, dims=1).conj() * U2.conj()
    return torch.angle(loop)


def berry_curvature_formula(dhat: torch.Tensor, ddx: torch.Tensor, ddy: torch.Tensor) -> torch.Tensor:
    r"""Ω(k) = -½ ε_{abc} d̂_a ∂_x d̂_b ∂_y d̂_c  for the lower band (a, b, c summed): minus half the solid angle of d̂.

    The sign matches the FHS fluxes above, i.e. the Berry connection A_μ = Im⟨u|∂_μ u⟩.

    dhat, ddx, ddy: (..., 3) unit vector (d_x, d_y, d_z)/|d| and its k_x, k_y derivatives → (...)
    """
    return -0.5 * torch.einsum("abc,...a,...b,...c->...", EPS.to(dhat.device), dhat, ddx, ddy)


def hoppings(
    r: torch.Tensor, r_other: torch.Tensor, xi: torch.Tensor, t1: float, t2: float, phi: float
) -> torch.Tensor:
    r"""Real-space couplings H_{ij} from site i at r_i to site j at r'_j (nearest-neighbour distance 1).

    t_1 at distance 1; t_2 e^{-i ν_{ij} φ} at distance √3, with ν_{ij} = ξ_i cos 3θ_{ij} = ±1, where θ_{ij} is the
    direction of r'_j - r_i and ξ_i = ±1 the sublattice: the same left/right-turn sign as in the Bloch d_0 and d_z.
    r: (N, 2), r_other: (N', 2), xi: (N,) → (N, N') complex
    """
    d = r_other[None, :, :] - r[:, None, :]  # (N, N', 2)
    dist = d.norm(dim=-1)
    nn = (dist - 1).abs() < 1e-6
    nnn = (dist - math.sqrt(3)).abs() < 1e-6
    nu = xi[:, None] * torch.cos(3 * torch.atan2(d[..., 1], d[..., 0])).round()  # (N, N') ±1
    H = torch.where(nn, torch.tensor(t1 + 0j, dtype=torch.complex128), torch.tensor(0j, dtype=torch.complex128))
    return torch.where(nnn, t2 * torch.exp(-1j * nu * phi), H)


def evolve(E: torch.Tensor, V: torch.Tensor, psi0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    r"""ψ(t) = e^{-iHt} ψ_0 = V_{in} e^{-i E_n t} V^*_{jn} ψ_{0j}   (n, j summed), for a batch of lattices s.

    E: (S, N), V: (S, N, N), psi0: (N,), times: (T,) → (S, T, N)
    """
    c = torch.einsum("sjn,j->sn", V.conj(), psi0)  # (S, N) mode amplitudes
    phase = torch.exp(-1j * E[:, None, :] * times[None, :, None])  # (S, T, N)
    return torch.einsum("sin,stn,sn->sti", V, phase, c)
