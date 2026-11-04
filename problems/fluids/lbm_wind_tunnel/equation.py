r"""Physics → tensor translation.

Lattice Boltzmann in 3D (D3Q19). Populations f_{ixyz} carry the fluid at node (x, y, z) moving with lattice velocity
e_{ia} (i = 0..18, a ∈ {x, y, z}). The population index comes first, so each f[i] is one contiguous 3D grid.
Macroscopic fields are moments, sums over i:

    ρ = Σ_i f_i,        ρ u_a = Σ_i e_{ia} f_i

Each step relaxes f towards the local equilibrium (collision) and moves every population one link (streaming):

    f^{eq}_i = w_i ρ (1 + 3 e_{ia} u_a + 9/2 (e_{ia} u_a)² − 3/2 u_a u_a)
    f_i(x + e_i, t + 1) = f_i^*(x, t)

Collision operators, all in the same notation (Δ = f − f^eq):
  BGK:  f* = f − ω Δ                                         one rate, ν = (1/ω − ½)/3
  TRT:  f* = f − ω⁺ Δ⁺ − ω⁻ Δ⁻,  Δ^± = (Δ_i ± Δ_ī)/2         ī = opposite of i; Λ = (1/ω⁺ − ½)(1/ω⁻ − ½) = 3/16
  MRT:  f* = f − M⁻¹ S M Δ                                   M = 19 × 19 moment basis, S = diag(rates)

Smagorinsky LES raises the shear relaxation time locally from the non-equilibrium momentum flux
Π_{ab} = Σ_i e_{ia} e_{ib} Δ_i:  τ = ½ (τ₀ + √(τ₀² + 18√2 C_s² |Π| / ρ)).

Drag by momentum exchange: a population f*_i leaving a fluid node along a link into the body bounces back and hands
the body 2 f*_i e_i, so F_a = 2 Σ_{ixyz} L_{ixyz} f*_{ixyz} e_{ia}.
"""

import math

import torch


def density(f: torch.Tensor) -> torch.Tensor:
    r"""ρ = Σ_i f_i.   (19, ...) → (...)"""
    return torch.einsum("i...->...", f)


def momentum(f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""ρu_a = Σ_i e_{ia} f_i  (i summed).   (19, ...), (19, 3) → (3, ...)"""
    return torch.einsum("ia,i...->a...", e, f)


def per_direction(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-direction vector (19,) so it broadcasts against a (19, ...) field."""
    return v.reshape(-1, *[1] * (like.dim() - 1))


def equilibrium(rho: torch.Tensor, u: torch.Tensor, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""f^{eq}_i = w_i ρ (1 + 3 e_{ia}u_a + 4.5 (e_{ia}u_a)² − 1.5 u_a u_a).   (...), (3, ...) → (19, ...)"""
    eu = torch.einsum("ia,a...->i...", e, u)  # e_ia u_a, a summed
    uu = torch.einsum("a...,a...->...", u, u)  # u_a u_a
    return per_direction(w, eu) * rho * (1 - 1.5 * uu + eu * (3 + 4.5 * eu))  # same polynomial, fewer temporaries


def collide_bgk(f: torch.Tensor, f_eq: torch.Tensor, omega) -> torch.Tensor:
    """BGK: relax every population at the same rate ω = 1/τ (a number or a field broadcastable to f[0])."""
    return f - omega * (f - f_eq)


def collide_trt(f: torch.Tensor, f_eq: torch.Tensor, omega_plus, omega_minus, opp: torch.Tensor) -> torch.Tensor:
    """TRT: the even part (Δ_i + Δ_ī)/2 relaxes with ω⁺ (sets the viscosity), the odd part with ω⁻.

    omega_plus / omega_minus are numbers or fields broadcastable to f[0] (local rates for LES).
    """
    d = f - f_eq
    d_opp = d[opp]  # Δ_ī: the same node, opposite direction
    # ω⁺ (Δ + Δ_ī)/2 + ω⁻ (Δ − Δ_ī)/2, regrouped by Δ and Δ_ī
    return f - 0.5 * (omega_plus + omega_minus) * d - 0.5 * (omega_plus - omega_minus) * d_opp


def collide_mrt(f: torch.Tensor, f_eq: torch.Tensor, m: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    r"""MRT: go to moment space, relax each moment k at its own rate s_k, come back.

    f* = f − M⁻¹_{ik} s_k M_{kj} (f_j − f^eq_j)  (k, j summed). rates: (19,), or (19, ...) for local rates.
    """
    dm = torch.einsum("kj,j...->k...", m, f - f_eq)  # non-equilibrium moments
    s = per_direction(rates, dm) if rates.dim() == 1 else rates
    return f - torch.einsum("ik,k...->i...", torch.linalg.inv(m), s * dm)


def magic_omega_minus(omega_plus, magic: float = 3 / 16):
    """ω⁻ from Λ = (1/ω⁺ − ½)(1/ω⁻ − ½): Λ = 3/16 puts bounce-back walls exactly half-way between nodes."""
    return 1 / (magic / (1 / omega_plus - 0.5) + 0.5)


def smagorinsky_tau(f: torch.Tensor, f_eq: torch.Tensor, rho: torch.Tensor, e: torch.Tensor, tau0: float,
                    cs: float = 0.17) -> torch.Tensor:  # fmt: skip
    r"""Local relaxation time with a Smagorinsky eddy viscosity ν_t = (C_s Δx)² |S|.

    Π_ab = Σ_i e_ia e_ib (f_i − f^eq_i)  (i summed);  τ = ½ (τ₀ + √(τ₀² + 18√2 C_s² √(Π_ab Π_ab) / ρ)).
    """
    pi = torch.einsum("ia,ib,i...->ab...", e, e, f - f_eq)  # (3, 3, ...)
    norm = torch.sqrt(torch.einsum("ab...,ab...->...", pi, pi))  # |Π|
    return 0.5 * (tau0 + torch.sqrt(tau0**2 + 18 * math.sqrt(2) * cs**2 * norm / rho))


def stream(f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Move every population one node along its own velocity e_i (periodic wrap-around via torch.roll)."""
    return torch.stack([torch.roll(f[i], shifts=(ex, ey, ez), dims=(0, 1, 2))
                        for i, (ex, ey, ez) in enumerate(e.int().tolist())])  # fmt: skip


def bounce_back(f_streamed: torch.Tensor, f_post: torch.Tensor, incoming: tuple, opp: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back as a masked swap: where population i would have come out of the body (the entries of the
    mask incoming[i, x, y, z], given as its index tuple), put the population that left x towards the body, reversed:
    f_i(x) = f*_ī(x)."""
    i, x, y, z = incoming
    f_streamed[i, x, y, z] = f_post[opp[i], x, y, z]
    return f_streamed


def momentum_exchange_force(f_post: torch.Tensor, links: tuple, e: torch.Tensor) -> torch.Tensor:
    r"""F_a = 2 Σ_{ixyz} L_{ixyz} f*_{ixyz} e_{ia}  (i, x, y, z summed): force on the body (lattice units).   → (3,)

    links: index tuple of the boundary-link mask L, so only those few populations are gathered."""
    i, x, y, z = links
    return 2 * torch.einsum("l,la->a", f_post[i, x, y, z], e[i])


def drag_coefficient(force, u: float, area: float):
    """C = F / (½ ρ U² A) with ρ = 1."""
    return force / (0.5 * u**2 * area)


def schiller_naumann(re: torch.Tensor) -> torch.Tensor:
    """Sphere drag correlation (Schiller & Naumann 1933), a few % for Re < 800: C_D = 24/Re (1 + 0.15 Re^0.687)."""
    return 24 / re * (1 + 0.15 * re**0.687)


def viscosity_to_tau(nu: float) -> float:
    """Relaxation time from the lattice viscosity: ν = (τ − ½)/3."""
    return 3 * nu + 0.5
