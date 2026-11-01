r"""Physics → tensor translation.

Lattice Boltzmann (D2Q9, BGK). Instead of solving Navier–Stokes for u directly, we track
populations f_{xyi}: the amount of fluid at node (x, y) moving with lattice velocity e_{ia}
(i = 0..8, a ∈ {x, y}). The macroscopic fields are moments (sums over i):

    ρ_{xy} = Σ_i f_{xyi},        ρ u_{xya} = Σ_i f_{xyi} e_{ia}

Each step relaxes f towards a local equilibrium (collision) and moves it one link (streaming):

    f_{xyi}^{eq} = w_i ρ (1 + 3 e_{ia} u_a + 9/2 (e_{ia} u_a)^2 - 3/2 u_a u_a)
    f_i(\mathbf x + \mathbf e_i, t + 1) = f_i - (f_i - f_i^{eq}) / τ,        ν = (τ - ½) / 3

which reproduces the Navier–Stokes equations at low Mach number.

Drag by momentum exchange: every population that hits the cylinder along a boundary link bounces
back, handing the body twice its momentum: F_a = 2 Σ_{xyi} L_{xyi} f_{xyi} e_{ia}.
"""

import torch


def density(f: torch.Tensor) -> torch.Tensor:
    r"""ρ_{xy} = Σ_i f_{xyi}.   (Nx, Ny, 9) → (Nx, Ny)"""
    return torch.einsum("xyi->xy", f)


def momentum(f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""ρu_{xya} = Σ_i f_{xyi} e_{ia}  (i summed).   (Nx, Ny, 9), (9, 2) → (Nx, Ny, 2)"""
    return torch.einsum("xyi,ia->xya", f, e)


def equilibrium(rho: torch.Tensor, u: torch.Tensor, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""f^{eq}_{xyi} = w_i ρ (1 + 3 e_{ia}u_a + 4.5 (e_{ia}u_a)^2 - 1.5 u_a u_a).   → (Nx, Ny, 9)"""
    eu = torch.einsum("ia,xya->xyi", e, u)  # e_ia u_a, a summed
    uu = torch.einsum("xya,xya->xy", u, u)  # u_a u_a
    return torch.einsum("i,xy,xyi->xyi", w, rho, 1 + 3 * eu + 4.5 * eu**2 - 1.5 * uu[..., None])


def collide(f: torch.Tensor, f_eq: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK collision: relax towards equilibrium with relaxation time τ."""
    return f - (f - f_eq) / tau


def stream(f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Move every population one node along its own velocity e_i (periodic wrap-around via torch.roll)."""
    return torch.stack([torch.roll(f[..., i], shifts=(int(ex), int(ey)), dims=(0, 1))
                        for i, (ex, ey) in enumerate(e.tolist())], dim=-1)  # fmt: skip


def momentum_exchange_force(f: torch.Tensor, links: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""F_a = 2 Σ_{xyi} L_{xyi} f_{xyi} e_{ia}  (x, y, i summed): force on the body in lattice units.   → (2,)"""
    return 2 * torch.einsum("xyi,xyi,ia->a", links, f, e)


def viscosity_to_tau(nu: float) -> float:
    """Relaxation time from the lattice viscosity: ν = (τ - ½)/3."""
    return 3 * nu + 0.5
