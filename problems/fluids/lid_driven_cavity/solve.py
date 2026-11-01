"""Time-march vorticity, solving the Poisson equation for ψ at every step."""

import numpy as np
import torch

from problems._common.progress import steps

from .equation import vorticity_rhs, wall_vorticity

# Ghia, Ghia & Shin (1982), Re = 100: u along the vertical centreline x = 0.5 (y, u)
GHIA_RE100 = torch.tensor(
    [[1.0000, 1.00000], [0.9766, 0.84123], [0.9688, 0.78871], [0.9609, 0.73722], [0.9531, 0.68717],
     [0.8516, 0.23151], [0.7344, 0.00332], [0.6172, -0.13641], [0.5000, -0.20581], [0.4531, -0.21090],
     [0.2813, -0.15662], [0.1719, -0.10150], [0.1016, -0.06434], [0.0703, -0.04775], [0.0625, -0.04192],
     [0.0547, -0.03717], [0.0000, 0.00000]], dtype=torch.float64,
)  # fmt: skip


def poisson_solve(f: torch.Tensor, V: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r"""Solve ∂_j∂_j ψ = f with ψ = 0 on the walls.   f: interior (n, n) → full ψ (n+2, n+2).

    \hat f_{ab} = V_{ia} f_{ij} V_{jb},   \hat ψ_{ab} = \hat f_{ab} / (Λ_a + Λ_b),   ψ_{ij} = V_{ia} \hat ψ_{ab} V_{jb}
    """
    f_hat = torch.einsum("ia,ij,jb->ab", V, f, V)
    psi_hat = f_hat / (lam[:, None] + lam[None, :])
    psi = torch.einsum("ia,ab,jb->ij", V, psi_hat, V)
    return torch.nn.functional.pad(psi, (1, 1, 1, 1))  # add the walls, where ψ = 0


def time_step(h: float, nu: float, lid: float) -> float:
    """Forward Euler with central differences: stable below h²/(4ν) (diffusion) and h/U (advection)."""
    return 0.5 * min(h**2 / (4 * nu), h / lid)


def simulate(
    omega: torch.Tensor, V: torch.Tensor, lam: torch.Tensor, h: float, nu: float, lid: float, t_max: float,
    save_every: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Return (times (F,), ω snapshots (F, N, N), ψ snapshots (F, N, N))."""
    dt = time_step(h, nu, lid)
    n_steps = int(t_max / dt)
    psi = poisson_solve(-omega[1:-1, 1:-1], V, lam)
    times, omegas, psis = [0.0], [omega], [psi]
    for n in steps(n_steps, "time steps"):
        omega = wall_vorticity(omega, psi, h, lid)
        omega[1:-1, 1:-1] = omega[1:-1, 1:-1] + dt * vorticity_rhs(omega, psi, h, nu)  # forward Euler
        psi = poisson_solve(-omega[1:-1, 1:-1], V, lam)  # ∇²ψ = -ω
        if (n + 1) % save_every == 0:
            times.append((n + 1) * dt)
            omegas.append(omega.clone())
            psis.append(psi)
    return torch.tensor(times), torch.stack(omegas), torch.stack(psis)


def centreline_u(psi: torch.Tensor, h: float) -> torch.Tensor:
    """u = ∂_y ψ along x = 0.5 (N odd, so the centre is a grid line); includes the walls: (N,)."""
    mid = psi.shape[0] // 2
    u = torch.zeros(psi.shape[1], dtype=psi.dtype, device=psi.device)
    u[1:-1] = (psi[mid, 2:] - psi[mid, :-2]) / (2 * h)
    u[-1] = 1.0  # the lid
    return u


def ghia_deviation(y: torch.Tensor, u: torch.Tensor) -> float:
    """Largest |u - u_Ghia| at the Ghia et al. points (linear interpolation of our centreline u)."""
    u_here = np.interp(GHIA_RE100[:, 0].numpy(), y.cpu().numpy(), u.cpu().numpy())
    return float(np.abs(u_here - GHIA_RE100[:, 1].numpy()).max())
