"""The lattice Boltzmann loop: collide → stream → measure force → bounce back → inlet/outlet."""

import torch

from problems._common.progress import steps

from .equation import collide, density, equilibrium, momentum, momentum_exchange_force, stream


def macroscopic(f: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ρ (Nx, Ny) and u (Nx, Ny, 2) from the populations."""
    rho = density(f)
    return rho, momentum(f, e) / rho[..., None]


def vorticity(u: torch.Tensor) -> torch.Tensor:
    """ω = ∂_x u_y - ∂_y u_x by central differences: (Nx, Ny, 2) → (Nx, Ny)."""
    return torch.gradient(u[..., 1], dim=0)[0] - torch.gradient(u[..., 0], dim=1)[0]


def simulate(
    f: torch.Tensor, solid: torch.Tensor, links: torch.Tensor, e: torch.Tensor, w: torch.Tensor, opp: torch.Tensor,
    tau: float, u_in: float, n_steps: int, save_every: int, open_ends: bool = True,
) -> dict:  # fmt: skip
    """Run n_steps lattice updates. Returns force history (n_steps, 2), vorticity snapshots and final f.

    open_ends=True: equilibrium inflow at x = 0 and a copy-the-neighbour outflow at x = Nx-1.
    open_ends=False: fully periodic box (used by the tests to check exact mass conservation).
    """
    fluid = ~solid[..., None]  # (Nx, Ny, 1)
    ny = f.shape[1]
    inflow = equilibrium(
        torch.ones(1, ny, device=f.device), torch.tensor([u_in, 0.0], device=f.device).expand(1, ny, 2), e, w
    )
    forces, snapshots, times = [], [], []
    for n in steps(n_steps, "lattice steps"):
        rho, u = macroscopic(f, e)
        f = torch.where(fluid, collide(f, equilibrium(rho, u, e, w), tau), f)  # no collision inside the solid
        f = stream(f, e)
        forces.append(momentum_exchange_force(f, links, e))  # populations that just hit the cylinder
        f = torch.where(solid[..., None], f[..., opp], f)  # bounce-back: reverse every population in the solid
        if open_ends:
            f[0] = inflow[0]
            f[-1] = f[-2]
        if n % save_every == 0:
            times.append(n)
            snapshots.append(vorticity(u).masked_fill(solid, float("nan")))
    return {"force": torch.stack(forces), "vorticity": torch.stack(snapshots), "steps": torch.tensor(times), "f": f}


def coefficients(force: torch.Tensor, u_in: float, diameter: float) -> torch.Tensor:
    """C_D, C_L = F / (½ ρ U² D) with ρ = 1: (T, 2) → (T, 2)."""
    return force / (0.5 * u_in**2 * diameter)


def strouhal(lift: torch.Tensor, u_in: float, diameter: float) -> float:
    """St = f D / U from the dominant frequency of the lift coefficient over the second half of the run.

    The signal is windowed (Hann) and zero-padded 16× so the spectrum peak is located finely.
    """
    signal = lift[len(lift) // 2 :]
    signal = (signal - signal.mean()) * torch.hann_window(len(signal), device=signal.device)
    n_fft = 16 * len(signal)
    spectrum = torch.fft.rfft(signal, n=n_fft).abs()
    spectrum[0] = 0
    frequency = int(spectrum.argmax()) / n_fft  # cycles per lattice step
    return frequency * diameter / u_in
