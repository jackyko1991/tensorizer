"""Feinberg's reinterpretation of backward-in-time signals, and tachyon condensation on a lattice."""

import torch

from problems._common.progress import steps as progress_steps

from . import equation


def reinterpret(dt: torch.Tensor, energy: torch.Tensor) -> dict:
    """Feinberg (1967): a tachyon seen arriving before it left carries negative energy; read it instead as a
    positive-energy antitachyon sent by the receiver to the emitter, forward in time. No observer then sees negative
    energy, but who 'sent' the signal depends on the frame, which is why the antitelephone is still a paradox.
    """
    backward = dt < 0
    return {"emitter_is_receiver": backward & (energy < 0), "energy": energy.abs(), "duration": dt.abs()}


def condense(phi0, m2: float, lam: float, length: float, t_max: float, dt: float, friction: float = 0.0) -> dict:
    """φ_tt + γ φ_t = φ_xx − m²φ − λφ³ from tiny noise, velocity Verlet with the friction as an exact damping factor.

    phi0: (N,) → dict with phi (S+1, N), energy, kinetic and mean potential (S+1,).
    The friction γ plays the role of Hubble friction 3H in an expanding universe.
    """
    n = phi0.shape[-1]
    dx = length / n
    steps = int(round(t_max / dt))
    damp = torch.exp(torch.tensor(-friction * dt / 2, dtype=phi0.dtype))
    phi, pi = phi0.clone(), torch.zeros_like(phi0)
    traj, energy, kinetic, pot = [], [], [], []

    def record():
        grad = (torch.roll(phi, -1, -1) - phi) / dx
        kin = 0.5 * (pi**2).sum() * dx
        v = equation.potential(phi, m2, lam)
        traj.append(phi.clone())
        kinetic.append(kin)
        energy.append(kin + (0.5 * grad**2 + v).sum() * dx)
        pot.append(v.mean())

    record()
    for _ in progress_steps(steps, "lattice"):
        pi = damp * pi + 0.5 * dt * equation.force(phi, m2, lam, dx)
        phi = phi + dt * pi
        pi = damp * (pi + 0.5 * dt * equation.force(phi, m2, lam, dx))
        record()
    return {
        "phi": torch.stack(traj),
        "energy": torch.stack(energy),
        "kinetic": torch.stack(kinetic),
        "potential": torch.stack(pot),
        "x": torch.arange(n, dtype=phi0.dtype) * dx,
    }
