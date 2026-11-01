"""March the Yee scheme in time with a soft source, keeping snapshots of E_z for the animation."""

import math
from typing import Callable

import torch

from .equation import yee_step


def continuous_wave(wavelength: float, c: float = 1.0) -> Callable[[float], float]:
    """sin(ω t) with ω = 2πc/λ, switched on smoothly over about two periods (a sudden start sprays high frequencies)."""
    omega = 2 * math.pi * c / wavelength
    period = 2 * math.pi / omega
    return lambda t: math.sin(omega * t) * (1 - math.exp(-((t / (2 * period)) ** 2)))


def gaussian_pulse(t0: float, width: float) -> Callable[[float], float]:
    """A single bump exp(−((t − t0)/width)²), used by the 1D checks."""
    return lambda t: math.exp(-(((t - t0) / width) ** 2))


def simulate(E, H, eps_r, damping, source_mask, waveform, eps, dt: float, h: float, n_steps: int, save_every=1):
    """Start from fields E, H (3, Nx, Ny); return (times (F,), E_z snapshots (F, Nx, Ny)).

    source_mask (Nx, Ny) says where the source adds to E_z ("soft" source: the wave passes straight through it).
    """
    times, frames = [], []
    for n in range(n_steps):
        E, H = yee_step(E, H, eps, eps_r, damping, dt, h)
        E[2] += source_mask * waveform((n + 1) * dt)
        if n % save_every == 0:
            times.append((n + 1) * dt)
            frames.append(E[2].clone())
    return torch.tensor(times), torch.stack(frames)


def point_source(nx: int, ny: int, device) -> torch.Tensor:
    """Source mask (Nx, Ny): a single cell a quarter of the way in, slightly below the middle."""
    mask = torch.zeros(nx, ny, device=device)
    mask[nx // 4, int(0.45 * ny)] = 1.0
    return mask


def line_source(nx: int, ny: int, device) -> torch.Tensor:
    """Source mask (Nx, Ny): a whole column of cells, which launches a plane wave travelling along ±x."""
    mask = torch.zeros(nx, ny, device=device)
    mask[nx // 4, :] = 0.3
    return mask
