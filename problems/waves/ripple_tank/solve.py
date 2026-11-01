"""March the ripple tank forward and keep a running average of u² (the intensity)."""

import math

import torch

from problems._common.progress import steps

from .equation import step


def solve(blob, gamma, K, wavelength: float, c: float, h: float, dt: float, t_max: float, n_frames: int = 120):
    """Return (times (F,), u (F, n, n), intensity (F, n, n)).

    The sources add f = A sin(ωt) · blob with ω = 2πc/λ. The intensity is an exponential running mean of u²
    with a memory of about one period, so it settles to ⟨u²⟩ once the tank is full.
    """
    omega = 2 * math.pi * c / wavelength
    period = wavelength / c
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // (n_frames - 1))
    u = torch.zeros_like(blob)
    u_prev = torch.zeros_like(blob)
    intensity = torch.zeros_like(blob)
    times, us, Is = [0.0], [u], [intensity]
    for n in steps(n_steps, "time steps"):
        f = omega**2 * math.sin(omega * n * dt) * blob  # amplitude ∝ ω² keeps the ripple height similar for any λ
        u, u_prev = step(u, u_prev, f, gamma, K, c, h, dt), u
        intensity = intensity + (u**2 - intensity) * (dt / period)
        if (n + 1) % save_every == 0:
            times.append((n + 1) * dt)
            us.append(u)
            Is.append(intensity)
    return torch.tensor(times), torch.stack(us), torch.stack(Is)


def nodal_contrast(intensity: torch.Tensor, path_diff: torch.Tensor, wavelength: float, region: torch.Tensor):
    """Mean ⟨u²⟩ where |r1 − r2| ≈ (m + ½)λ (predicted nodes) and where |r1 − r2| ≈ mλ (antinodes), inside region."""
    z = path_diff / wavelength
    nodes = region & ((z - 0.5) - torch.round(z - 0.5)).abs().lt(0.08)
    antinodes = region & (z - torch.round(z)).abs().lt(0.08)
    return intensity[nodes].mean().item(), intensity[antinodes].mean().item()
