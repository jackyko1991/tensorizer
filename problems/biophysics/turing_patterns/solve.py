"""Explicit Euler time stepping of the batched reaction–diffusion system, and the wavelength of a pattern."""

import torch

from problems._common.progress import steps

from .equation import growth_rate, jacobian, rhs, uniform_states


def solve(y0: torch.Tensor, F: torch.Tensor, k: torch.Tensor, Du: float, Dv: float, n_steps: int, n_frames: int = 90):
    """y ← y + dt · rhs(y) with dt = 1 (stable while D dt / h² < ¼).

    Returns (frame steps (T,), v snapshots (T, B, n, n), pattern amplitude std(v) after every step (n_steps + 1, B)).
    """
    every = max(1, n_steps // n_frames)
    y = y0.clone()
    at, frames, amp = [0], [y[1].clone()], [y[1].std((-1, -2))]
    for step in steps(n_steps, "reaction–diffusion steps"):
        y = y + rhs(y, F, k, Du, Dv)
        amp.append(y[1].std((-1, -2)))
        if (step + 1) % every == 0:
            at.append(step + 1)
            frames.append(y[1].clone())
    return torch.tensor(at), torch.stack(frames), torch.stack(amp)


def dominant_wavelength(v: torch.Tensor) -> torch.Tensor:
    """Wavelength (grid cells) of the strongest ring of the power spectrum |FFT v|² of each pattern v (B, n, n).

    Returns (B,)."""
    n = v.shape[-1]
    power = torch.fft.fft2(v - v.mean((-1, -2), keepdim=True)).abs() ** 2  # (B, n, n)
    f = torch.fft.fftfreq(n, device=v.device) * n  # waves per box along each axis
    rings = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).round().long().clamp(max=n // 2)  # (n, n)
    spectrum = torch.zeros(v.shape[0], n // 2 + 1, device=v.device)
    spectrum.index_add_(1, rings.flatten(), power.flatten(1))  # sum the power on each ring
    spectrum[:, 0] = 0  # ignore the mean
    return n / spectrum.argmax(1).float()


def stability_map(F: torch.Tensor, k: torch.Tensor, Du: float, Dv: float) -> torch.Tensor:
    """Classify every (F, k) of a grid: 0 only u = 1, v = 0 exists; 1 more uniform states, none Turing-unstable;
    2 a uniform state that is stable at q = 0 but grows at some q ≠ 0 (Turing).   F (nF,), k (nk,) → (nF, nk)."""
    q = torch.linspace(0, 1.5, 151)
    out = torch.zeros(len(F), len(k), dtype=torch.long)
    for a, f in enumerate(F.tolist()):
        for b, kk in enumerate(k.tolist()):
            states = uniform_states(f, kk)[1:]
            if not states:
                continue
            out[a, b] = 1
            for u, v in states:
                lam = growth_rate(jacobian(u, v, f, kk), Du, Dv, q)
                if lam[0] < 0 < lam.max():
                    out[a, b] = 2
    return out
