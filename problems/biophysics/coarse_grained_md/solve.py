"""Langevin dynamics (BAOAB splitting) for every temperature and replica at once."""

import math

import torch

from problems._common.progress import steps

from .equation import forces, fraction_native


def langevin(
    x0: torch.Tensor, T: torch.Tensor, model: dict, dt: float, gamma: float, n_steps: int, n_frames: int = 120
):
    """Integrate m dv = F dt - γ m v dt + √(2 γ m k_B T) dW for x0 (S, N, 3), one temperature T_s per system.

    BAOAB: half kick (B), half drift (A), exact friction + noise (O), half drift, half kick. Mass m = 1.
    Returns (times (K,), Q history (K, S), frame times (F,), frames (F, S, N, 3)).
    """
    gen = torch.Generator().manual_seed(1)
    c1 = math.exp(-gamma * dt)  # velocity kept by the friction step
    c2 = torch.sqrt((1 - c1**2) * T)[:, None, None]  # (S, 1, 1): thermal kick size, √(k_B T / m) per component
    x = x0.clone()
    v = torch.randn(x.shape, generator=gen).to(x.device) * torch.sqrt(T)[:, None, None]  # Maxwell–Boltzmann
    f = forces(x, model)
    q_every = max(1, n_steps // 600)
    f_every = max(1, n_steps // n_frames)
    times, qs, frame_t, frames = [0.0], [fraction_native(x, model)], [0.0], [x.clone()]
    for step in steps(n_steps, "Langevin steps"):
        v = v + 0.5 * dt * f  # B
        x = x + 0.5 * dt * v  # A
        v = c1 * v + c2 * torch.randn(v.shape, generator=gen).to(v.device)  # O
        x = x + 0.5 * dt * v  # A
        f = forces(x, model)
        v = v + 0.5 * dt * f  # B
        t = (step + 1) * dt
        if (step + 1) % q_every == 0:
            times.append(t)
            qs.append(fraction_native(x, model))
        if (step + 1) % f_every == 0:
            frame_t.append(t)
            frames.append(x.clone())
    return torch.tensor(times), torch.stack(qs), torch.tensor(frame_t), torch.stack(frames)


def melting_curve(times: torch.Tensor, Q: torch.Tensor, n_temps: int, n_rep: int, t_now: float | None = None):
    """⟨Q⟩ per temperature over the second half of the run (up to t_now), split by start.

    Q: (K, B·R). Returns (from native (B,), from unfolded (B,), all (B,)).
    """
    t_now = times[-1].item() if t_now is None else t_now
    window = (times >= 0.5 * t_now) & (times <= t_now)
    q = Q[window].reshape(-1, n_temps, n_rep).mean(0)  # (B, R)
    half = n_rep // 2
    return q[:, :half].mean(1), q[:, half:].mean(1), q.mean(1)


def melting_temperature(T: torch.Tensor, q: torch.Tensor) -> float:
    """Where ⟨Q⟩ first drops through ½, by linear interpolation (nan if it never does)."""
    for b in range(len(T) - 1):
        if q[b] >= 0.5 > q[b + 1]:
            return (T[b] + (q[b] - 0.5) / (q[b] - q[b + 1]) * (T[b + 1] - T[b])).item()
    return float("nan")
