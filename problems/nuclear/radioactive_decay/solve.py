"""Monte Carlo: one Bernoulli trial per nucleus per time step, all nuclei at once."""

import torch

from problems._common.progress import steps

from .equation import decay_probability, decay_step


def simulate(alive0: torch.Tensor, lam: float, dt: float, n_steps: int, seed: int = 0) -> torch.Tensor:
    """Return the state of every nucleus at every step: shape (T, *alive0.shape) bool.

    alive0 can carry extra leading axes (e.g. R repeated experiments of an S×S grid); each step draws one
    uniform number per nucleus and keeps it only if u ≥ p. A fixed seed makes the run repeatable.
    """
    gen = torch.Generator(device=alive0.device).manual_seed(seed)
    p = decay_probability(lam, dt)
    alive = alive0
    history = [alive]
    for _ in steps(n_steps, "decay steps"):
        u = torch.rand(alive.shape, generator=gen, device=alive.device)  # one random number per nucleus
        alive = decay_step(alive, u, p)
        history.append(alive)
    return torch.stack(history)
