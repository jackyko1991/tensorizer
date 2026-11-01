"""Build the concrete tensors: the ensemble's initial state and the random-number generator."""

import torch

X0 = 2.0  # every particle is released from rest at x0 = 2, well outside the thermal width
SEED = 0  # fixed seed: the same noise ξ_{nb}, hence the same run, every time


def initial_state(n_particles: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """x_b(0) = x0 and v_b(0) = 0 for every particle b:  two tensors of shape (B,)."""
    x = torch.full((n_particles,), X0, device=device)
    v = torch.zeros(n_particles, device=device)
    return x, v


def generator(device: torch.device, seed: int = SEED) -> torch.Generator:
    """A seeded random-number generator on the same device as the tensors."""
    return torch.Generator(device=device).manual_seed(seed)


def time_step(gamma: float, k: float, m: float) -> float:
    """Δt well below both the damping time m/γ and the oscillation time √(m/k)."""
    return min(0.01, 0.05 * m / gamma, 0.05 * (m / k) ** 0.5)
