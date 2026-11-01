"""Build the concrete tensors: network weights, the few noisy measurements and the collocation times."""

import math

import torch

from .equation import exact_solution


def init_weights(widths: list[int], seed: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Xavier-initialised weights for layer sizes widths = [1, 32, 32, 32, 1].

    Layer l has W^{(l)} of shape (widths[l+1], widths[l]) and b^{(l)} of shape (widths[l+1],).
    Every tensor requires gradients: these are the parameters w that training adjusts.
    """
    gen = torch.Generator().manual_seed(seed)  # same seed ⇒ same starting network
    weights = []
    for n_in, n_out in zip(widths[:-1], widths[1:]):
        bound = math.sqrt(6.0 / (n_in + n_out))
        W = (torch.rand(n_out, n_in, generator=gen) * 2 - 1) * bound  # uniform in [-bound, bound]
        b = torch.zeros(n_out)
        weights.append((W.to(device).requires_grad_(True), b.to(device).requires_grad_(True)))
    return weights


def observations(
    n_data: int, t_data_max: float, noise: float, m: float, mu: float, k: float, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """n_data noisy measurements (t_d, x_d), evenly spaced on [0, t_data_max]: shapes (D,), (D,)."""
    gen = torch.Generator().manual_seed(seed + 1)
    t_d = torch.linspace(0.0, t_data_max, n_data)
    x_d = exact_solution(t_d, m, mu, k) + noise * torch.randn(n_data, generator=gen)
    return t_d.to(device), x_d.to(device)


def collocation_points(n_colloc: int, t_max: float, device: torch.device) -> torch.Tensor:
    """t_c, shape (C,): where the equation of motion is enforced, spread over the WHOLE interval."""
    return torch.linspace(0.0, t_max, n_colloc, device=device)


def plot_grid(t_max: float, device: torch.device, n: int = 300) -> torch.Tensor:
    """Fine time grid (n,) for drawing the curves and measuring the error."""
    return torch.linspace(0.0, t_max, n, device=device)
