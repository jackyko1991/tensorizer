"""March the 2D packet with split-step Fourier, recording |ψ|² snapshots and the detector screen."""

import torch

from .equation import kinetic_phase, potential_phase, probability, split_step


def solve(
    psi0: torch.Tensor,
    V: torch.Tensor,
    KX: torch.Tensor,
    KY: torch.Tensor,
    dA: float,
    t_max: float,
    M: torch.Tensor | None = None,
    screen_column: int = 0,
    dt: float = 0.02,
    n_saved: int = 120,
) -> dict[str, torch.Tensor]:
    """Return a dict with times (T,), prob (T, N, N) = |ψ|², norm (T,), screen (T, N) and the final psi.

    screen[t, y] = ∫_0^t |ψ(x_screen, y)|² dt': the probability that has crossed the detector line so far.
    M = None means no absorber (every step is then exactly unitary).
    """
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // n_saved)
    P = potential_phase(V, dt)  # (N, N)
    K = kinetic_phase(KX, KY, dt)  # (N, N)
    M = torch.ones_like(V) if M is None else M

    psi = psi0
    screen = torch.zeros(psi0.shape[1], dtype=torch.float64, device=psi0.device)  # (N,) along y
    times, probs, norms, screens = [0.0], [(psi0.abs() ** 2).float()], [probability(psi0, dA)], [screen.clone()]
    for step in range(1, n_steps + 1):
        psi = split_step(psi, P, K, M)
        screen += psi[screen_column].abs() ** 2 * dt  # accumulate the column x = x_screen
        if step % save_every == 0:
            times.append(step * dt)
            probs.append((psi.abs() ** 2).float())  # float32 snapshots keep memory small
            norms.append(probability(psi, dA))
            screens.append(screen.clone())
    return {
        "times": torch.tensor(times),
        "prob": torch.stack(probs),
        "norm": torch.stack(norms),
        "screen": torch.stack(screens),
        "psi": psi,
    }


def mean_position(prob: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """(⟨x⟩, ⟨y⟩) of a probability density |ψ|² on the grid: (N, N) → (2,)."""
    weights = prob / prob.sum()
    return torch.stack([(weights * X).sum(), (weights * Y).sum()])
