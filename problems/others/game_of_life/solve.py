"""Run the automaton for many generations and record the boards and their populations."""

import torch

from problems._common.progress import steps

from .equation import step


def evolve(x0: torch.Tensor, k: torch.Tensor, generations: int) -> torch.Tensor:
    """History of all boards, shape (G+1, B, H, W), stored as uint8 to save memory."""
    history = torch.empty((generations + 1, *x0.shape), dtype=torch.uint8, device=x0.device)
    x = x0
    history[0] = x.to(torch.uint8)
    for g in steps(generations, "generations"):
        x = step(x, k)
        history[g + 1] = x.to(torch.uint8)
    return history


def population(history: torch.Tensor) -> torch.Tensor:
    """Live cells per board and generation: (G+1, B, H, W) → (G+1, B)."""
    return history.sum(dim=(-2, -1))


def first_repeat(history: torch.Tensor) -> int | None:
    """Generation at which board 0 first repeats an earlier state (a still life or oscillator), or None."""
    seen: dict[bytes, int] = {}
    for g, board in enumerate(history[:, 0]):
        key = board.cpu().numpy().tobytes()
        if key in seen:
            return g
        seen[key] = g
    return None
