"""Find the stationary distribution three ways: power iteration, an eigen-decomposition and many random surfers."""

import torch

from problems._common.progress import steps

from .equation import power_step, walk_step


def power_iteration(g: torch.Tensor, n_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Start from the uniform distribution and apply G n_iter times, for every damping factor in the batch.

    g: (B, N, N). Returns the history x (n_iter+1, B, N) and the residuals ‖x_{k+1} − x_k‖₁ (n_iter, B).
    """
    b, n, _ = g.shape
    x = torch.full((b, n), 1.0 / n, dtype=g.dtype, device=g.device)
    history = [x]
    for _ in steps(n_iter, "power iterations"):
        x = power_step(g, x)
        history.append(x)
    history = torch.stack(history)
    residual = (history[1:] - history[:-1]).abs().sum(dim=-1)
    return history, residual


def eig_stationary(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The eigenvector of G (N, N) for eigenvalue 1, scaled to sum to 1, and all |λ| in decreasing order.

    G is not symmetric, so this needs the general torch.linalg.eig (eigh is only for symmetric matrices).
    """
    vals, vecs = torch.linalg.eig(g.double())
    top = vals.real.argmax()
    pi = vecs[:, top].real
    return (pi / pi.sum()).to(g.dtype), vals.abs().sort(descending=True).values


def convergence_factor(residual: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    """Observed rate r with residual_k ∝ r^k: median ratio of successive residuals above round-off.   (K, B) → (B,)"""
    out = []
    for res in residual.T:
        good = res[res > floor]
        ratios = good[1:] / good[:-1]
        out.append(ratios[len(ratios) // 2 :].median())  # the late ratios, after the start-up transient
    return torch.stack(out)


def monte_carlo(s: torch.Tensor, d: float, n_walkers: int, n_steps: int, seed: int) -> torch.Tensor:
    """Fraction of W surfers on each page after every step, starting uniformly: (n_steps+1, N).

    Walkers live on the CPU because the random generator does.
    """
    s = s.cpu()
    n = s.shape[0]
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, n, (n_walkers,), generator=gen)  # (W,)
    occupancy = [torch.bincount(pos, minlength=n) / n_walkers]
    for _ in steps(n_steps, "surfer steps"):
        pos = walk_step(pos, s, d, gen)
        occupancy.append(torch.bincount(pos, minlength=n) / n_walkers)
    return torch.stack(occupancy)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation of two score vectors (Pearson correlation of their ranks)."""
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
