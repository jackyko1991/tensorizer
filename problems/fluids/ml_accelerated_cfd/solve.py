"""Fine ground truth, plain coarse solver, and training the flux correction through an unrolled simulation."""

import math

import torch

from problems._common.progress import steps

from . import tensors
from .equation import cell_average, step

DT_COARSE = 0.02  # coarse time step (CFL ≈ 0.2 on 32 cells)


def substeps(dx_fine: float, nu: float, u_max: float = 2.0) -> int:
    """Fine steps per coarse step, so the fine solver respects both the CFL and the diffusion limit."""
    dt_max = min(0.3 * dx_fine / u_max, 0.2 * dx_fine**2 / nu)
    return math.ceil(DT_COARSE / dt_max)


@torch.no_grad()
def ground_truth(
    u0_fine: torch.Tensor, nu: float, dx_fine: float, factor: int, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fine solver; return its snapshots every coarse step (n_steps + 1, B, N_f) and their
    coarse-cell averages (n_steps + 1, B, N_c), which are what a perfect coarse solver would produce."""
    k = substeps(dx_fine, nu)
    u, out = u0_fine, [u0_fine]
    for _ in steps(n_steps, "fine solver"):
        for _ in range(k):
            u = step(u, DT_COARSE / k, nu, dx_fine)
        out.append(u)
    fine = torch.stack(out)
    return fine, cell_average(fine, factor)


def rollout(u0: torch.Tensor, nu: float, dx: float, n_steps: int, correction=None) -> torch.Tensor:
    """Coarse trajectory (n_steps + 1, B, N_c), plain (correction=None) or learned."""
    us = [u0]
    for _ in range(n_steps):
        us.append(step(us[-1], DT_COARSE, nu, dx, correction))
    return torch.stack(us)


def train(
    model: torch.nn.Module, truth: torch.Tensor, nu: float, dx: float, n_iter: int, unroll: int, seed: int = 0
) -> list[float]:
    """Fit δF by backpropagating through `unroll` coarse steps (autograd through the whole simulation).

    Each iteration picks random start times and trajectories from truth (T, B, N_c), runs the learned
    solver from those states and compares every step with the truth: L = mean (u_learned - u_true)².
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_iter, 1))
    T, B, _ = truth.shape
    losses = []
    for _ in steps(n_iter, "training"):
        t0 = torch.randint(0, T - unroll, (32,), generator=g)
        b = torch.randint(0, B, (32,), generator=g)
        target = torch.stack([truth[t0 + s, b] for s in range(unroll + 1)])  # (unroll+1, 32, N_c)
        pred = rollout(target[0], nu, dx, unroll, model)
        loss = ((pred[1:] - target[1:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def rms_error(u: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error over space at each time and trajectory: (T, B, N) → (T, B)."""
    return ((u - truth) ** 2).mean(-1).sqrt()


def setup(n_fine: int, n_coarse: int, device: torch.device) -> dict:
    """Both grids and the coarsening factor."""
    x_f, dx_f = tensors.grid(n_fine, device)
    x_c, dx_c = tensors.grid(n_coarse, device)
    return {"x_f": x_f, "dx_f": dx_f, "x_c": x_c, "dx_c": dx_c, "factor": n_fine // n_coarse}
