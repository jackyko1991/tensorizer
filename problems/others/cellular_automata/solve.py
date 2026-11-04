"""Run all 256 elementary rules side by side, sort them by behaviour, and follow a Lenia glider."""

import torch

from problems._common.progress import steps

from .equation import complement_rule, eca_step, lenia_step, mirror_rule

MU, SIGMA, T = 0.15, 0.015, 10.0  # Orbium's growth centre, width and time resolution (dt = 1/T)


def run_eca(x0: torch.Tensor, table: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Space-time diagrams of every rule: (R, W), (R, 8) → (T+1, R, W) uint8."""
    hist = torch.empty((n_steps + 1, *x0.shape), dtype=torch.uint8, device=x0.device)
    hist[0] = x = x0
    for t in range(n_steps):
        hist[t + 1] = x = eca_step(x, table)
    return hist


def equivalence_classes(rules: torch.Tensor) -> torch.Tensor:
    """Smallest rule number among {r, mirror, complement, both}: the representative of r's class.   (R,) → (R,)"""
    m = mirror_rule(rules)
    family = torch.stack([rules, m, complement_rule(rules), complement_rule(m)])  # (4, R)
    return family.min(dim=0).values


def behaviour(x0: torch.Tensor, table: torch.Tensor, n_steps: int, max_period: int = 30) -> torch.Tensor:
    """A rough Wolfram classification after n_steps from random lines: 0 uniform (class 1), 1 periodic up to a shift
    (class 2), 2 no period found (class 3 or 4).   (R, W) → (R,)"""
    hist = run_eca(x0, table, n_steps)
    last = hist[-1]  # (R, W)
    uniform = (last == last[:, :1]).all(-1)
    w = last.shape[-1]
    shifts = torch.arange(-max_period, max_period + 1, device=x0.device)  # (S,)
    idx = (torch.arange(w, device=x0.device) - shifts[:, None]) % w  # (S, W) roll by s
    earlier = hist[-1 - max_period : -1]  # (P, R, W): states 1 … P steps before the end
    rolled = earlier[:, :, idx]  # (P, R, S, W)
    periodic = (rolled == last[None, :, None, :]).all(-1).any(-1).any(0)  # (R,)
    return torch.where(uniform, 0, torch.where(periodic, 1, 2))


def run_lenia(a0: torch.Tensor, kernel: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames (T+1, S, S) float32 and total mass Σ A (T+1,) of a Lenia world."""
    k_hat = torch.fft.fft2(kernel)
    frames = torch.empty((n_steps + 1, *a0.shape), dtype=torch.float32, device=a0.device)
    mass = torch.empty(n_steps + 1, dtype=torch.float64, device=a0.device)
    a = a0
    for t in steps(n_steps + 1, "Lenia steps"):
        frames[t], mass[t] = a, a.sum()
        a = lenia_step(a, k_hat, MU, SIGMA, T)
    return frames, mass


def best_shift(a: torch.Tensor, b: torch.Tensor) -> tuple[tuple[int, int], float]:
    """Translation (dy, dx) that best maps a onto b on the periodic grid, and the normalised overlap ⟨a, b⟩ there."""
    a, b = a.double(), b.double()
    corr = torch.fft.ifft2(torch.fft.fft2(b) * torch.fft.fft2(a).conj()).real  # corr[s] = Σ_p a[p] b[p + s]
    s = corr.shape[-1]
    flat = int(corr.argmax())
    dy, dx = flat // s, flat % s
    dy, dx = (dy + s // 2) % s - s // 2, (dx + s // 2) % s - s // 2  # signed shifts
    return (dy, dx), (corr.max() / (a.norm() * b.norm())).item()
