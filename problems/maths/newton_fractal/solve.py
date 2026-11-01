"""Run Newton's method on whole grids, record which root every pixel reaches and when, and measure the basins."""

import torch
import torch.nn.functional as F

from problems._common.progress import steps

from .equation import derivative, nearest_root, newton_step, poly


def iterate(z0: torch.Tensor, c: torch.Tensor, roots: torch.Tensor, n_iter: int, tol: float):
    """Newton on every pixel for n_iter steps.

    Returns labels (n_iter+1, ...) int8, the root each pixel is within tol of at step k (or −1), and
    conv_iter (...), the first step at which it got there (n_iter + 1 if it never did).
    """
    labels = torch.empty((n_iter + 1, *z0.shape), dtype=torch.int8, device=z0.device)
    z = z0
    for k in steps(n_iter + 1, "Newton steps"):
        dist, j = nearest_root(z, roots)
        labels[k] = torch.where(dist < tol, j, -1)  # NaN (from p' = 0) compares False: not converged
        if k < n_iter:
            z = newton_step(z, c)
    converged = labels >= 0
    first = converged.to(torch.uint8).argmax(dim=0)  # index of the first True
    conv_iter = torch.where(converged.any(dim=0), first, n_iter + 1)
    return labels, conv_iter


def error_trace(z0: complex, c: torch.Tensor, n_iter: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow one starting point in double precision: errors e_k = |z_k − r| to the root r it converges to."""
    c = c.to(torch.complex128).cpu()
    z = torch.tensor([z0], dtype=torch.complex128)
    path = [z]
    for _ in range(n_iter):
        z = newton_step(z, c)
        path.append(z)
    path = torch.cat(path)  # (n_iter+1,)
    root = path[-1]
    return (path - root).abs(), root


def newton_constant(root: torch.Tensor, c: torch.Tensor) -> float:
    r"""|p''(r) / (2 p'(r))|: the limit of e_{k+1}/e_k² (quadratic convergence)."""
    c = c.to(torch.complex128).cpu()
    d1 = derivative(c)
    return (poly(root, derivative(d1)) / (2 * poly(root, d1))).abs().item()


def roots_present(label: torch.Tensor, n_roots: int) -> torch.Tensor:
    """Which roots occur in the 3×3 neighbourhood of every pixel: (H, W) labels → (J, H, W) booleans."""
    one_hot = torch.stack([(label == j).float() for j in range(n_roots)])  # (J, H, W)
    return F.max_pool2d(one_hot[None], kernel_size=3, stride=1, padding=1)[0] > 0


def boundary_stats(label: torch.Tensor, n_roots: int) -> tuple[float, float]:
    """(fraction of pixels on a basin boundary, fraction of those boundary pixels that touch every basin)."""
    count = roots_present(label, n_roots).sum(dim=0)  # (H, W): number of different basins nearby
    boundary = count >= 2
    wada = count == n_roots
    return boundary.float().mean().item(), (wada.sum() / boundary.sum().clamp(min=1)).item()


def zoom_center(z: torch.Tensor, label: torch.Tensor, n_roots: int, half_width: float) -> complex:
    """A boundary pixel where all basins meet (a Wada point): the one nearest a fixed off-axis spot."""
    wada = roots_present(label, n_roots).all(dim=0)
    target = complex(0.3 * half_width, 0.45 * half_width)
    dist = torch.where(wada, (z - target).abs(), torch.inf)
    return complex(z.flatten()[dist.argmin()].item())
