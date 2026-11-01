"""Diagonalise all (l, j) channels at once with a batched eigh, then sort the levels and find the shell gaps."""

import torch

from problems._common.progress import steps

from .equation import hamiltonian, occupancy, potential


def bound_levels(
    r: torch.Tensor,
    d2: torch.Tensor,
    ell: torch.Tensor,
    j: torch.Tensor,
    v0: float,
    v_so: float,
    radius: float,
    a: float,
    r0: float,
    n_radial: int = 6,
) -> torch.Tensor:
    """E_{kn} of the bound states (E < 0) for every channel k: shape (K, n_radial), NaN where unbound."""
    h = hamiltonian(d2, potential(r, ell, j, v0, v_so, radius, a, r0))  # (K, N, N)
    energies = torch.linalg.eigvalsh(h)[:, :n_radial]  # (K, n_radial) lowest levels of each channel
    return torch.where(energies < 0, energies, torch.nan)


def sweep(
    r: torch.Tensor,
    d2: torch.Tensor,
    ell: torch.Tensor,
    j: torch.Tensor,
    v0: float,
    strengths: torch.Tensor,
    radius: float,
    a: float,
    r0: float,
) -> torch.Tensor:
    """Bound levels while the spin–orbit strength is turned up: shape (S, K, n_radial)."""
    return torch.stack(
        [bound_levels(r, d2, ell, j, v0, strengths[s].item(), radius, a, r0) for s in steps(len(strengths), "eigh")]
    )


def level_scheme(energies: torch.Tensor, ell: torch.Tensor, j: torch.Tensor) -> dict[str, torch.Tensor]:
    """Flatten the bound levels, sort them by energy and count nucleons.

    Returns energy, channel index k, radial index n, and the cumulative occupancy Σ (2j + 1) up to and
    including each level: all of shape (M,) for the M bound levels.
    """
    k, n = torch.nonzero(~torch.isnan(energies), as_tuple=True)
    e = energies[k, n]
    order = torch.argsort(e)
    k, n, e = k[order], n[order], e[order]
    return {"energy": e, "channel": k, "radial": n, "filled": torch.cumsum(occupancy(j[k]), dim=0)}


def largest_gaps(scheme: dict[str, torch.Tensor], count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The `count` biggest jumps between consecutive levels: (gap sizes, nucleon number below each gap)."""
    gaps = torch.diff(scheme["energy"])  # (M − 1,)
    top = torch.argsort(gaps, descending=True)[:count]
    return gaps[top], scheme["filled"][top]
