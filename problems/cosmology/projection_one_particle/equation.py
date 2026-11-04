"""One worldline -> tensors: slicing it in time, counting particles and antiparticles, the asymmetry it allows.

Shapes in comments: B worldlines, S samples along each, L time slices (levels u). A segment from t_a to t_b crosses
the slice t = u forwards (a particle) when t_a < u <= t_b and backwards (an antiparticle) when t_b < u <= t_a.
Summed indices: the segment index (counting crossings), the harmonic k in Rice's formula.
"""

import math

import torch


def segments(t: torch.Tensor, closed: bool):
    """Start and end times (B, S') of every straight piece of the sampled worldline; a closed one wraps round."""
    if closed:
        return t, torch.roll(t, -1, dims=1)
    return t[:, :-1], t[:, 1:]


def crossings(t: torch.Tensor, levels: torch.Tensor, closed: bool):
    """(B, L) forward (particle) and backward (antiparticle) crossings of every slice, in O(B S) memory.

    A forward segment crosses the sorted levels with index in [#(levels <= t_a), #(levels <= t_b)); each segment
    adds +1 at the first index and -1 past the last into a difference array, and a cumulative sum over the level
    index gives the counts. Backward segments are handled the same way with t_a and t_b swapped.
    """
    a, b = segments(t, closed)  # (B, S')
    ia = torch.searchsorted(levels, a.contiguous(), right=True)  # (B, S') number of levels <= t_a
    ib = torch.searchsorted(levels, b.contiguous(), right=True)
    L = len(levels)
    out = []
    for lo, hi in ((ia, ib), (ib, ia)):  # forward: a -> b; backward: b -> a
        go = (hi > lo).long()
        diff = torch.zeros(t.shape[0], L + 1, dtype=torch.long, device=t.device)
        diff.scatter_add_(1, lo, go)  # summed over segments
        diff.scatter_add_(1, hi, -go)
        out.append(torch.cumsum(diff, 1)[:, :L])
    return out[0], out[1]


def crossings_reference(t: torch.Tensor, levels: torch.Tensor, closed: bool):
    """Slow reference: the (B, S', L) table of which segment crosses which slice, summed over segments."""
    a, b = segments(t, closed)
    a, b, u = a[..., None], b[..., None], levels
    up = (a < u) & (u <= b)
    down = (b < u) & (u <= a)
    return up.sum(1), down.sum(1)


def crossings_periodic(t: torch.Tensor, levels: torch.Tensor, period: float):
    """Crossings of slices u (0 <= u < period) when time is a circle; t (B, S + 1) is unwrapped and closes up to a
    multiple of the period. Every copy u + j * period of a slice is counted, then summed over j."""
    j = torch.arange(math.floor(t.min() / period) - 1, math.ceil(t.max() / period) + 2, device=t.device)
    ext = (levels[None, :] + period * j[:, None]).reshape(-1)  # (J L,) sorted because 0 <= u < period
    up, down = crossings(t, ext, closed=False)
    return up.view(len(t), len(j), -1).sum(1), down.view(len(t), len(j), -1).sum(1)


def net_charge_from_ends(t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """(B, L) the crossing number Q(u) = [t_end >= u] - [t_start >= u]: only the ends of the worldline matter."""
    return (t[:, -1:] >= levels).long() - (t[:, :1] >= levels).long()


def turning_points(t: torch.Tensor, closed: bool):
    """(B, S) masks of local minima (a pair is created) and local maxima (a pair annihilates) of t(s)."""
    d_next = torch.roll(t, -1, 1) - t  # slope after sample j
    d_prev = t - torch.roll(t, 1, 1)  # slope before sample j
    births, deaths = (d_prev < 0) & (d_next > 0), (d_prev > 0) & (d_next < 0)
    if not closed:
        births[:, [0, -1]] = False
        deaths[:, [0, -1]] = False
    return births, deaths


def morse_count(t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """(B, L) crossings of a closed loop from its turning points alone: N(u) = 2 #(minima < u) - 2 #(maxima < u)."""
    births, deaths = turning_points(t, closed=True)
    below = t[..., None] < levels  # (B, S, L)
    return 2 * ((births[..., None] & below).sum(1) - (deaths[..., None] & below).sum(1))


def rice_mean_crossings(levels: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """(L,) Rice's formula for a random Fourier loop over s in [0, 2 pi]: 2 sqrt(l2 / l0) exp(-u^2 / 2 l0),
    with spectral moments l0 = sum_k sigma_k^2 and l2 = sum_k k^2 sigma_k^2 (summed over harmonics k)."""
    k = torch.arange(1, len(sigmas) + 1, dtype=sigmas.dtype, device=sigmas.device)
    l0, l2 = (sigmas**2).sum(), (k**2 * sigmas**2).sum()
    return 2 * torch.sqrt(l2 / l0) * torch.exp(-(levels**2) / (2 * l0))


def crossing_properties(t: torch.Tensor, levels: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
    """(n, 2) mass and charge seen at every crossing: both are read off the one worldline (props = (mass, charge));
    a backward crossing shows the opposite charge (Feynman-Stueckelberg)."""
    a, b = segments(t, closed=True)
    a, b = a[..., None], b[..., None]
    up = ((a < levels) & (levels <= b)).nonzero()  # (n_up, 3): worldline, segment, slice
    down = ((b < levels) & (levels <= a)).nonzero()
    direction = torch.cat([torch.ones(len(up)), -torch.ones(len(down))]).to(t)  # (n,) +1 forward, -1 backward
    mass = props[0].expand(len(direction))
    return torch.stack([mass, props[1] * direction], -1)


def max_asymmetry(N: torch.Tensor, closed: bool) -> torch.Tensor:
    """Largest |Q| / N one worldline allows with N crossings: 0 if closed, (N mod 2) / N if open (|Q| <= 1)."""
    if closed:
        return torch.zeros_like(N, dtype=torch.float64)
    return (N % 2).double() / N.double()


def asymmetry_from_fraction(f: torch.Tensor) -> torch.Tensor:
    """Charge asymmetry (N_- - N_+) / (N_- + N_+) = 1 - 2 f for an antiparticle fraction f."""
    return 1 - 2 * f


def eta_from_omega_b(omega_b_h2: float) -> float:
    """Baryon-to-photon ratio eta = 273.9e-10 Omega_b h^2 (Steigman 2006)."""
    return 273.9e-10 * omega_b_h2


def baryons_in_observable_universe(eta: float, n_gamma_cm3: float, radius_gly: float) -> float:
    """N_B = eta n_gamma V for a sphere of comoving radius radius_gly (1 Gly = 9.4607e24 m)."""
    R = radius_gly * 9.4607e24
    return eta * n_gamma_cm3 * 1e6 * 4 / 3 * math.pi * R**3


def efficiency_ratio(f: torch.Tensor) -> torch.Tensor:
    """Extension (biased projection): with equal numbers N_+ = N_-, an observer who sees antiparticles with
    efficiency r relative to particles measures f = r / (1 + r), so r = f / (1 - f)."""
    return f / (1 - f)
