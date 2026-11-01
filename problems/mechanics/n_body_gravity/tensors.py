"""Build the concrete tensors: masses m_a (N,), positions x_ai (N, 3) and velocities v_ai (N, 3).

Units are chosen so that G = 1. We use float64 so that energy conservation is limited
by the integrator, not by round-off.
"""

import math

import torch

G = 1.0
DTYPE = torch.float64


def to_center_of_mass(m: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift so the centre of mass sits at rest at the origin: x_ai -= m_b x_bi / M (same for v)."""
    M = m.sum()
    x_com = torch.einsum("b,bi->i", m, x) / M  # (3,)
    v_com = torch.einsum("b,bi->i", m, v) / M  # (3,)
    return x - x_com, v - v_com


def figure_eight(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three equal masses chasing each other on a figure-8 (Chenciner & Montgomery). Period ≈ 6.326."""
    m = torch.ones(3, dtype=DTYPE)
    x = torch.tensor([[0.97000436, -0.24308753, 0.0], [-0.97000436, 0.24308753, 0.0], [0.0, 0.0, 0.0]], dtype=DTYPE)
    v3 = torch.tensor([-0.93240737, -0.86473146, 0.0], dtype=DTYPE)
    v = torch.stack([-v3 / 2, -v3 / 2, v3])  # (3, 3) total momentum is zero
    return m.to(device), x.to(device), v.to(device)


def star_and_planets(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A star of mass 1 with four light planets on circular, slightly tilted orbits."""
    radius = [0.6, 1.0, 1.5, 2.2]
    planet_mass = [1e-3, 3e-3, 2e-3, 1e-3]
    tilt = [math.radians(d) for d in (0.0, 6.0, -8.0, 12.0)]  # inclination about the x axis
    phase = [0.0, 2.0, 4.0, 1.0]  # starting angle on the orbit

    m, x, v = [1.0], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]  # the star
    for r, mp, inc, th in zip(radius, planet_mass, tilt, phase):
        speed = math.sqrt(G * 1.0 / r)  # circular orbit: v² / r = G M / r²
        m.append(mp)
        x.append([r * math.cos(th), r * math.sin(th) * math.cos(inc), r * math.sin(th) * math.sin(inc)])
        v.append([-speed * math.sin(th), speed * math.cos(th) * math.cos(inc), speed * math.cos(th) * math.sin(inc)])
    m, x, v = (torch.tensor(a, dtype=DTYPE, device=device) for a in (m, x, v))
    x, v = to_center_of_mass(m, x, v)
    return m, x, v


def binary_and_planet(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two equal stars on a tight circular orbit, plus a light planet circling both on a tilted orbit."""
    m, x, v = two_body_circular(0.5, 0.5, 0.5, device)  # the binary, already in its centre-of-mass frame
    r, inc = 2.0, math.radians(20.0)
    speed = math.sqrt(G * m.sum().item() / r)  # from far away the binary pulls like one mass of 1
    planet_x = torch.tensor([[0.0, r * math.cos(inc), r * math.sin(inc)]], dtype=DTYPE, device=device)
    planet_v = torch.tensor([[-speed, 0.0, 0.0]], dtype=DTYPE, device=device)
    m = torch.cat([m, torch.tensor([1e-3], dtype=DTYPE, device=device)])  # (3,)
    x, v = torch.cat([x, planet_x]), torch.cat([v, planet_v])  # (3, 3) each
    x, v = to_center_of_mass(m, x, v)
    return m, x, v


def two_body_circular(m1: float, m2: float, separation: float, device: torch.device | None = None):
    """Two bodies on a circular orbit about their centre of mass (used to check Kepler's third law)."""
    m = torch.tensor([m1, m2], dtype=DTYPE, device=device)
    omega = math.sqrt(G * (m1 + m2) / separation**3)  # Kepler: ω² a³ = G (m1 + m2)
    x = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=DTYPE, device=device)
    v = torch.tensor([[0.0, 0.0, 0.0], [0.0, omega * separation, 0.0]], dtype=DTYPE, device=device)
    x, v = to_center_of_mass(m, x, v)
    return m, x, v


def cluster(n: int = 6, seed: int = 34, device: torch.device | None = None):
    """n equal stars scattered in 3D (seeded), velocities scaled to virial equilibrium 2K = −U."""
    gen = torch.Generator().manual_seed(seed)
    m = torch.full((n,), 1.0 / n, dtype=DTYPE)
    x = torch.randn(n, 3, generator=gen, dtype=DTYPE)  # (n, 3)
    v = torch.randn(n, 3, generator=gen, dtype=DTYPE)  # (n, 3)
    x, v = to_center_of_mass(m, x, v)
    r = (x[:, None, :] - x[None, :, :]).norm(dim=-1)  # (n, n) pair distances r_ab
    pairs = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    potential = -(G * m[:, None] * m[None, :] / r)[pairs].sum()
    kinetic = 0.5 * torch.einsum("a,ai,ai->", m, v, v)
    v = v * torch.sqrt(-potential / (2 * kinetic))  # virial ratio K / |U| = 1/2: a bound, breathing cluster
    return m.to(device), x.to(device), v.to(device)


def initial_state(scenario: str, device: torch.device | None = None):
    """(m (N,), x (N, 3), v (N, 3)) for the chosen scenario."""
    if scenario == "3D cluster":
        return cluster(device=device)
    if scenario == "figure-8":
        return figure_eight(device)
    if scenario == "star and planets":
        return star_and_planets(device)
    return binary_and_planet(device)
