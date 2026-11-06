"""Concrete tensors: the four data sets, all simulated here (nothing downloaded), and the noise model.

Every data set is a dict with U (F, *axes) whose last axis is time, the spacing h per axis (None for a batch axis of
independent trajectories), coordinates, the library to search, the true ξ (F, K) and the weak-form test functions.

1. Lorenz system (ODE): 8 trajectories from random starts, RK4.
2. Heat equation u_t = D u_xx on a periodic rod: exact, by Fourier modes decaying as e^{-D k² t}.
3. Viscous Burgers u_t = −u u_x + ν u_xx: exact, by the Cole–Hopf transform u = −2ν φ_x / φ of a heat solution φ.
4. Maxwell's equations in 2D vacuum (TM mode E_z, B_x, B_y): a Yee FDTD simulation (Yee 1966), sampled as a
   measurement would be (all fields interpolated to the same points and times).
"""

import math

import torch

from . import equation as eq

DT = torch.float64  # every tensor in float64: derivatives of data amplify round-off

TARGETS = ("lorenz", "heat", "burgers", "maxwell")
LORENZ_PARAMS = (10.0, 28.0, 8.0 / 3.0)  # σ, ρ, β
HEAT_D = 0.2  # thermal diffusivity (cm²/s)
BURGERS_NU = 0.1  # viscosity (m²/s)
C_LIGHT = 29.9792458  # speed of light (cm/ns)
NOISE_LEVELS = (0.0, 0.001, 0.01, 0.05)  # Gaussian noise, fraction of each field's standard deviation


# ---------------------------------------------------------------- 1. Lorenz
def lorenz_rhs(X: torch.Tensor) -> torch.Tensor:
    """(ẋ, ẏ, ż) = (σ(y − x), x(ρ − z) − y, xy − βz).   X: (..., 3)"""
    s, r, b = LORENZ_PARAMS
    x, y, z = X.unbind(-1)
    return torch.stack([s * (y - x), x * (r - z) - y, x * y - b * z], -1)


def lorenz(n_traj: int = 8, t_end: float = 5.0, dt: float = 0.0025, substeps: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n_traj, 3, generator=g, dtype=DT) * torch.tensor([30.0, 40.0, 35.0], dtype=DT) - torch.tensor(
        [15.0, 20.0, -5.0], dtype=DT
    )
    n_t, k = round(t_end / dt) + 1, dt / substeps
    out = [X]
    for _ in range((n_t - 1) * substeps):  # RK4, all trajectories at once
        k1 = lorenz_rhs(X)
        k2 = lorenz_rhs(X + 0.5 * k * k1)
        k3 = lorenz_rhs(X + 0.5 * k * k2)
        X = X + k / 6 * (k1 + 2 * k2 + 2 * k3 + lorenz_rhs(X + k * k3))
        out.append(X)
    U = torch.stack(out[::substeps], -1).permute(1, 0, 2)  # (3, traj, t)
    s, r, b = LORENZ_PARAMS
    truth = eq.truth_matrix(eq.LORENZ, [{"x": -s, "y": s}, {"x": r, "y": -1.0, "x z": -1.0}, {"x y": 1.0, "z": -b}])
    return {"name": "lorenz", "U": U, "h": (None, dt), "coords": [torch.arange(n_traj, dtype=DT),
            torch.arange(n_t, dtype=DT) * dt],
            "fields": ["x", "y", "z"], "lhs": ["\\dot{x}", "\\dot{y}", "\\dot{z}"], "terms": eq.LORENZ,
            "truth": truth, "weak": {"radius": (None, 60), "stride": (None, 30)}}  # fmt: skip


# ---------------------------------------------------------------- 2. heat
def heat(n_x: int = 256, L: float = 10.0, t_end: float = 5.0, n_t: int = 101, bumps=None, seed: int = 0) -> dict:
    """Temperature excess u (K) on a periodic rod of length L (cm); initial Gaussian bumps (centre, width, height)."""
    g = torch.Generator().manual_seed(seed)
    if bumps is None:
        r = torch.rand(4, 3, generator=g, dtype=DT)
        bumps = [(L * a, 0.3 + 0.5 * b, 1.0 + 2.0 * c) for a, b, c in r.tolist()]
    x, t = torch.arange(n_x, dtype=DT) * L / n_x, torch.linspace(0, t_end, n_t, dtype=DT)
    u0 = sum(A * torch.exp(-(((x - c + L / 2) % L - L / 2) ** 2) / (2 * s**2)) for c, s, A in bumps)
    k = 2 * math.pi * torch.fft.rfftfreq(n_x, d=L / n_x, dtype=DT)
    u = torch.fft.irfft(torch.fft.rfft(u0)[:, None] * torch.exp(-HEAT_D * k[:, None] ** 2 * t[None, :]), n=n_x, dim=0)
    return {"name": "heat", "U": u[None], "h": (L / n_x, t_end / (n_t - 1)), "coords": [x, t], "fields": ["u"],
            "lhs": ["u_t"], "terms": eq.SCALAR_1D, "truth": eq.truth_matrix(eq.SCALAR_1D, [{"u_{xx}": HEAT_D}]),
            "weak": {"radius": (16, 10), "stride": (8, 5)}, "bumps": bumps}  # fmt: skip


# ---------------------------------------------------------------- 3. Burgers
def burgers(n_x: int = 256, t_end: float = 3.0, n_t: int = 151) -> dict:
    """Velocity u (m/s) on a periodic line of length 2π m. Cole–Hopf: u0 = −2ν ∂_x log φ0, φ0 = exp(−∫u0 / 2ν), then φ
    solves the heat equation φ_t = ν φ_xx exactly (Fourier modes), and u = −2ν φ_x / φ solves Burgers exactly."""
    L = 2 * math.pi
    x, t = torch.arange(n_x, dtype=DT) * L / n_x, torch.linspace(0, t_end, n_t, dtype=DT)
    U0 = 0.6 * (1 - torch.cos(x)) + 0.15 * (
        math.cos(1.0) - torch.cos(2 * x + 1)
    )  # ∫ u0, u0 = 0.6 sin x + 0.3 sin(2x+1)
    phi0 = torch.exp(-U0 / (2 * BURGERS_NU))
    k = 2 * math.pi * torch.fft.rfftfreq(n_x, d=L / n_x, dtype=DT)
    ph = torch.fft.rfft(phi0)[:, None] * torch.exp(-BURGERS_NU * k[:, None] ** 2 * t[None, :])  # (k, t)
    phi = torch.fft.irfft(ph, n=n_x, dim=0)
    phi_x = torch.fft.irfft(1j * k[:, None] * ph, n=n_x, dim=0)
    u = -2 * BURGERS_NU * phi_x / phi
    truth = eq.truth_matrix(eq.SCALAR_1D, [{"u u_{x}": -1.0, "u_{xx}": BURGERS_NU}])
    return {"name": "burgers", "U": u[None], "h": (L / n_x, t_end / (n_t - 1)), "coords": [x, t], "fields": ["u"],
            "lhs": ["u_t"], "terms": eq.SCALAR_1D, "truth": truth,
            "weak": {"radius": (16, 10), "stride": (8, 6)}}  # fmt: skip


# ---------------------------------------------------------------- 4. Maxwell (Yee FDTD)
def yee_fdtd(n: int = 128, h: float = 0.05, courant: float = 0.5, steps: int = 240, stride: int = 3, start: int = 6,
             seed: int = 0, record: bool = False) -> dict:  # fmt: skip
    """TM mode in a periodic square box (side n h cm): E_z at cell corners (i, j) and integer steps, B_x at (i, j+½),
    B_y at (i+½, j) and half steps (Yee 1966). Leap-frog:
        B_x ← B_x − Δt ∂_y E_z,   B_y ← B_y + Δt ∂_x E_z,   E_z ← E_z + c² Δt (∂_x B_y − ∂_y B_x)
    Snapshots every `stride` steps, with B interpolated (4-point cubic, in space and time) to the E_z points and times.
    Courant number S = c Δt / h must stay ≤ 1/√2 in 2D."""
    if courant > 1 / math.sqrt(2):
        raise ValueError(f"Courant number {courant} > 1/√2: the 2D Yee scheme would be unstable")
    c, dt, L = C_LIGHT, courant * h / C_LIGHT, n * h
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(n, dtype=DT) * h
    E = torch.zeros(n, n, dtype=DT)
    for cx, cy, s, A in (
        torch.rand(3, 4, generator=g, dtype=DT) * torch.tensor([L, L, 0.3, 1.5], dtype=DT)
        + torch.tensor([0, 0, 0.5, 0.5], dtype=DT)
    ).tolist():
        dx, dy = (x[:, None] - cx + L / 2) % L - L / 2, (x[None, :] - cy + L / 2) % L - L / 2
        E = E + A * torch.exp(-(dx**2 + dy**2) / (2 * s**2))
    Bx, By = torch.zeros(n, n, dtype=DT), torch.zeros(n, n, dtype=DT)  # B^{-1/2} = 0: the pulses start at rest
    Es, Bxs, Bys, energy, div_b = [E], [], [], [], []
    for _ in range(steps):
        Bx_new = Bx - dt * (torch.roll(E, -1, 1) - E) / h  # ∂_y E_z at (i, j+½)
        By_new = By + dt * (torch.roll(E, -1, 0) - E) / h  # ∂_x E_z at (i+½, j)
        if record:  # Yee's conserved energy: E^n·E^n + c² B^{n−½}·B^{n+½}
            energy.append(float((E * E).sum() + c**2 * (Bx * Bx_new + By * By_new).sum()) * h * h)
        Bx, By = Bx_new, By_new
        curl = (By - torch.roll(By, 1, 0)) / h - (Bx - torch.roll(Bx, 1, 1)) / h  # at (i, j)
        E = E + c**2 * dt * curl
        Es.append(E), Bxs.append(Bx), Bys.append(By)
        if record:
            div_b.append(((torch.roll(Bx, -1, 0) - Bx) / h + (torch.roll(By, -1, 1) - By) / h).abs().max())

    def cubic(a, b, cc, d):  # value midway between b and cc from 4 equally spaced samples
        return (-a + 9 * b + 9 * cc - d) / 16

    snaps = list(range(start, steps - 1, stride))
    Ez = torch.stack([Es[k] for k in snaps], -1)
    Bx_t = torch.stack([cubic(Bxs[k - 2], Bxs[k - 1], Bxs[k], Bxs[k + 1]) for k in snaps], -1)  # B^{k±½, k±3/2} → B^k
    By_t = torch.stack([cubic(Bys[k - 2], Bys[k - 1], Bys[k], Bys[k + 1]) for k in snaps], -1)
    Bx_c = cubic(torch.roll(Bx_t, 2, 1), torch.roll(Bx_t, 1, 1), Bx_t, torch.roll(Bx_t, -1, 1))  # (i, j+½) → (i, j)
    By_c = cubic(torch.roll(By_t, 2, 0), torch.roll(By_t, 1, 0), By_t, torch.roll(By_t, -1, 0))  # (i+½, j) → (i, j)
    out = {"U": torch.stack([Ez, Bx_c, By_c]), "h": (h, h, stride * dt), "x": x,
           "t": torch.tensor(snaps, dtype=DT) * dt, "dt": dt}  # fmt: skip
    if record:
        out |= {"energy": torch.tensor(energy), "div_b": torch.stack(div_b),
                "b_scale": max(float(b.abs().max()) for b in Bxs + Bys) / h}  # fmt: skip
    return out


def maxwell(n: int = 128, steps: int = 240) -> dict:
    sim = yee_fdtd(n=n, steps=steps)
    c2 = C_LIGHT**2
    truth = eq.truth_matrix(eq.MAXWELL, [{"\\partial_x B_y": c2, "\\partial_y B_x": -c2}, {"\\partial_y E_z": -1.0},
                                         {"\\partial_x E_z": 1.0}])  # fmt: skip
    return {"name": "maxwell", "U": sim["U"], "h": sim["h"], "coords": [sim["x"], sim["x"], sim["t"]],
            "fields": ["E_z", "B_x", "B_y"], "lhs": ["\\partial_t E_z", "\\partial_t B_x", "\\partial_t B_y"],
            "terms": eq.MAXWELL, "truth": truth, "weak": {"radius": (10, 10, 8), "stride": (8, 8, 6)}}  # fmt: skip


def dataset(name: str, small: bool = False) -> dict:
    """One target. small=True: a lighter version for the tests."""
    if name == "lorenz":
        return lorenz(n_traj=4 if small else 8)
    if name == "heat":
        return heat()
    if name == "burgers":
        return burgers()
    if name == "maxwell":
        return maxwell(n=96, steps=150) if small else maxwell()
    raise ValueError(name)


def add_noise(U: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    """Gaussian noise of standard deviation level × std(U_f), field by field."""
    if level == 0:
        return U
    g = torch.Generator().manual_seed(seed)
    std = U.reshape(len(U), -1).std(1).view(-1, *[1] * (U.dim() - 1))
    return U + level * std * torch.randn(U.shape, generator=g, dtype=U.dtype)
