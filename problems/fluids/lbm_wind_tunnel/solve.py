"""The wind-tunnel loop: collide (BGK / TRT / MRT, optional Smagorinsky LES) → stream → bounce-back on the body →
moving ground and free-slip ceiling → inlet/outlet or a fringe zone that restores the free stream; plus tracer
particles carried by the flow."""

import math

import torch
import torch.nn.functional as F

from problems._common.progress import steps

from . import tensors
from .equation import (
    bounce_back,
    collide_bgk,
    collide_mrt,
    collide_trt,
    density,
    drag_coefficient,
    equilibrium,
    magic_omega_minus,
    momentum,
    momentum_exchange_force,
    smagorinsky_tau,
    stream,
)

# d'Humières et al. (2002) rates for the non-hydrodynamic moments of D3Q19, in the order of tensors.moment_basis;
# the five shear-stress moments (rows 5-9) relax with 1/τ and set the viscosity.
MRT_RATES = [0.0, 0.0, 0.0, 0.0, 1.19, None, None, None, None, None, 1.2, 1.2, 1.2, 1.98, 1.98, 1.98, 1.4, 1.4, 1.4]
SHEAR = slice(5, 10)


def macroscopic(f: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ρ (Nx, Ny, Nz) and u (3, Nx, Ny, Nz) from the populations."""
    rho = density(f)
    return rho, momentum(f, e) / rho


def collide(f, f_eq, rho, e, opp, tau0, collision, les, cs, m=None):
    """One collision with the chosen operator; LES makes the shear relaxation time a field τ(x)."""
    tau = smagorinsky_tau(f, f_eq, rho, e, tau0, cs) if les else torch.tensor(tau0, dtype=f.dtype, device=f.device)
    omega = 1 / tau
    if collision == "bgk":
        return collide_bgk(f, f_eq, omega)
    if collision == "trt":
        return collide_trt(f, f_eq, omega, magic_omega_minus(omega), opp)
    rates = torch.tensor([0.0 if r is None else r for r in MRT_RATES], dtype=f.dtype, device=f.device)
    rates = rates.reshape(19, *[1] * omega.dim()).expand(19, *omega.shape).clone()
    rates[SHEAR] = omega
    return collide_mrt(f, f_eq, m, rates)


def interpolate(u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Trilinear velocity at particle positions: u (3, Nx, Ny, Nz), p (P, 3) in node coordinates → (P, 3)."""
    size = torch.tensor(u.shape[1:], dtype=p.dtype, device=p.device)
    grid = (2 * p / (size - 1) - 1).flip(-1)  # grid_sample wants (W, H, D) = (z, y, x) order in [-1, 1]
    out = F.grid_sample(u[None], grid[None, None, None], align_corners=True)  # (1, 3, 1, 1, P)
    return out[0, :, 0, 0].T


def streamlines(u: torch.Tensor, seeds: torch.Tensor, solid: torch.Tensor, n_points: int, h: float, exit_x: float):
    """Integrate dx/ds = u(x) from all seeds at once with RK4 (trilinear u). Points past exit_x, outside the box or
    inside the body become NaN (the line stops). Returns lines (S, K+1, 3) and speeds |u| along them (S, K+1)."""
    size = torch.tensor(u.shape[1:], dtype=seeds.dtype, device=seeds.device)
    p, points = seeds, [seeds]
    for _ in range(n_points):  # integration steps; every seed advances together
        k1 = interpolate(u, p)
        k2 = interpolate(u, p + 0.5 * h * k1)
        k3 = interpolate(u, p + 0.5 * h * k2)
        k4 = interpolate(u, p + h * k3)
        p = torch.minimum(torch.clamp(p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), min=0), size - 1)
        points.append(p)
    line = torch.stack(points, 1)  # (S, K+1, 3)
    speed = interpolate(u, line.reshape(-1, 3)).norm(dim=-1).reshape(line.shape[:2])
    idx = line.round().long()
    stop = (line[..., 0] > exit_x) | solid[idx[..., 0], idx[..., 1], idx[..., 2]]
    stop = torch.cummax(stop.int(), dim=1).values.bool()  # once stopped, stays stopped
    return line.masked_fill(stop[..., None], float("nan")), speed.masked_fill(stop, float("nan"))


def simulate(
    f: torch.Tensor, solid: torch.Tensor, u_in: float, tau: float, n_steps: int, collision: str = "trt",
    les: bool = False, cs: float = 0.17, walls: bool = True, open_ends: bool = True, save_every: int = 0,
    n_particles: int = 0, window=None, seed: int = 0, ramp: int = 0, fringe: int = 0, seeds=None, save_from: int = 0,
) -> dict:  # fmt: skip
    """Run n_steps lattice updates on populations f (19, Nx, Ny, Nz).

    walls=True: moving ground (speed u_in) under z = 0 and a free-slip ceiling above z = Nz−1; otherwise periodic in z.
    open_ends=True: equilibrium inlet at x = 0 and zero-gradient outlet at x = Nx−1; otherwise periodic in x.
    fringe > 0 (use with open_ends=False): over the last `fringe` nodes the populations are relaxed towards the free
    stream f_eq(1, U), so the periodic box becomes a wind tunnel: the wake is wiped out and uniform flow re-enters at
    x = 0, and the sound waves of the start are absorbed instead of bouncing between an inlet and an outlet.
    ramp > 0: the free-stream and ground speed rise smoothly from 0 to u_in over `ramp` steps (no impulsive start).
    y is always periodic. Returns the force history (n_steps, 3), final populations, and (if save_every, from step
    save_from on) snapshots of the speed in the mid-span x–z plane, of the tracer particles (released continuously
    near x = 1 inside `window`, recycled when they leave) and of streamlines from the rake `seeds` (S, 3).
    """
    dev, dt = f.device, f.dtype
    _, nx, ny, nz = f.shape
    e, w, opp = tensors.lattice(dev, dt)
    m = tensors.moment_basis(e)[0] if collision == "mrt" else None
    links_mask = tensors.body_links(solid, e)  # (19, Nx, Ny, Nz)
    links = links_mask.nonzero(as_tuple=True)  # index tuple of the few boundary links
    incoming = links_mask[opp].nonzero(as_tuple=True)  # population i at x would come out of the body
    frozen = f[:, solid]  # nodes inside the body keep their (rest) state
    up = (e[:, 2] > 0).nonzero().squeeze(1)  # directions entering the fluid from the ground
    down = (e[:, 2] < 0).nonzero().squeeze(1)  # directions entering from the ceiling
    mirror = [
        int(((e == e[i] * torch.tensor([1, 1, -1], dtype=dt, device=dev)).all(1)).nonzero()) for i in down.tolist()
    ]
    wall_kick = (6 * w[up] * e[up, 0])[:, None, None]  # moving wall: 2 w_i ρ (e_i·u_w)/c_s² per unit speed, ρ ≈ 1
    ones = torch.ones(ny, nz, device=dev, dtype=dt)

    def free_stream(speed: float) -> torch.Tensor:
        """f_eq(ρ = 1, u = (speed, 0, 0)) on a y–z plane: (19, Ny, Nz)."""
        return equilibrium(
            ones, torch.tensor([speed, 0.0, 0.0], device=dev, dtype=dt)[:, None, None].expand(3, ny, nz), e, w
        )

    x = torch.arange(fringe, device=dev, dtype=dt)
    sigma = (0.3 * torch.sin(torch.pi * (x + 1) / (fringe + 1)) ** 2)[:, None, None]  # (fringe, 1, 1): smooth bump
    exit_x = nx - 3 - fringe  # tracer particles are recycled here

    gen = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = (torch.tensor(window[0], dtype=dt), torch.tensor(window[1], dtype=dt)) if window else (None, None)

    def spawn(k: int, anywhere: bool) -> torch.Tensor:
        r = torch.rand(k, 3, generator=gen, dtype=dt)
        p = lo + r * (hi - lo)
        p[:, 0] = 1 + r[:, 0] * ((exit_x - 1) if anywhere else 1.5)
        return p.to(dev)

    particles = spawn(n_particles, True) if n_particles else None
    age = torch.zeros(n_particles, device=dev, dtype=dt)  # lattice steps since each particle was (re)released
    forces, slices, tracks, speeds, saved, lines, line_speeds, ux_xy, ages = [], [], [], [], [], [], [], [], []
    for n in steps(n_steps, "lattice steps"):
        speed_now = u_in * math.sin(0.5 * math.pi * min(1.0, n / ramp)) ** 2 if ramp else u_in
        rho, u = macroscopic(f, e)
        f_eq = equilibrium(rho, u, e, w)
        f_post = collide(f, f_eq, rho, e, opp, tau, collision, les, cs, m)
        if fringe:  # f* ← f* − σ(x) (f* − f_eq(1, U)) in the last nodes: the free stream is restored before x wraps
            tail = f_post[:, nx - fringe :]
            f_post[:, nx - fringe :] = tail - sigma * (tail - free_stream(speed_now)[:, None])
        forces.append(momentum_exchange_force(f_post, links, e))  # populations about to hit the body
        f = bounce_back(stream(f_post, e), f_post, incoming, opp)
        f[:, solid] = frozen
        if walls:
            f[down, :, :, -1] = f[mirror, :, :, 0]  # specular reflection: arrived through the wrap = mirrored copy
            f[up, :, :, 0] = f_post[opp[up], :, :, 0] + wall_kick * speed_now  # moving ground: bounce back + push
        if open_ends:
            f[:, 0] = free_stream(u_in)  # equilibrium inlet
            f[:, -1] = f[:, -2]  # zero-gradient outlet
        if n_particles:
            particles = particles + interpolate(u, particles)  # dx = u dt with dt = 1
            age += 1
            idx = particles.round().long()
            idx = torch.minimum(idx.clamp(min=0), torch.tensor([nx - 1, ny - 1, nz - 1], device=dev))
            gone = (particles[:, 0] > exit_x) | solid[idx[:, 0], idx[:, 1], idx[:, 2]] | (particles[:, 1:] < 0).any(1)
            gone |= (particles[:, 1] > ny - 1) | (particles[:, 2] > nz - 1)
            if gone.any():
                particles[gone] = spawn(int(gone.sum()), False)
                age[gone] = 0
        if save_every and n >= save_from and (n - save_from) % save_every == 0:
            saved.append(n)
            speed = u.norm(dim=0).masked_fill(solid, float("nan"))
            slices.append(speed[:, ny // 2].clone())  # mid-span x–z plane
            ux_xy.append(u[0, :, :, nz // 2].masked_fill(solid[:, :, nz // 2], 0.0))  # mid-height x–y plane, u_x
            if n_particles:
                tracks.append(particles.clone())
                ages.append(age.clone())
                speeds.append(interpolate(u, particles).norm(dim=-1))
            if seeds is not None:
                line, line_speed = streamlines(u, seeds, solid, n_points=int(1.1 * exit_x), h=1 / u_in, exit_x=exit_x)
                lines.append(line)
                line_speeds.append(line_speed)
    out = {"force": torch.stack(forces), "f": f, "steps": torch.tensor(saved)}
    if save_every:
        out["slices"], out["ux_xy"] = torch.stack(slices), torch.stack(ux_xy)
        if n_particles:
            out["tracks"], out["speeds"], out["ages"] = torch.stack(tracks), torch.stack(speeds), torch.stack(ages)
        if seeds is not None:
            out["lines"], out["line_speeds"] = torch.stack(lines), torch.stack(line_speeds)
    return out


# ---------------------------------------------------------------- validation: a sphere at Re = 20

SPHERE_CASE = {"shape": (80, 40, 40), "diameter": 10.0, "centre": (20.0, 19.5, 19.5), "re": 20.0, "u": 0.05,
               "n_steps": 2000, "collision": "bgk"}  # fmt: skip


def sphere_case(shape, diameter, centre, re, u, n_steps, collision, device=None, dtype=torch.float32) -> dict:
    """Uniform flow past a sphere in a box periodic across the flow (the OpenLB reference case)."""
    e, w, _ = tensors.lattice(device, dtype)
    nx, ny, nz = shape
    grid = torch.stack(torch.meshgrid(*[torch.arange(s, device=device, dtype=dtype) for s in shape], indexing="ij"), -1)
    solid = ((grid - torch.tensor(centre, device=device, dtype=dtype)) ** 2).sum(-1) <= (diameter / 2) ** 2
    tau = 0.5 + 3 * u * diameter / re
    f = tensors.initial_populations(solid, u, e, w)
    out = simulate(f, solid, u, tau, n_steps, collision=collision, walls=False, open_ends=True)
    cd = drag_coefficient(out["force"][:, 0], u, math.pi * diameter**2 / 4)
    _, vel = macroscopic(out["f"], e)
    return {"cd": cd, "cd_mean": cd[int(0.8 * n_steps) :].mean().item(),
            "midplane_ux": vel[0, :, :, nz // 2].masked_fill(solid[:, :, nz // 2], 0.0).double()}  # fmt: skip


def openlb_residuals(cd: torch.Tensor, ux_plane: torch.Tensor, data_dir) -> dict:
    """Our sphere run minus the precomputed OpenLB run (data/openlb_reference_*.csv): relative C_D difference
    (time averages over the last 40 % of each run) and the relative L2 / L∞ errors and max |Δu|/U of u_x on the
    mid-height plane (fluid nodes only)."""
    from pathlib import Path

    ref = tensors.read_reference_csv(Path(data_dir) / "openlb_reference_sphere_force.csv")
    u, area = SPHERE_CASE["u"], math.pi * SPHERE_CASE["diameter"] ** 2 / 4
    cd_ref_t = drag_coefficient(ref["Fx_lattice"], u, area)
    cd_ref = cd_ref_t[ref["step"] >= 0.6 * ref["step"].max()].mean().item()
    cd_ours = cd[int(0.6 * len(cd)) :].mean().item()
    plane = tensors.read_reference_csv(Path(data_dir) / "openlb_reference_sphere_midplane.csv")
    ux_ref = plane["ux"].reshape(SPHERE_CASE["shape"][0], SPHERE_CASE["shape"][1])
    fluid = ux_ref != 0
    diff = (ux_plane.double().cpu() - ux_ref)[fluid]
    return {
        "cd": cd_ours, "cd_ref": cd_ref, "cd_rel": cd_ours / cd_ref - 1,
        "l2": (diff.norm() / ux_ref[fluid].norm()).item(),
        "linf": (diff.abs().max() / ux_ref[fluid].abs().max()).item(),
        "max_du": diff.abs().max().item() / u, "ux_ref": ux_ref, "cd_ref_t": cd_ref_t, "cd_ref_steps": ref["step"],
    }  # fmt: skip
