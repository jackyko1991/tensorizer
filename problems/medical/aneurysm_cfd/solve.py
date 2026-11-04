"""The solver: voxel finite-volume mesh (owner/neighbour lists), Chorin projection with a conjugate-gradient
pressure solve, Womersley inflow, wall shear stress, tracer particles, and the VMTK-style centreline."""

import math

import torch
import torch.nn.functional as Fn

from problems._common.progress import steps

from . import equation as eq

DIRS = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]  # neighbour slot s = 2·axis + (step > 0)


# ---------------------------------------------------------------- mesh: cells, faces, neighbour tables
def shift(a: torch.Tensor, axis: int, step: int, periodic: bool, fill) -> torch.Tensor:
    """out[i] = a[i + step] along axis; `fill` where that falls outside a non-periodic box."""
    out = torch.roll(a, -step, axis)
    if not periodic:
        sl = [slice(None)] * a.dim()
        sl[axis] = slice(-step, None) if step > 0 else slice(0, -step)
        out[tuple(sl)] = fill
    return out


def build_mesh(grid: dict, periodic=(False, False, False)) -> dict:
    """Fluid cells, faces between fluid neighbours (owner → neighbour along +x, +y, +z), face neighbour tables with
    ghost factors for the no-slip wall, and cell kinds from the caps: 0 interior, 1 inlet, 2 outlet."""
    inside, h, lo, phi = grid["inside"], grid["h"], grid["lo"], grid["phi"]
    shape = inside.shape
    n = int(inside.sum())
    idx = torch.full(shape, -1, dtype=torch.long)
    idx[inside] = torch.arange(n)
    cells = inside.nonzero()  # (N, 3) grid indices, same order as idx
    pos = lo + h * cells.double()  # (N, 3) cell centres (m)
    nb = torch.stack([shift(idx, a, s, periodic[a], -1)[inside] for a, s in DIRS], 1)  # (N, 6)

    owner, neigh, dirn = [], [], []
    face_of = torch.full((n, 3), -1, dtype=torch.long)
    for d in range(3):
        o = (nb[:, 2 * d + 1] >= 0).nonzero()[:, 0]
        start = sum(len(x) for x in owner)
        face_of[o, d] = torch.arange(start, start + len(o))
        owner.append(o), neigh.append(nb[o, 2 * d + 1]), dirn.append(torch.full_like(o, d))
    owner, neigh, dirn = torch.cat(owner), torch.cat(neigh), torch.cat(dirn)
    e = torch.eye(3, dtype=torch.float64)
    fpos = pos[owner] + 0.5 * h * e[dirn]  # (F, 3) face centres

    # same-direction neighbour faces; a missing one is a wall: 0 across it (normal) or a ghost value (tangential)
    phi_at = lambda g: phi[tuple(g.T)]  # noqa: E731  φ at integer grid indices (F, 3)
    gmax = torch.tensor(shape) - 1
    wrap = lambda g: torch.where(torch.tensor(periodic), g % torch.tensor(shape), g.clamp(min=0).minimum(gmax))  # noqa: E731
    phi_f = 0.5 * (phi_at(cells[owner]) + phi_at(cells[neigh]))  # (F,)
    fnb = torch.full((len(owner), 6), -1, dtype=torch.long)
    ghost = torch.zeros(len(owner), 6, dtype=torch.float64)
    for s, (a, sg) in enumerate(DIRS):
        o2 = nb[owner, s]
        fnb[:, s] = torch.where(o2 >= 0, face_of[o2.clamp(min=0), dirn], -1)
        off = (sg * e[a]).long()
        phi_g = 0.5 * (phi_at(wrap(cells[owner] + off)) + phi_at(wrap(cells[owner] + off + e[dirn].long())))
        theta = (phi_f / (phi_f - phi_g).clamp(min=1e-12)).clamp(0.25, 1.0)  # wall at θh from the face centre
        ghost[:, s] = torch.where(dirn == a, 0.0, 1 - 1 / theta)  # linear profile through u = 0 at the wall

    # the 4 faces of each other direction e around face f (both sides of owner and neighbour): (F, 3, 4), −1 = wall
    def faces_e(c, e_):
        minus = torch.where(nb[c, 2 * e_] >= 0, face_of[nb[c, 2 * e_].clamp(min=0), e_], -1)
        return torch.stack([face_of[c, e_], minus], 1)

    tnb = torch.stack([torch.cat([faces_e(owner, e_), faces_e(neigh, e_)], 1) for e_ in range(3)], 1)

    kind = torch.zeros(n, dtype=torch.long)
    for k, cap in enumerate(grid["caps"]):
        s_n = (pos - cap["centre"]) @ cap["normal"]  # signed distance from the cap plane (negative inside)
        lateral = (pos - cap["centre"] - s_n[:, None] * cap["normal"]).norm(dim=1)
        near = (s_n > -1.5 * h) & (lateral < cap["radius"] + 2 * h)
        kind[near] = 1 if k == 0 else 2
    fixed = (kind[owner] == 1) | (kind[neigh] == 1)
    corr = ~fixed & ~((kind[owner] == 2) & (kind[neigh] == 2))
    s_in = torch.where((kind[owner] == 1) & (kind[neigh] != 1), 1.0, 0.0) - torch.where(
        (kind[neigh] == 1) & (kind[owner] != 1), 1.0, 0.0
    )
    s_out = torch.where((kind[neigh] == 2) & (kind[owner] == 0), 1.0, 0.0) - torch.where(
        (kind[owner] == 2) & (kind[neigh] == 0), 1.0, 0.0
    )
    interior = kind == 0
    deg = torch.zeros(n, dtype=torch.float64).index_add(
        0, owner[corr], torch.ones(int(corr.sum()), dtype=torch.float64)
    )
    deg = deg.index_add(0, neigh[corr], torch.ones(int(corr.sum()), dtype=torch.float64))
    outlet_of = torch.full((n,), -1, dtype=torch.long)
    if len(grid["caps"]) > 1:
        centres = torch.stack([c["centre"] for c in grid["caps"][1:]])
        outlet_of = torch.where(kind == 2, torch.cdist(pos, centres).argmin(1), -1)
    return {
        "h": h, "lo": lo, "shape": shape, "idx": idx, "cells": cells, "pos": pos, "nb": nb, "n": n,
        "owner": owner, "neigh": neigh, "dirn": dirn, "fpos": fpos, "face_of": face_of, "fnb": fnb, "ghost": ghost,
        "kind": kind, "fixed": fixed, "corr": corr, "s_in": s_in, "s_out": s_out, "interior": interior,
        "diag": deg / h**2, "periodic": periodic, "phi": phi, "inside": inside, "caps": grid["caps"],
        "outlet_of": outlet_of, "tnb": tnb,
    }  # fmt: skip


# ---------------------------------------------------------------- pressure: matrix assembly and two-level PCG
def pressure_matrix(mesh: dict, block: int = 4) -> dict:
    """Assemble A = −∇² on the interior cells from the face list (OpenFOAM's fvm::laplacian → lduMatrix), as a
    sparse CSR matrix, plus a coarse space: aggregates of block³ voxels with A_c = Pᵀ A P (piecewise constant P)."""
    h, corr, inter = mesh["h"], mesh["corr"], mesh["interior"]
    I = inter.nonzero()[:, 0]  # noqa: E741  (the unknowns: interior cells)
    m = len(I)
    loc = torch.full((mesh["n"],), -1, dtype=torch.long)
    loc[I] = torch.arange(m)
    a_, b_ = loc[mesh["owner"][corr]], loc[mesh["neigh"][corr]]
    diag = torch.zeros(m, dtype=torch.float64)
    rows, cols, vals = [torch.arange(m)], [torch.arange(m)], []
    for a, b in ((a_, b_), (b_, a_)):  # each face couples its two cells symmetrically
        ok = a >= 0
        diag.index_add_(0, a[ok], torch.full((int(ok.sum()),), 1 / h**2, dtype=torch.float64))
        both = ok & (b >= 0)  # a neighbour in an outlet cell has p = 0 and only adds to the diagonal
        (
            rows.append(a[both]),
            cols.append(b[both]),
            vals.append(torch.full((int(both.sum()),), -1 / h**2, dtype=torch.float64)),
        )
    vals.insert(0, diag)
    A = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (m, m)).coalesce()
    c = mesh["cells"][I] // block
    _, agg = torch.unique(c[:, 0] * 10**8 + c[:, 1] * 10**4 + c[:, 2], return_inverse=True)
    na = int(agg.max()) + 1
    # A_c[I, J] = Σ_{a ∈ I, b ∈ J} A_ab: sum the matrix entries by aggregate pairs
    ij = A.indices()
    Ac = (
        torch.zeros(na * na, dtype=torch.float64)
        .index_add_(0, agg[ij[0]] * na + agg[ij[1]], A.values())
        .reshape(na, na)
    )
    return {"A": A.to_sparse_csr(), "I": I, "agg": agg, "Aci": torch.linalg.pinv(Ac), "dinv": 1 / diag, "m": m}


def pcg(A, b, x, M, tol: float, maxit: int = 1000, atol: float = 0.0):
    """Preconditioned conjugate gradients for A x = b (A symmetric positive (semi-)definite)."""
    r = b - A @ x
    z = M(r)
    p, rz = z.clone(), r @ z
    stop = max(tol * b.norm().item(), atol)  # ‖r‖ ≤ max(tol ‖b‖, atol)
    k = 0
    for k in range(maxit):  # noqa: B007  (k is returned)
        if r.norm().item() <= stop:
            break
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x, r = x + alpha * p, r - alpha * Ap
        z = M(r)
        rz, rz_old = r @ z, rz
        p = z + (rz / rz_old) * p
    return x, k


def project(u: torch.Tensor, mesh: dict, dt: float, p0: torch.Tensor, tol: float = 1e-8):
    """Chorin's projection: solve −∇²p = −∇·u*/Δt on the interior cells (p = 0 in outlet cells, ∂p/∂n = 0 on walls
    and the inlet) and correct u = u* − Δt∇p on the correctable faces. Returns (u, p, PCG iterations).

    Preconditioner: Jacobi plus a coarse-grid correction on the voxel aggregates (two-level additive Schwarz),
    M⁻¹r = D⁻¹r + P A_c⁻¹ Pᵀ r, which cuts the iterations about tenfold.
    """
    if "pm" not in mesh:
        mesh["pm"] = pressure_matrix(mesh)
    pm = mesh["pm"]
    dt_ = u.dtype
    if pm.get("dtype") != dt_:
        pm.update(Ad=pm["A"].to(dt_), Acid=pm["Aci"].to(dt_), dinvd=pm["dinv"].to(dt_), dtype=dt_)
    agg, na = pm["agg"], pm["Aci"].shape[0]

    def M(r):
        return pm["dinvd"] * r + (pm["Acid"] @ torch.zeros(na, dtype=dt_).index_add_(0, agg, r))[agg]

    o, nn, corr = mesh["owner"], mesh["neigh"], mesh["corr"]
    b = (-eq.divergence(u, o, nn, mesh["n"], mesh["h"]) / dt)[pm["I"]]  # right-hand side −∇·u*/Δt on the unknowns
    atol = 1e-3 * tol * u.norm().item() / (dt * mesh["h"])  # floor for b ≈ 0 (periodic pipe: div u* is round-off)
    x, iters = pcg(pm["Ad"], b, p0[pm["I"]].to(dt_), M, tol, atol=atol)  # A = −∇²: A p = −∇·u*/Δt ⇔ ∇²p = ∇·u*/Δt
    p = torch.zeros(mesh["n"], dtype=dt_)
    p[pm["I"]] = x
    u = u.clone()
    u[corr] = u[corr] - dt * eq.face_gradient(p, o[corr], nn[corr], mesh["h"])
    return u, p, iters


# ---------------------------------------------------------------- semi-Lagrangian advection on the staggered grid
def sample(field: torch.Tensor, X: torch.Tensor, lo: torch.Tensor, h: float, offset, border: bool) -> torch.Tensor:
    """Trilinear interpolation of dense fields (C, nx, ny, nz) at points X (M, 3); `offset` (3,) shifts the sample
    positions (½ for a staggered component). → (M, C)."""
    shape = torch.tensor(field.shape[1:], dtype=X.dtype)
    g = (X - lo) / h - torch.as_tensor(offset, dtype=X.dtype)
    g = 2 * g / (shape - 1) - 1  # normalised to [−1, 1]
    grid = (
        g.flip(-1).reshape(1, 1, 1, -1, 3).to(field.dtype)
    )  # grid_sample wants (z, y, x) order for (D, H, W) = (x, y, z)
    pad = "border" if border else "zeros"
    out = Fn.grid_sample(field[None], grid, mode="bilinear", padding_mode=pad, align_corners=True)
    return out[0, :, 0, 0].T  # (M, C)


def dense_faces(u: torch.Tensor, mesh: dict) -> torch.Tensor:
    """Scatter face velocities to three dense staggered grids (3, nx, ny, nz); walls and solid stay 0."""
    D = torch.zeros((3, *mesh["shape"]), dtype=u.dtype)
    c = mesh["cells"][mesh["owner"]]
    D[mesh["dirn"], c[:, 0], c[:, 1], c[:, 2]] = u
    return D


def advect(u: torch.Tensor, mesh: dict, dt: float) -> torch.Tensor:
    """u_new(x_f) = u(x_f − Δt v(x_f)): trace each face centre back along the flow and interpolate there."""
    D, lo, h, border = dense_faces(u, mesh), mesh["lo"], mesh["h"], any(mesh["periodic"])
    e = torch.eye(3, dtype=torch.float64)
    uu = torch.cat([u, u.new_zeros(1)])  # index −1 (a wall face) → 0
    v = uu[mesh["tnb"]].mean(-1)  # (F, 3): other components = mean of the 4 surrounding faces
    v[torch.arange(len(u)), mesh["dirn"]] = u
    X = mesh["fpos"] - dt * v.double()
    out = torch.empty_like(u)
    for c in range(3):
        m = mesh["dirn"] == c
        out[m] = sample(D[c : c + 1], X[m], lo, h, 0.5 * e[c], border)[:, 0]
    return out


# ---------------------------------------------------------------- cell-centred velocity, wall shear stress
def cell_velocity(u: torch.Tensor, mesh: dict) -> torch.Tensor:
    """Dense cell-centred velocity (3, nx, ny, nz): mean of the two faces per direction, extended into the first
    solid layers by linear extrapolation through u = 0 on the wall (u_s = u_f φ_s/φ_f), for wall sampling."""
    fo, nb = mesh["face_of"], mesh["nb"]
    uu = torch.cat([u, u.new_zeros(1)])  # index −1 → 0 (a wall face)
    Uc = torch.stack(
        [
            0.5 * (uu[fo[:, d]] + uu[torch.where(nb[:, 2 * d] >= 0, fo[nb[:, 2 * d].clamp(min=0), d], -1)])
            for d in range(3)
        ]
    )  # (3, N)
    dense = torch.zeros((3, *mesh["shape"]), dtype=u.dtype)
    c = mesh["cells"]
    dense[:, c[:, 0], c[:, 1], c[:, 2]] = Uc
    return extend_into_wall(dense, mesh)


def extend_into_wall(dense: torch.Tensor, mesh: dict) -> torch.Tensor:
    """Fill the first solid layers of a dense cell-centred field (3, nx, ny, nz) by linear extrapolation through
    u = 0 on the wall, u_s = φ_s · mean(u_f/φ_f) over the fluid neighbours."""
    phi, inside, h = mesh["phi"].to(dense.dtype), mesh["inside"], mesh["h"]
    w = torch.where(inside, 1 / phi.clamp(min=0.1 * h), 0.0)  # weight u_f/φ_f from fluid neighbours
    num, cnt = torch.zeros_like(dense), torch.zeros_like(phi)
    for a, s in DIRS:
        num += shift(dense * w, a + 1, s, mesh["periodic"][a], 0.0)
        cnt += shift(inside.to(dense.dtype), a, s, mesh["periodic"][a], 0.0)
    ext = phi.clamp(min=-3 * h) * num / cnt.clamp(min=1)
    return torch.where(inside, dense, torch.where(cnt > 0, ext, 0.0))


def streamlines(Uc: torch.Tensor, mesh: dict, seeds: torch.Tensor, n_steps: int = 150, ds: float | None = None):
    """Batched RK4 streamlines of the frozen cell-centred field (3, nx, ny, nz) from seeds (S, 3), parametrised
    by arc length (step ds along u/|u|). Returns points (S, L, 3) with NaN after a line leaves the fluid, and the
    speed |u| (S, L)."""
    lo, h = mesh["lo"], mesh["h"]
    ds = ds or 0.8 * h
    inside = mesh["inside"].to(Uc.dtype)[None]

    def f(x):
        v = sample(Uc, x, lo, h, (0, 0, 0), False).double()
        return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)

    x = seeds.clone()
    alive = torch.ones(len(x), dtype=torch.bool)
    X, S = [x.clone()], [sample(Uc, x, lo, h, (0, 0, 0), False).double().norm(dim=1)]
    for _ in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * ds * k1)
        k3 = f(x + 0.5 * ds * k2)
        k4 = f(x + ds * k3)
        x = x + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        alive &= sample(inside, x, lo, h, (0, 0, 0), False)[:, 0] > 0.5
        X.append(torch.where(alive[:, None], x, float("nan")))
        S.append(sample(Uc, x, lo, h, (0, 0, 0), False).double().norm(dim=1))
    return torch.stack(X, 1), torch.stack(S, 1)


def wss_at(Uc: torch.Tensor, V: torch.Tensor, N: torch.Tensor, mesh: dict, mu: float) -> torch.Tensor:
    """Wall shear stress vectors (P, 3) at wall points V with outward normals N, from samples at h and 2h inside."""
    m, h, lo, border = -N, mesh["h"], mesh["lo"], any(mesh["periodic"])
    u1 = sample(Uc, V + h * m, lo, h, (0, 0, 0), border).double()
    u2 = sample(Uc, V + 2 * h * m, lo, h, (0, 0, 0), border).double()
    return eq.wall_shear(u1, u2, m, h, mu)


# ---------------------------------------------------------------- one time step and the pulsatile run
def inlet_setup(mesh: dict, Qhat: torch.Tensor, omega: float, nu: float) -> dict:
    """Womersley shape per harmonic on the prescribed inlet faces, normalised so that the discrete inflow
    Σ_f s_f u_f h² equals Q(t) exactly. Faces: (Fi,), shapes: (H+1, Fi) complex."""
    cap = mesh["caps"][0]
    f = mesh["fixed"].nonzero()[:, 0]
    rel = mesh["fpos"][f] - cap["centre"]
    r = (rel - (rel @ cap["normal"])[:, None] * cap["normal"]).norm(dim=1)
    comp = -cap["normal"][mesh["dirn"][f]]  # inflow direction component along each face normal
    S = eq.womersley_shapes(r, len(Qhat) - 1, omega, cap["radius"], nu) * comp  # (H+1, Fi)
    flux = (S * mesh["s_in"][f]).sum(1) * mesh["h"] ** 2  # discrete flux of each unit shape
    return {"faces": f, "S": S / flux[:, None], "Qhat": Qhat, "omega": omega}


def inlet_values(inl: dict, t: float) -> torch.Tensor:
    n = torch.arange(len(inl["Qhat"]), dtype=torch.float64)
    phase = torch.exp(1j * inl["omega"] * t * n)
    return torch.einsum("h,h,hf->f", phase, inl["Qhat"].to(torch.complex128), inl["S"]).real


def step(u, p, mesh, dt, nu, t, inl=None, body=None, tol=1e-8):
    r"""u* = advect(u) + Δt(ν∇²u + g); inlet faces prescribed; then project. Returns (u, p, CG iterations)."""
    us = advect(u, mesh, dt) + dt * nu * eq.face_laplacian(u, mesh["fnb"], mesh["ghost"].to(u.dtype), mesh["h"])
    if body is not None:
        us = us + dt * body(t)[mesh["dirn"]].to(u.dtype)
    if inl is not None:
        us[inl["faces"]] = inlet_values(inl, t + dt).to(u.dtype)
    return project(us, mesh, dt, p, tol)


def fourier(Q: torch.Tensor, n_harm: int) -> torch.Tensor:
    """Q(t) = Re Σ_n Q̂_n e^{inωt} from K equally spaced samples over one period (Q̂_0 = mean)."""
    X = torch.fft.fft(Q.to(torch.complex128)) / len(Q)
    X[1:] *= 2
    return X[: n_harm + 1]


def flows(u: torch.Tensor, mesh: dict) -> tuple[float, torch.Tensor]:
    """Inflow Σ s_in u h² and the outflow through each outlet cap (m³/s)."""
    h2 = mesh["h"] ** 2
    q_in = (mesh["s_in"] * u).sum().item() * h2
    which = mesh["outlet_of"][torch.where(mesh["kind"][mesh["neigh"]] == 2, mesh["neigh"], mesh["owner"])]
    q_out = torch.zeros(len(mesh["caps"]) - 1, dtype=torch.float64).index_add(
        0, which.clamp(min=0), (mesh["s_out"] * u).double() * h2
    )
    return q_in, q_out


def simulate(mesh, waveform, cycles, dt, nu, rho, V, N, n_frames=48, n_wss=50, n_particles=2000, slice_axis=1,
             slice_at=None, seed=0, dtype=torch.float32, probe=None, n_stream=60):  # fmt: skip
    """Pulsatile run. Records (in the last cycle) WSS at the wall vertices, a velocity-magnitude slice, tracer
    particles, streamlines, inflow/outflow and CG iterations; with `probe` = {"times": absolute times, "points":
    (S, 3) in m} also |u| at those points (trilinear) and |WSS| at V at those times (OpenFOAM comparison)."""
    t_wave, Q = waveform
    T = float(t_wave[-1] + (t_wave[1] - t_wave[0]))
    omega = 2 * math.pi / T
    inl = inlet_setup(mesh, fourier(Q, 12), omega, nu)
    n_steps = int(round(cycles * T / dt))
    last = n_steps - int(round(T / dt))
    frame_steps = set(last + round(k * (n_steps - last - 1) / (n_frames - 1)) for k in range(n_frames))
    wss_steps = set(last + round(k * (n_steps - last - 1) / (n_wss - 1)) for k in range(n_wss))
    u = torch.zeros(len(mesh["owner"]), dtype=dtype)
    u[inl["faces"]] = inlet_values(inl, 0.0).to(dtype)
    p = torch.zeros(mesh["n"], dtype=dtype)
    gen = torch.Generator().manual_seed(seed)
    cap = mesh["caps"][0]
    ex = torch.linalg.cross(cap["normal"], torch.tensor([0.3, 0.5, 0.8], dtype=torch.float64))
    ex = ex / ex.norm()
    ey = torch.linalg.cross(cap["normal"], ex)

    def spawn(k):  # k points on the inlet disc, 2–12 mm downstream of the cap plane
        rr = 0.8 * cap["radius"] * torch.sqrt(torch.rand(k, generator=gen, dtype=torch.float64))
        th = 2 * math.pi * torch.rand(k, generator=gen, dtype=torch.float64)
        depth = 2e-3 + 10e-3 * torch.rand(k, generator=gen, dtype=torch.float64)
        return (
            cap["centre"]
            - depth[:, None] * cap["normal"]
            + rr[:, None] * (torch.cos(th)[:, None] * ex + torch.sin(th)[:, None] * ey)
        )

    P = spawn(n_particles)
    age = torch.zeros(n_particles, dtype=torch.float64)  # time since each tracer entered (s)
    inside_f = mesh["inside"].to(dtype)[None]
    rec = {
        "t": [],
        "Q_in": [],
        "Q_out": [],
        "wss": [],
        "wss_t": [],
        "slice": [],
        "P": [],
        "iters": [],
        "dp": [],
        "stream": [],
        "stream_speed": [],
        "probe_u": [],
        "probe_wss": [],
        "probe_t": [],
        "P_age": [],
        "P_speed": [],
    }
    gs = torch.Generator().manual_seed(seed + 1)
    rr = 0.85 * cap["radius"] * torch.sqrt(torch.rand(n_stream, generator=gs, dtype=torch.float64))
    th = 2 * math.pi * torch.rand(n_stream, generator=gs, dtype=torch.float64)
    stream_seeds = (
        cap["centre"]
        - 3 * mesh["h"] * cap["normal"]
        + rr[:, None] * (torch.cos(th)[:, None] * ex + torch.sin(th)[:, None] * ey)
    )
    s_in = (mesh["pos"] - cap["centre"]) @ cap["normal"]
    near_in = (mesh["kind"] == 0) & (s_in > -4 * mesh["h"])  # first interior layers after the inlet
    sidx = slice_at if slice_at is not None else mesh["shape"][slice_axis] // 2
    t = 0.0
    for k in steps(n_steps, "time steps"):
        u, p, it = step(u, p, mesh, dt, nu, t, inl=inl, tol=1e-4 if dtype == torch.float32 else 1e-9)
        t += dt
        rec["iters"].append(it)
        if k in wss_steps or k in frame_steps:
            Ucp = cell_velocity(u, mesh)
        # tracers: forward Euler with the staggered velocity; respawn when they leave the fluid or reach an outlet
        D = dense_faces(u, mesh)
        e = torch.eye(3, dtype=torch.float64)
        vP = torch.cat([sample(D[c : c + 1], P, mesh["lo"], mesh["h"], 0.5 * e[c], False) for c in range(3)], 1)
        P = P + dt * vP.double()
        age += dt
        out = sample(inside_f, P, mesh["lo"], mesh["h"], (0, 0, 0), False)[:, 0] < 0.5
        for c in mesh["caps"][1:]:
            out |= (P - c["centre"]) @ c["normal"] > -mesh["h"]
        if out.any():
            P[out] = spawn(int(out.sum()))
            age[out] = 0.0
        if k in wss_steps:
            rec["wss"].append(wss_at(Ucp, V, N, mesh, rho * nu).float())
            rec["wss_t"].append(t)
        if probe is not None and (probe["times"] - t).abs().min() < 0.5 * dt:
            Upr = cell_velocity(u, mesh)
            rec["probe_u"].append(
                sample(Upr, probe["points"], mesh["lo"], mesh["h"], (0, 0, 0), False).norm(dim=1).double()
            )
            rec["probe_wss"].append(wss_at(Upr, V, N, mesh, rho * nu).norm(dim=1))
            rec["probe_t"].append(t)
        if k in frame_steps:
            sl_x, sl_s = streamlines(Ucp, mesh, stream_seeds)
            rec["stream"].append(sl_x.float()), rec["stream_speed"].append(sl_s.float())
            q_in, q_out = flows(u, mesh)
            rec["t"].append(t), rec["Q_in"].append(q_in), rec["Q_out"].append(q_out)
            rec["slice"].append(Ucp.norm(dim=0).select(slice_axis, sidx).clone())
            rec["P"].append(P.clone())
            rec["P_age"].append(age.clone()), rec["P_speed"].append(vP.double().norm(dim=1))
            rec["dp"].append(rho * p[near_in].mean().item())  # inlet-to-outlet pressure drop (outlets: p = 0)
    rec["p"] = p
    rec["u"] = u
    rec["T"] = T
    rec["wss"] = torch.stack(rec["wss"]).double()
    rec["slice"] = torch.stack(rec["slice"])
    rec["P"] = torch.stack(rec["P"])
    rec["Q_out"] = torch.stack(rec["Q_out"])
    rec["slice_index"] = sidx
    rec["stream"], rec["stream_speed"] = torch.stack(rec["stream"]), torch.stack(rec["stream_speed"])
    rec["P_age"], rec["P_speed"] = torch.stack(rec["P_age"]), torch.stack(rec["P_speed"])
    if rec["probe_u"]:
        rec["probe_u"], rec["probe_wss"] = torch.stack(rec["probe_u"]), torch.stack(rec["probe_wss"]).double()
    return rec


def pipe_run(grid, mesh, G0, G1, omega, t_end, dt, nu, rho=1060.0, n_rec=None):
    """Periodic pipe driven by −∂_z p/ρ = G0 + G1 sin(ωt): axis velocity, half-radius velocity, mean WSS over time."""
    body = lambda t: torch.tensor([0.0, 0.0, G0 + G1 * math.sin(omega * t)], dtype=torch.float64)  # noqa: E731
    u = torch.zeros(len(mesh["owner"]), dtype=torch.float64)
    p = torch.zeros(mesh["n"], dtype=torch.float64)
    n_steps = int(round(t_end / dt))
    every = 1 if n_rec is None else max(1, n_steps // n_rec)
    r = (mesh["pos"][:, :2]).norm(dim=1)
    axis = r < 1e-12
    half = (r - 0.5 * grid["phi"].max().item()).abs() < 0.5 * mesh["h"]  # cells near r = R/2
    out = {"t": [], "u_axis": [], "u_half": [], "wss": [], "Q": []}
    t = 0.0
    for k in range(n_steps):
        u, p, _ = step(u, p, mesh, dt, nu, t, body=body)
        t += dt
        if (k + 1) % every == 0:
            Uc = cell_velocity(u, mesh)
            uz = Uc[2][tuple(mesh["cells"].T)]
            tau = wss_at(Uc, grid["V"], grid["N"], mesh, rho * nu)
            (
                out["t"].append(t),
                out["u_axis"].append(uz[axis].mean().item()),
                out["u_half"].append(uz[half].mean().item()),
            )
            out["wss"].append(tau[:, 2].mean().item())
            out["Q"].append((u * (mesh["dirn"] == 2)).sum().item() * mesh["h"] ** 2 / mesh["shape"][2])
    out = {k: torch.tensor(v, dtype=torch.float64) for k, v in out.items()}
    out["u"] = u
    return out


# ---------------------------------------------------------------- centreline (tensor version of vmtkcenterlines)
def medial_graph(grid: dict, power: float = 2.0):
    """Fluid cells as graph nodes, 26-neighbour edges with eikonal travel times ℓ/R^p (R = φ, the inscribed radius)."""
    inside, h = grid["inside"], grid["h"]
    shape = inside.shape
    n = int(inside.sum())
    idx = torch.full(shape, -1, dtype=torch.long)
    idx[inside] = torch.arange(n)
    R = grid["phi"][inside].clamp(min=0.05 * h)  # (N,)
    src, dst, w = [], [], []
    for off in torch.cartesian_prod(*[torch.tensor([-1, 0, 1])] * 3):
        if (off == 0).all():
            continue
        nbr = idx
        for a in range(3):
            if off[a] != 0:
                nbr = shift(nbr, a, int(off[a]), False, -1)
        j = nbr[inside]
        ok = j >= 0
        i = torch.arange(n)[ok]
        src.append(i), dst.append(j[ok])
        w.append(
            eq.edge_cost(
                h * off.double().norm() * torch.ones(len(i), dtype=torch.float64), 0.5 * (R[i] + R[j[ok]]), power
            )
        )
    pos = grid["lo"] + h * inside.nonzero().double()
    return {"src": torch.cat(src), "dst": torch.cat(dst), "w": torch.cat(w), "R": R, "pos": pos, "n": n}


def eikonal(graph: dict, sources: torch.Tensor, max_iter: int = 5000):
    """Arrival times T from the source nodes by iterated min-plus relaxation (a tensor stand-in for fast marching),
    and the predecessor of every node on its shortest path."""
    T = torch.full((graph["n"],), float("inf"), dtype=torch.float64)
    T[sources] = 0.0
    for it in range(max_iter):  # noqa: B007  (it is returned)
        Tn = eq.relax(T, graph["src"], graph["dst"], graph["w"])
        if torch.equal(Tn, T):
            break
        T = Tn
    cand = T[graph["src"]] + graph["w"]
    best = (cand <= T[graph["dst"]] * (1 + 1e-12)) & torch.isfinite(cand)
    pred = torch.full((graph["n"],), -1, dtype=torch.long)
    pred[graph["dst"][best]] = graph["src"][best]
    pred[sources] = -1
    return T, pred, it


def backtrack(pred: torch.Tensor, seeds: torch.Tensor, max_len: int = 100000) -> list:
    """Steepest descent of T from all outlet seeds at once (follow the predecessors to the inlet)."""
    cur = seeds.clone()
    paths = [cur.clone()]
    for _ in range(max_len):
        nxt = torch.where(cur >= 0, pred[cur.clamp(min=0)], -1)
        if (nxt < 0).all():
            break
        cur = nxt
        paths.append(cur.clone())
    P = torch.stack(paths, 1)  # (K, L) node ids, −1 after reaching the inlet
    return [row[row >= 0].flip(0) for row in P]  # inlet → outlet


def smooth(x: torch.Tensor, half: int) -> torch.Tensor:
    """Moving average along a polyline (L, C), ends kept by edge padding."""
    if len(x) < 3:
        return x
    k = torch.ones(1, 1, 2 * half + 1, dtype=x.dtype) / (2 * half + 1)
    xp = Fn.pad(x.T[None], (half, half), mode="replicate")
    return Fn.conv1d(xp.reshape(-1, 1, xp.shape[-1]), k).reshape(x.shape[1], -1).T


def centrelines(grid: dict, inlet_nodes: torch.Tensor, outlet_seeds: torch.Tensor, power: float = 2.0, half: int = 3):
    """Centrelines from the inlet to every outlet: positions (L, 3), radius (L,), abscissa (L,) per line, and the
    bifurcation point where the first two lines separate."""
    g = medial_graph(grid, power)
    T, pred, iters = eikonal(g, inlet_nodes)
    lines = []
    for path in backtrack(pred, outlet_seeds):
        x = smooth(g["pos"][path], half)
        R = smooth(g["R"][path][:, None], half)[:, 0]
        s = torch.cat([x.new_zeros(1), (x[1:] - x[:-1]).norm(dim=1).cumsum(0)])
        lines.append({"nodes": path, "x": x, "R": R, "s": s})
    bif = None
    if len(lines) >= 2:
        a, b = lines[0]["nodes"], lines[1]["nodes"]
        m = min(len(a), len(b))
        same = (a[:m] == b[:m]).long()
        j = int(same.cumprod(0).sum()) - 1  # last shared node
        bif = {"x": lines[0]["x"][max(j, 0)], "s": lines[0]["s"][max(j, 0)].item()}
    return lines, bif, {"iters": iters, "graph": g}


def cap_nodes(grid: dict, cap: dict) -> torch.Tensor:
    """Graph nodes in the slab next to a cap (within 1.5 voxels of its plane)."""
    pos = grid["lo"] + grid["h"] * grid["inside"].nonzero().double()
    s = (pos - cap["centre"]) @ cap["normal"]
    lat = (pos - cap["centre"] - s[:, None] * cap["normal"]).norm(dim=1)
    return ((s > -1.5 * grid["h"]) & (lat < cap["radius"])).nonzero()[:, 0]


def outlet_seed(grid: dict, cap: dict) -> int:
    """The most central node of an outlet slab (largest inscribed radius)."""
    nodes = cap_nodes(grid, cap)
    return int(nodes[grid["phi"][grid["inside"]][nodes].argmax()])


# ---------------------------------------------------------------- residuals against the precomputed references
def harmonic(y: torch.Tensor, t: torch.Tensor, omega: float) -> complex:
    """First Fourier coefficient c with y ≈ mean + Re(c e^{iωt}) over whole periods."""
    return complex((y * torch.exp(-1j * omega * t)).mean().item()) * 2


def openfoam_residuals(ours: dict, ref: dict, t0: float, t1: float, rho: float) -> dict:
    """Our pipe run minus OpenFOAM's on the OpenFOAM sample times in (t0, t1]: relative L2 and L∞ errors of the
    axis velocity, WSS and flow rate, and the amplitude/phase error of the axis velocity's first harmonic."""
    sel = (ref["t"] > t0 + 1e-9) & (ref["t"] <= t1 + 1e-9)
    t = ref["t"][sel]
    k = torch.searchsorted(ours["t"], t - 1e-9)
    out = {"t": t}
    for name, scale in (("u_axis", 1.0), ("wss", 1 / rho), ("Q", 1.0)):
        a, b = ours[name][k] * scale, ref[name][sel]
        out[name] = (a, b)
        out[name + "_L2"] = ((a - b).norm() / b.norm()).item()
        out[name + "_Linf"] = ((a - b).abs().max() / b.abs().max()).item()
    omega = 2 * math.pi / (t1 - t0)
    ca, cb = harmonic(out["u_axis"][0], t, omega), harmonic(out["u_axis"][1], t, omega)
    out["amp_err"] = abs(ca) / abs(cb) - 1
    out["phase_err"] = math.remainder(math.atan2(ca.imag, ca.real) - math.atan2(cb.imag, cb.real), 2 * math.pi)
    return out


def vmtk_residuals(lines: list, P: torch.Tensor, R: torch.Tensor, lid: torch.Tensor) -> list:
    """For each VMTK line: distance of its points to our polyline and |ΔR| at the nearest of our points (m)."""
    res = []
    for k in lid.unique().tolist():  # VMTK lines that exist (a degenerate one may have been left out)
        line = lines[k]
        ref, Rr = P[lid == k], R[lid == k]
        D = torch.cdist(ref, line["x"])
        d, j = D.min(1)
        dr = (line["R"][j] - Rr).abs()
        s_ref = torch.cat([ref.new_zeros(1), (ref[1:] - ref[:-1]).norm(dim=1).cumsum(0)])
        res.append({"k": k, "d": d, "dR": dr, "s_ref": s_ref, "R_ref": Rr})
    return res
