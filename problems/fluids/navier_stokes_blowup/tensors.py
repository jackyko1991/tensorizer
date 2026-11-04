"""Build the concrete tensors: the periodic grid, the wavevectors with the 2/3 dealiasing mask, and the Taylor–Green
initial velocity fields. Everything lives on the torus T³ = [0, 2π)³ with N points per side."""

import math

import torch


def grid(n: int, device: torch.device) -> torch.Tensor:
    """x_j = 2π j / N, j = 0 … N−1: (N,)"""
    return torch.arange(n, dtype=torch.float64, device=device) * (2 * math.pi / n)


def wavenumbers(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer wavevectors of the real FFT: k (3, N, N, M) with M = N/2 + 1, |k|² (N, N, M), and the 2/3-rule mask
    (N, N, M) that zeroes every mode with some |k_i| > N/3 (removes aliasing from the quadratic term)."""
    k_full = torch.fft.fftfreq(n, 1 / n).to(device=device, dtype=torch.float64)  # 0, 1, …, N/2−1, −N/2, …, −1
    k_half = torch.fft.rfftfreq(n, 1 / n).to(device=device, dtype=torch.float64)  # 0 … N/2
    kx, ky, kz = torch.meshgrid(k_full, k_full, k_half, indexing="ij")
    k = torch.stack([kx, ky, kz])  # (3, N, N, M)
    k2 = torch.einsum("ixyz,ixyz->xyz", k, k)  # |k|², repeated i summed
    cut = n / 3
    mask = (kx.abs() < cut) & (ky.abs() < cut) & (kz.abs() < cut)  # (N, N, M) bool
    return k, k2, mask


def taylor_green(n: int, device: torch.device) -> torch.Tensor:
    """3D Taylor–Green vortex u = (sin x cos y cos z, −cos x sin y cos z, 0): (3, N, N, N). E₀ = 1/8."""
    x = grid(n, device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    return torch.stack(
        [torch.sin(X) * torch.cos(Y) * torch.cos(Z), -torch.cos(X) * torch.sin(Y) * torch.cos(Z), torch.zeros_like(X)]
    )


def taylor_green_2d(n: int, device: torch.device) -> torch.Tensor:
    """2D Taylor–Green vortex u = (sin x cos y, −cos x sin y, 0), independent of z: an exact decaying solution."""
    x = grid(n, device)
    X, Y, _ = torch.meshgrid(x, x, x, indexing="ij")
    return torch.stack([torch.sin(X) * torch.cos(Y), -torch.cos(X) * torch.sin(Y), torch.zeros_like(X)])


def surface_nets(field: torch.Tensor, level: float, periodic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Iso-surface {field = level} by naive surface nets, for all cells at once.

    field (Nx, Ny, Nz) on integer grid points. Every cell (cube of 8 neighbouring points) whose corners straddle the
    level gets one vertex: the mean of the points where the level crosses its 12 edges (linear interpolation along
    each edge). Every grid edge that crosses the level gets a quad joining the 4 cells around it, split into two
    triangles oriented away from the inside (field > level). periodic=True wraps all three directions (a torus).
    Returns vertices (V, 3) in grid units and triangles (F, 3) of vertex indices."""
    dims = torch.tensor(field.shape, device=field.device)
    cells = tuple(int(n) if periodic else int(n) - 1 for n in field.shape)

    def at(offset):  # field at cell corner (i, j, k) + offset, for every cell
        f = torch.roll(field, shifts=tuple(-int(o) for o in offset), dims=(0, 1, 2)) if any(offset) else field
        return f[: cells[0], : cells[1], : cells[2]]

    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    values = {c: at(c) - level for c in corners}
    edges = [(p, q) for p in corners for q in corners if p < q and sum(abs(x - y) for x, y in zip(p, q)) == 1]
    total = torch.zeros(*cells, 3, dtype=field.dtype, device=field.device)
    count = torch.zeros(cells, dtype=field.dtype, device=field.device)
    for p, q in edges:  # the 12 edges of a cell (a loop over edge types, not over cells)
        a, b = values[p], values[q]
        cross = (a > 0) != (b > 0)
        t = torch.where(cross, a / (a - b).masked_fill(~cross, 1.0), torch.zeros_like(a))
        point = torch.tensor(p, dtype=field.dtype, device=field.device) + t[..., None] * torch.tensor(
            [qq - pp for pp, qq in zip(p, q)], dtype=field.dtype, device=field.device)  # fmt: skip
        total += point * cross[..., None]
        count += cross
    active = count > 0
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=field.device) for n in cells], indexing="ij"), -1)
    verts = (grid + total / count.clamp(min=1)[..., None])[active]  # (V, 3)
    index = torch.full(cells, -1, dtype=torch.long, device=field.device)
    index[active] = torch.arange(int(active.sum()), device=field.device)

    inside = field > level
    faces = []
    for axis in range(3):  # grid edges along each axis
        b_ax, c_ax = [d for d in range(3) if d != axis]
        step = [0, 0, 0]
        step[axis] = 1
        nxt = torch.roll(inside, shifts=-1, dims=axis)
        crossing = inside != nxt
        if not periodic:  # the edge and all four cells around it must lie inside the box
            valid = torch.zeros_like(crossing)
            sl = [slice(None)] * 3
            sl[axis] = slice(0, int(dims[axis]) - 1)
            sl[b_ax] = slice(1, int(dims[b_ax]) - 1)
            sl[c_ax] = slice(1, int(dims[c_ax]) - 1)
            valid[tuple(sl)] = True
            crossing &= valid
        p = crossing.nonzero()  # (E, 3) edge start points
        flip = ~inside[p[:, 0], p[:, 1], p[:, 2]] ^ (axis == 1)  # (axis, b, c) is left-handed for the y axis

        around = []  # vertex index of the cells (p − db e_b − dc e_c) sharing this edge
        for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
            q = p.clone()
            q[:, b_ax] -= db
            q[:, c_ax] -= dc
            q %= torch.tensor(cells, device=field.device)
            around.append(index[q[:, 0], q[:, 1], q[:, 2]])
        v00, v10, v11, v01 = around
        quad = torch.stack([v00, v10, v11, v01], 1)
        quad = torch.where(flip[:, None], quad.flip(1), quad)
        faces.append(torch.cat([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
    faces = torch.cat(faces)
    return verts, faces[(faces >= 0).all(1)]
