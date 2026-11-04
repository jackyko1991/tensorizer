"""Solvers: check the known embeddings, compute GR curvature by autograd, measure fold exponents, build the fold toy."""

import math

import torch

from . import equation, tensors


def _rel_err(g: torch.Tensor, target: torch.Tensor) -> float:
    """Largest |g - target| relative to the largest component of target, point by point."""
    scale = target.abs().flatten(1).amax(1) if target.dim() > 1 else target.abs()
    return (
        ((g - target).abs().flatten(1).amax(1) / scale).max().item()
        if target.dim() > 1
        else (((g - target).abs() / scale).max().item())
    )


def embeddings(rs: float, device) -> dict:
    """Induced metrics of Flamm's and Fronsdal's embeddings against the target metrics; Jacobian rank along r."""
    eta3 = torch.eye(3, dtype=torch.float64, device=device)
    xf = tensors.flamm_points(rs, device)
    gf = equation.induced_metric(lambda q: equation.flamm_embedding(q, rs), xf, eta3)
    r = xf[:, 0]
    tf = torch.diag_embed(torch.stack([1 / (1 - rs / r), r**2], 1))
    errs = {"Flamm (r, phi)": _rel_err(gf, tf)}
    for region in ("exterior", "interior"):
        x = tensors.spacetime_points(rs, region, device)
        g = equation.induced_metric(lambda q: equation.fronsdal_embedding(q, rs), x, equation.ETA6)
        errs[f"Fronsdal (t, r), {region}"] = _rel_err(g, torch.vmap(lambda q: equation.schwarzschild_metric(q, rs))(x))
    xk = tensors.kruskal_points(rs, device)
    gk = equation.induced_metric(lambda q: equation.fronsdal_kruskal(q, rs), xk, equation.ETA6)
    errs["Fronsdal (T, X), across r_s"] = _rel_err(gk, torch.vmap(lambda q: equation.kruskal_metric(q, rs))(xk))

    # smallest singular value of dz/dx along r: Schwarzschild coordinates vs Kruskal coordinates
    r_line = tensors.radius_line(rs, 161, device, r_min=0.05, r_max=6.0)
    r_line = r_line[(r_line - rs).abs() > 1e-6]
    n = len(r_line)
    ones = torch.ones(n, dtype=torch.float64, device=device)
    xs = torch.stack([0.3 * ones, r_line, ones, 0.5 * ones], 1)
    sv_s = equation.jacobian_singular_values(lambda q: equation.fronsdal_embedding(q, rs), xs).amin(1)
    T, X = tensors.kruskal_from_schwarzschild(r_line, 0.3 * ones, rs)
    xk_line = torch.stack([T, X, ones, 0.5 * ones], 1)
    sv_k = equation.jacobian_singular_values(lambda q: equation.fronsdal_kruskal(q, rs), xk_line).amin(1)
    z2 = equation.fronsdal_z2(r_line, rs)

    # Flamm paraboloid mesh for the 3D view (both sheets of the Einstein-Rosen bridge)
    rr = rs * (1 + torch.linspace(0, 1, 14, dtype=torch.float64) ** 2 * 5)
    ph = torch.linspace(0, 2 * math.pi, 25, dtype=torch.float64)
    R, P = torch.meshgrid(rr, ph, indexing="ij")
    mesh = (R * torch.cos(P), R * torch.sin(P), equation.flamm_height(R, rs))
    return {"errors": errs, "r_line": r_line, "sv_schw": sv_s, "sv_krus": sv_k, "z2": z2, "flamm_mesh": mesh,
            "sv_krus_min": sv_k.min().item()}  # fmt: skip


def curvature(rs: float, R: float, device) -> dict:
    """Kretschmann scalar (autograd vs 12 r_s^2/r^6), static tidal eigenvalues, and the fold-shell caricature."""
    r = tensors.radius_line(rs, 121, device, r_min=0.02, r_max=6.0)
    r = r[(r - rs).abs() > 1e-3]
    ones = torch.ones_like(r)
    x = torch.stack([0.0 * ones, r, 1.1 * ones, 0.4 * ones], 1)
    K = equation.kretschmann(lambda q: equation.schwarzschild_metric(q, rs), x)
    xe = x[r > rs]
    tid = equation.static_tidal_eigenvalues(lambda q: equation.schwarzschild_metric(q, rs), xe)
    toy = equation.fold_shell_field(r, R, rs)
    near = (R - r > 1e-3 * R) & (R - r < 5e-2 * R)
    r_fine = R * (1 - torch.logspace(-7, -3, 40, dtype=torch.float64, device=device))
    fine = equation.fold_shell_field(r_fine, R, rs)
    return {"r": r, "K": K, "K_exact": 12 * rs**2 / r**6, "K_err": _rel_err(K, 12 * rs**2 / r**6),
            "r_ext": xe[:, 1], "tidal": tid, "toy": toy,
            "slope_density": equation.loglog_slope(R - r_fine, fine["density"]),
            "slope_tidal": equation.loglog_slope(R - r_fine, fine["tidal_radial"].abs()),
            "has_near": bool(near.any()), "tidal_centre": toy["tidal_radial"][0].item(),
            "proxy_K_centre": toy["proxy_K"][0].item()}  # fmt: skip


def folds(a: float, D: int, samples: int, n_frames: int, generator, device) -> dict:
    """Catastrophe exponents, sphere-shadow exponents for every sheet dimension D allows, and the rotating sheet."""
    n = torch.arange(2, 5)  # fold, cusp, swallowtail-type A3
    cat = [equation.catastrophe_density(int(k), samples) for k in n]
    alpha_cat = torch.tensor([equation.loglog_slope(c, rho) for c, rho in cat], dtype=torch.float64)
    d = 3
    k = torch.arange(d, min(D - 1, d + 3) + 1)  # S^k needs k + 1 <= D axes
    alpha_sph = equation.sphere_rim_exponents(k, d, D, samples, generator)
    u_sph = [equation.sphere_rim_samples(int(kk), d, D, samples, generator) for kk in k]
    sph = [equation._log_histogram(u, 1e-3, 10**-1.5, 10, samples) for u in u_sph]

    # the wavy sheet z = a sin u tilted by alpha and seen along z: folds form once a tan(alpha) > 1
    alpha = torch.linspace(0.08, 1.3, n_frames, dtype=torch.float64, device=device)
    u = torch.linspace(-2 * math.pi, 2 * math.pi, samples, dtype=torch.float64, device=device)
    x = equation.sheet_projection(u, alpha, a)  # (F, U)
    L = 2 * math.pi + a  # |x| <= 2 pi + a for every tilt
    edges = torch.linspace(-L, L, 181, dtype=torch.float64, device=device)
    idx = torch.bucketize(x, edges).clamp(1, len(edges) - 1) - 1  # (F, U) bin of every projected point
    counts = torch.zeros(n_frames, len(edges) - 1, dtype=torch.float64, device=device)
    counts.scatter_add_(1, idx, torch.ones_like(x))
    density = counts / (samples * edges.diff()) * (4 * math.pi)  # per unit length of u
    n_fold = equation.sheet_fold_count(alpha, a)
    cos_u = (-1 / (a * torch.tan(alpha))).clamp(-1, 1)  # fold points: cos u = -1/(a tan alpha)
    u_f = torch.acos(cos_u)
    uf = torch.stack([u_f, -u_f, u_f - 2 * math.pi, 2 * math.pi - u_f], 1)  # (F, 4) inside the window
    x_fold = uf * torch.cos(alpha)[:, None] + a * torch.sin(uf) * torch.sin(alpha)[:, None]  # (F, 4)
    return {"n": n, "cat": cat, "alpha_cat": alpha_cat, "k": k, "d": d, "sph": sph, "alpha_sph": alpha_sph,
            "alpha": alpha, "edges": edges, "density": density, "n_fold": n_fold, "x_fold": x_fold, "u_fold": uf,
            "alpha_c": math.atan(1 / a)}  # fmt: skip
