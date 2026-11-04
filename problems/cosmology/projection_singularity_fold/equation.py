"""Geometry -> tensors: embeddings and their induced metrics, curvature by autograd, fold and cusp exponents.

Shapes in comments. Summed indices: the embedding index A in g_ij = eta_AB dz^A/dx^i dz^B/dx^j, the dummy index of
the Christoffel symbols and the Riemann tensor (d, e, f, g, h below), and the Gauss-Legendre node index in the
Fronsdal integral. Units: r_s = 2GM/c^2 sets lengths, G = c = 1.
"""

import math

import numpy as np
import torch
from torch.func import jacfwd, vmap

ETA6 = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64))  # flat R^(1,5)
_XI, _W = (torch.tensor(v, dtype=torch.float64) for v in np.polynomial.legendre.leggauss(64))  # nodes on [-1, 1]

# ---------- induced metric of an embedding ----------


def induced_metric(embed, x: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """(B, n, n) g_ij = eta_AB dz^A/dx^i dz^B/dx^j for points x (B, n); embed maps (n,) -> (N,). Autograd Jacobian."""
    J = vmap(jacfwd(embed))(x)  # (B, N, n) dz^A / dx^i
    return torch.einsum("bAi,AC,bCj->bij", J, eta.to(x), J)  # repeated A, C => summed


def jacobian_singular_values(embed, x: torch.Tensor) -> torch.Tensor:
    """(B, n) singular values of dz/dx: a fold of the map is where the smallest one reaches 0 (rank drops)."""
    return torch.linalg.svdvals(vmap(jacfwd(embed))(x))


# ---------- (a) Flamm's paraboloid ----------


def flamm_height(r: torch.Tensor, rs: float) -> torch.Tensor:
    """z(r) = 2 sqrt(r_s (r - r_s)): the equatorial slice t = const, theta = pi/2 drawn in flat R^3 (Flamm 1916)."""
    return 2 * torch.sqrt(rs * (r - rs))


def flamm_embedding(q: torch.Tensor, rs: float) -> torch.Tensor:
    """(2,) = (r, phi) -> (3,) point of the paraboloid of revolution in flat R^3."""
    r, phi = q[0], q[1]
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), flamm_height(r, rs)])


# ---------- (b) Fronsdal's embedding of the whole Schwarzschild spacetime in R^(1,5) ----------


def schwarzschild_metric(x: torch.Tensor, rs: float) -> torch.Tensor:
    """(4,) = (t, r, theta, phi) -> (4, 4) diag(-f, 1/f, r^2, r^2 sin^2 theta), f = 1 - r_s/r (inside too)."""
    r, th = x[1], x[2]
    f = 1 - rs / r
    return torch.diag(torch.stack([-f, 1 / f, r**2, (r * torch.sin(th)) ** 2]))


def fronsdal_z2(r: torch.Tensor, rs: float) -> torch.Tensor:
    """z2(r) = int_{r_s}^r sqrt(h(s)) ds, h = (r_s/s)(1 + r_s/s + r_s^2/s^2), by 64-point Gauss-Legendre in ln s.

    The integrand in l = ln s is smooth even as s -> 0, and the quadrature is differentiable in r.
    """
    l0, l1 = math.log(rs), torch.log(r)[..., None]  # (..., 1)
    ell = (l0 + l1) / 2 + (l1 - l0) / 2 * _XI.to(r)  # (..., G)
    s = torch.exp(ell)
    u = rs / s
    integrand = torch.sqrt(u * (1 + u + u**2)) * s  # sqrt(h) ds/dl
    return (l1[..., 0] - l0) / 2 * (integrand * _W.to(r)).sum(-1)  # repeated node index => summed


def _sphere(r, th, ph):
    return [r * torch.sin(th) * torch.cos(ph), r * torch.sin(th) * torch.sin(ph), r * torch.cos(th)]


def fronsdal_embedding(x: torch.Tensor, rs: float) -> torch.Tensor:
    """(4,) = (t, r, theta, phi) -> (6,) point of R^(1,5) (Fronsdal 1959), outside (r > r_s) and inside (r < r_s).

    z0, z1 = 2 r_s sqrt|f| (sinh, cosh) (t / 2 r_s) outside, (cosh, sinh) inside; z2 = fronsdal_z2(r); z3..5 = r n.
    """
    t, r, th, ph = x.unbind()
    f = 1 - rs / r
    A = 2 * rs * torch.sqrt(f.abs())
    tau = t / (2 * rs)
    out = f > 0
    z0 = A * torch.where(out, torch.sinh(tau), torch.cosh(tau))
    z1 = A * torch.where(out, torch.cosh(tau), torch.sinh(tau))
    return torch.stack([z0, z1, fronsdal_z2(r, rs), *_sphere(r, th, ph)])


def kruskal_r(T: torch.Tensor, X: torch.Tensor, rs: float, iters: int = 60) -> torch.Tensor:
    """r from X^2 - T^2 = (r/r_s - 1) e^(r/r_s): Newton steps from above (the function is convex), autograd-safe."""
    q = X**2 - T**2
    y = 2 + torch.log1p(q.abs())  # start to the right of the root
    for _ in range(iters):
        y = y - ((y - 1) * torch.exp(y) - q) / (y * torch.exp(y))
    return rs * y


def kruskal_metric(x: torch.Tensor, rs: float) -> torch.Tensor:
    """(4,) = (T, X, theta, phi) -> (4, 4) (4 r_s^3 / r) e^(-r/r_s) (-dT^2 + dX^2) + r^2 dOmega^2, regular at r_s."""
    T, X, th = x[0], x[1], x[2]
    r = kruskal_r(T, X, rs)
    c = 4 * rs**3 / r * torch.exp(-r / rs)
    return torch.diag(torch.stack([-c, c, r**2, (r * torch.sin(th)) ** 2]))


def fronsdal_kruskal(x: torch.Tensor, rs: float) -> torch.Tensor:
    """The same embedding in Kruskal coordinates: z0, z1 = 2 r_s sqrt(r_s/r) e^(-r/2r_s) (T, X), smooth at r = r_s."""
    T, X, th, ph = x.unbind()
    r = kruskal_r(T, X, rs)
    P = 2 * rs * torch.sqrt(rs / r) * torch.exp(-r / (2 * rs))
    return torch.stack([P * T, P * X, fronsdal_z2(r, rs), *_sphere(r, th, ph)])


# ---------- curvature by automatic differentiation ----------


def christoffel(metric, x: torch.Tensor) -> torch.Tensor:
    """(4, 4, 4) Gamma^a_bc = (1/2) g^ad (d_b g_dc + d_c g_db - d_d g_bc); metric maps (4,) -> (4, 4)."""
    g_inv = torch.linalg.inv(metric(x))
    dg = jacfwd(metric)(x)  # (4, 4, 4) dg[a, b, c] = d_c g_ab
    T = torch.einsum("dcb->dbc", dg) + dg - torch.einsum("bcd->dbc", dg)
    return 0.5 * torch.einsum("ad,dbc->abc", g_inv, T)  # repeated d => summed


def riemann(metric, x: torch.Tensor) -> torch.Tensor:
    """(4, 4, 4, 4) R^a_bcd = d_c Gamma^a_db - d_d Gamma^a_cb + Gamma^a_ce Gamma^e_db - Gamma^a_de Gamma^e_cb."""
    G = christoffel(metric, x)
    dG = jacfwd(lambda y: christoffel(metric, y))(x)  # (4, 4, 4, 4) dG[a, b, c, e] = d_e Gamma^a_bc
    return (
        torch.einsum("adbc->abcd", dG)
        - torch.einsum("acbd->abcd", dG)
        + torch.einsum("ace,edb->abcd", G, G)
        - torch.einsum("ade,ecb->abcd", G, G)
    )


def kretschmann(metric, x: torch.Tensor) -> torch.Tensor:
    """(B,) K = R_abcd R^abcd at points x (B, 4); all four indices summed."""

    def one(y):
        g = metric(y)
        gi = torch.linalg.inv(g)
        R = riemann(metric, y)  # R^a_bcd
        R_low = torch.einsum("ae,ebcd->abcd", g, R)
        R_up = torch.einsum("bf,cg,dh,afgh->abcd", gi, gi, gi, R)
        return torch.einsum("abcd,abcd->", R_low, R_up)

    return vmap(one)(x)


def static_tidal_eigenvalues(metric, x: torch.Tensor) -> torch.Tensor:
    """(B, 3) tidal acceleration per unit separation -R^i_titj along r, theta, phi for a static observer (diag metric).

    Geodesic deviation: d^2 xi^i / dtau^2 = -R^i_{t j t} xi^j in the observer's orthonormal frame. Positive = stretch.
    """

    def one(y):
        g = torch.diagonal(metric(y))
        R = riemann(metric, y)
        i = torch.arange(1, 4)
        return -R[i, 0, i, 0] / (-g[0])  # orthonormal: sqrt(g_ii) / sqrt(g_ii) for i = j, 1 / (-g_tt) for t, t

    return vmap(one)(x)


# ---------- the P4 claim: projections fold ----------


def loglog_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """Least-squares slope of log y against log x."""
    lx, ly = torch.log(x), torch.log(y)
    lx = lx - lx.mean()
    return float((lx * (ly - ly.mean())).sum() / (lx**2).sum())


def _log_histogram(v: torch.Tensor, lo: float, hi: float, bins: int, total: int):
    """Density of v > 0 on log-spaced bins in [lo, hi]: counts / (total * width)."""
    edges = torch.logspace(math.log10(lo), math.log10(hi), bins + 1, dtype=torch.float64)
    counts = torch.histogram(v.double().cpu(), bins=edges).hist
    centres = (edges[1:] * edges[:-1]).sqrt()
    return centres, counts / (total * edges.diff())


def catastrophe_density(n: int, samples: int, lo: float = 1e-5, hi: float = 1e-1, bins: int = 24):
    """Shadow of a uniform sheet s in [-1, 1] under x = s^n (fold n = 2, cusp n = 3, A_(n-1) in general).

    The density of x is sum over preimages 1/|dx/ds| ~ |x|^(1/n - 1): universal, set only by n.
    """
    s = torch.linspace(-1, 1, samples, dtype=torch.float64)
    return _log_histogram(s.pow(n).abs(), lo, hi, bins, samples)


def catastrophe_exponents(n: torch.Tensor, samples: int) -> torch.Tensor:
    """(N,) fitted density exponents for x = s^n, one per n."""
    return torch.tensor([loglog_slope(*catastrophe_density(int(k), samples)) for k in n], dtype=torch.float64)


def sphere_rim_samples(k: int, d: int, D: int, samples: int, generator) -> torch.Tensor:
    """u = 1 - |P x|^2 for x uniform on the unit sphere S^k in the first k + 1 axes of R^D, P onto the first d axes.

    u ~ Beta((k + 1 - d) / 2, d / 2), so near the rim (u -> 0) the shadow density goes like u^((k - d - 1) / 2).
    """
    if D < k + 1:
        raise ValueError("need D >= k + 1")
    z = torch.randn(samples, D, generator=generator, dtype=torch.float64)
    z[:, k + 1 :] = 0  # the sheet S^k lies in a (k + 1)-plane of R^D
    x = z / z.norm(dim=1, keepdim=True)
    return 1 - (x[:, :d] ** 2).sum(1)


def sphere_rim_exponents(k: torch.Tensor, d: int, D: int, samples: int, generator) -> torch.Tensor:
    """(K,) fitted rim exponents for spheres S^k projected onto R^d (k >= d)."""
    out = []
    for kk in k.tolist():  # a few sheet dimensions, each batched over all samples
        u = sphere_rim_samples(kk, d, D, samples, generator)
        out.append(loglog_slope(*_log_histogram(u, 1e-3, 10**-1.5, 10, samples)))
    return torch.tensor(out, dtype=torch.float64)


def sheet_projection(u: torch.Tensor, alpha: torch.Tensor, a: float) -> torch.Tensor:
    """(A, U) x = u cos(alpha) + a sin(u) sin(alpha): the wavy sheet z = a sin u, tilted by alpha, seen along z."""
    return u * torch.cos(alpha)[:, None] + a * torch.sin(u) * torch.sin(alpha)[:, None]


def sheet_fold_count(alpha: torch.Tensor, a: float, n: int = 4001) -> torch.Tensor:
    """(A,) number of fold points (dx/du = 0) per period: dx/du = cos(alpha) + a cos(u) sin(alpha)."""
    u = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    dx = torch.cos(alpha)[:, None] + a * torch.cos(u) * torch.sin(alpha)[:, None]
    return (torch.sign(dx).diff(dim=1) != 0).sum(1)


def fold_shell_field(r: torch.Tensor, R: float, rs: float) -> dict:
    """Weak-field caricature of a fold-capped interior: mass M = r_s/2 spread as the shadow of a 3-sphere of radius R.

    rho = rho0 (1 - r^2/R^2)^(-1/2) for r < R (a fold at r = R), M(<r) = 4 pi rho0 R^3 (chi/2 - sin(2 chi)/4) with
    chi = asin(r/R), rho0 = M / (pi^2 R^3). Newtonian tidal tensor: radial 2M(<r)/r^3 - 4 pi rho, transverse
    -M(<r)/r^3. Outside the shell it is exactly the Schwarzschild/Newton value r_s/r^3 (Birkhoff, shell theorem).
    """
    M = rs / 2
    rho0 = M / (math.pi**2 * R**3)
    inside = r < R
    chi = torch.asin((r / R).clamp(max=1.0))
    mass = 4 * math.pi * rho0 * R**3 * (chi / 2 - torch.sin(2 * chi) / 4)
    rho = torch.where(inside, rho0 / torch.sqrt((1 - (r / R) ** 2).clamp_min(1e-300)), torch.zeros_like(r))
    radial = 2 * mass / r**3 - 4 * math.pi * rho
    transverse = -mass / r**3
    proxy_K = 8 * (radial**2 + 2 * transverse**2)  # 8 sum E^2: equals Kretschmann in vacuum
    return {"mass": mass, "density": rho, "tidal_radial": radial, "tidal_transverse": transverse, "proxy_K": proxy_K}
