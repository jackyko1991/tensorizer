r"""Physics → tensor translation: incompressible Navier–Stokes on a voxel finite-volume mesh, plus the vessel laws.

The laws (index form, repeated j summed):

    ∂_t u_i + u_j ∂_j u_i = -∂_i p/ρ + ν ∂_j ∂_j u_i + g_i,        ∂_j u_j = 0

On the mesh every quantity lives where OpenFOAM keeps it: pressure p_c in cells c, the face-normal velocity
u_f on faces f (OpenFOAM's flux φ_f = u_f A_f). Each internal face has an owner cell o(f) and a neighbour cell
n(f) (OpenFOAM's lduAddressing). Then

    divergence      (∇·u)_c = (Σ_{f: o(f)=c} u_f − Σ_{f: n(f)=c} u_f) / h → index_add over faces (fvc::div(phi))
    Laplacian       (∇²p)_c = Σ_{f ∋ c} ±(p_n(f) − p_o(f)) / h² → index_add over faces (fvm::laplacian)
    gradient        (∇p)_f = (p_n(f) − p_o(f)) / h → gather (fvc::snGrad)

Chorin's projection makes one time step: u* = advect(u) + Δt(ν∇²u + g), then solve ∇²p = ∇·u*/Δt and set
u = u* − Δt ∇p, which leaves ∇·u = 0 (to the tolerance of the conjugate-gradient solve).

Vessel laws used as checks: Hagen–Poiseuille, Womersley's oscillatory pipe flow, wall shear stress τ = μ ∂u_t/∂n,
its time average (TAWSS) and the oscillatory shear index (OSI). For the centreline: the eikonal equation
|∇T| = 1/R^p solved by min-plus relaxation T_v = min_u (T_u + ℓ_uv / R_uv^p) on the voxel graph.
"""

import math

import torch


# ---------------------------------------------------------------- finite-volume operators on the face/cell lists
def divergence(u: torch.Tensor, owner: torch.Tensor, neigh: torch.Tensor, n_cells: int, h: float) -> torch.Tensor:
    r"""(∇·u)_c = (Σ_{o(f)=c} u_f − Σ_{n(f)=c} u_f)/h.   u: (F,) face velocities → (N,) per cell (1/s)."""
    out = torch.zeros(n_cells, dtype=u.dtype, device=u.device)
    return out.index_add(0, owner, u).index_add(0, neigh, -u) / h


def laplacian(p: torch.Tensor, owner: torch.Tensor, neigh: torch.Tensor, h: float) -> torch.Tensor:
    r"""(∇²p)_c = Σ_f ±(p_{n(f)} − p_{o(f)})/h² over the given faces.   p: (N,) → (N,)."""
    flux = (p[neigh] - p[owner]) / h**2  # (F,) one value per face, added to the owner and taken from the neighbour
    return torch.zeros_like(p).index_add(0, owner, flux).index_add(0, neigh, -flux)


def face_gradient(p: torch.Tensor, owner: torch.Tensor, neigh: torch.Tensor, h: float) -> torch.Tensor:
    r"""(∇p)_f = (p_{n(f)} − p_{o(f)})/h, the face-normal gradient.   (N,) → (F,)."""
    return (p[neigh] - p[owner]) / h


def face_laplacian(u: torch.Tensor, fnb: torch.Tensor, ghost: torch.Tensor, h: float) -> torch.Tensor:
    r"""ν∇²u for face velocities: (Σ_s u_{s} − 6 u_f)/h² over the 6 same-direction neighbours s.

    fnb: (F, 6) neighbour face index or −1 (a wall). A missing neighbour takes the ghost value g_fs u_f, chosen
    so that the velocity is 0 exactly on the wall (the immersed-boundary ghost-cell method).
    """
    nbv = torch.where(fnb >= 0, u[fnb.clamp(min=0)], ghost * u[:, None])  # (F, 6)
    return (nbv.sum(1) - 6 * u) / h**2


# ---------------------------------------------------------------- vessel laws (analytic checks)
def bessel_j(nu: int, z: torch.Tensor, terms: int = 60) -> torch.Tensor:
    r"""J_ν(z) = Σ_k (−1)^k (z/2)^{2k+ν} / (k! (k+ν)!) for complex z (power series, fine for |z| ≲ 20)."""
    # term ratios t_k / t_{k-1} = −(z/2)² / (k(k+ν)), multiplied up (no complex powers: 0⁰ is NaN on some builds)
    w = z / 2
    first = torch.ones_like(w)
    for j in range(1, nu + 1):
        first = first * w / j  # (z/2)^ν / ν!
    k = torch.arange(1, terms, dtype=torch.float64, device=z.device)
    ratio = -(w[..., None] ** 2) / (k * (k + nu))  # (..., K−1)
    return first * (1 + torch.cumprod(ratio, dim=-1).sum(-1))


def poiseuille_wss(Q: float, R: float, mu: float) -> float:
    r"""Hagen–Poiseuille wall shear stress τ_w = 4μQ/(πR³)."""
    return 4 * mu * Q / (math.pi * R**3)


def womersley_pressure(r: torch.Tensor, t: torch.Tensor, G: list, omega: float, R: float, nu: float) -> torch.Tensor:
    r"""Axial velocity driven by −∂_z p/ρ = Re Σ_n G_n e^{i n ω t} (G_0 steady):

    u = G_0 (R² − r²)/(4ν) + Re Σ_{n≥1} (G_n/(i n ω)) (1 − J_0(Λ_n r/R)/J_0(Λ_n)) e^{i n ω t},  Λ_n = i^{3/2} R √(nω/ν)
    r: (M,), t: (T,) → (T, M).
    """
    u = (G[0].real * (R**2 - r**2) / (4 * nu))[None, :].repeat(len(t), 1)
    for n, Gn in enumerate(G[1:], start=1):
        lam = complex(math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4)) * R * math.sqrt(n * omega / nu)
        lam_t = torch.tensor(lam, dtype=torch.complex128)
        shape = 1 - bessel_j(0, lam_t * r.to(torch.complex128) / R) / bessel_j(0, lam_t)  # (M,)
        phase = torch.exp(1j * n * omega * t.to(torch.complex128))  # (T,)
        u = u + (Gn / (1j * n * omega) * phase[:, None] * shape[None, :]).real
    return u


def womersley_shapes(r: torch.Tensor, n_harm: int, omega: float, R: float, nu: float) -> torch.Tensor:
    r"""Womersley's profile per unit flow for each harmonic n = 0 … H (each carries exactly 1 m³/s):

    S_0 = (2/πR²)(1 − r²/R²),   S_n = (1/πR²) (1 − J_0(Λ_n r/R)/J_0(Λ_n)) / (1 − 2J_1(Λ_n)/(Λ_n J_0(Λ_n)))
    r: (M,) → (H+1, M) complex.
    """
    area = math.pi * R**2
    rr = (r / R).clamp(max=1.0).to(torch.complex128)
    out = [2 / area * (1 - rr**2)]
    for n in range(1, n_harm + 1):
        lam = torch.tensor(
            complex(math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4)) * R * math.sqrt(n * omega / nu)
        )
        j0 = bessel_j(0, lam)
        out.append((1 - bessel_j(0, lam * rr) / j0) / (1 - 2 * bessel_j(1, lam) / (lam * j0)) / area)
    return torch.stack(out)


def womersley_flow(r: torch.Tensor, t: torch.Tensor, Qn: torch.Tensor, omega: float, R: float, nu: float):
    r"""Axial velocity for a given flow rate Q(t) = Re Σ_n Q_n e^{inωt}: u(r, t) = Re Σ_n Q_n S_n(r) e^{inωt}.
    r: (M,), t: (T,), Qn: (H+1,) complex → (T, M)."""
    S = womersley_shapes(r, len(Qn) - 1, omega, R, nu)  # (H+1, M)
    n = torch.arange(len(Qn), dtype=torch.float64)
    phase = torch.exp(1j * omega * t.to(torch.float64)[:, None] * n[None, :])  # (T, H+1)
    return torch.einsum("th,h,hm->tm", phase, Qn.to(torch.complex128), S).real


# ---------------------------------------------------------------- wall shear stress and its indices
def wall_shear(u1: torch.Tensor, u2: torch.Tensor, m: torch.Tensor, delta: float, mu: float) -> torch.Tensor:
    r"""τ_i = μ (δ_ij − m_i m_j) ∂_n u_j at the wall, with ∂_n u from a parabola through u = 0 at the wall and the
    samples u1 = u(δ), u2 = u(2δ) along the inward normal m: ∂_n u = (4 u1 − u2)/(2δ).   (V, 3) each → (V, 3)."""
    a = (4 * u1 - u2) / (2 * delta)
    return mu * (a - torch.einsum("vj,vj->v", a, m)[:, None] * m)


def tawss(tau: torch.Tensor) -> torch.Tensor:
    r"""TAWSS = (1/T)∫|τ| dt.   tau: (T, V, 3) → (V,)."""
    return tau.norm(dim=-1).mean(0)


def osi(tau: torch.Tensor) -> torch.Tensor:
    r"""OSI = ½ (1 − |∫τ dt| / ∫|τ| dt): 0 for one-way shear, ½ for purely oscillating shear.   (T, V, 3) → (V,)."""
    return 0.5 * (1 - tau.mean(0).norm(dim=-1) / tau.norm(dim=-1).mean(0).clamp(min=1e-12))


# ---------------------------------------------------------------- centreline: eikonal by min-plus relaxation
def edge_cost(length: torch.Tensor, R_mid: torch.Tensor, power: float) -> torch.Tensor:
    r"""Travel time along a graph edge, ℓ / R^p: the front moves fastest where the inscribed sphere is largest."""
    return length / R_mid.clamp(min=1e-9) ** power


def relax(T: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""One min-plus (Bellman–Ford) sweep of the discrete eikonal equation: T_v ← min(T_v, min_u T_u + w_uv)."""
    return T.scatter_reduce(0, dst, T[src] + w, reduce="amin", include_self=True)
