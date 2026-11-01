"""Chern number on a k-grid, zigzag-ribbon bands, and light travelling along the edge of a finite flake."""

import math

import torch

from . import tensors
from .equation import (
    berry_curvature_formula,
    bloch_hamiltonian,
    d_vector,
    evolve,
    hoppings,
    link_variables,
    plaquette_flux,
)


def chern_number(t1: float, t2: float, phi: float, M: float, n: int, device: torch.device) -> dict:
    """Fukui–Hatsugai–Suzuki on an n×n grid: lower-band eigenvectors → links → plaquette fluxes → C."""
    k, cell = tensors.bz_grid(n, device)  # (n, n, 2)
    delta, b = tensors.CELL_DELTA.to(device), tensors.B_NNN.to(device)
    H = bloch_hamiltonian(d_vector(k, delta, b, t1, t2, phi, M))  # (n, n, 2, 2): all k at once
    E, V = torch.linalg.eigh(H)  # (n, n, 2), (n, n, 2, 2)
    u = V[..., 0]  # (n, n, 2) lower band (any phase: the links are gauge invariant around a plaquette)
    F = plaquette_flux(*link_variables(u))  # (n, n)
    C = F.sum().item() / (2 * math.pi)
    gap = (E[..., 1] - E[..., 0]).min().item()
    return {"C": C, "omega": F / cell, "cell": cell, "gap": gap, "k": k}


def berry_curvature_autograd(k: torch.Tensor, t1: float, t2: float, phi: float, M: float) -> torch.Tensor:
    """Ω(k) = -½ ε_abc d̂_a ∂_x d̂_b ∂_y d̂_c, with the k-derivatives of d̂ taken by autograd.   k: (..., 2) → (...)"""
    k = k.detach().clone().requires_grad_(True)
    d = d_vector(k, tensors.CELL_DELTA.to(k.device), tensors.B_NNN.to(k.device), t1, t2, phi, M)[..., 1:]
    dhat = d / d.norm(dim=-1, keepdim=True)  # (..., 3)
    # every k is independent, so the gradient of Σ_k d̂_a(k) is ∂d̂_a/∂k at each point: (..., 2)
    grads = [torch.autograd.grad(dhat[..., a].sum(), k, retain_graph=True)[0] for a in range(3)]
    jac = torch.stack(grads, dim=-2)  # (..., 3, 2)  ∂_i d̂_a
    return berry_curvature_formula(dhat.detach(), jac[..., 0], jac[..., 1])


def ribbon_bands(width: int, t1: float, t2: float, phi: float, M: float, n_k: int, device: torch.device) -> dict:
    """Bands of a zigzag ribbon vs k_x: H(k) = Σ_m T(m) e^{i k_x m √3}, m = -1, 0, 1 neighbouring periods."""
    pos, xi = tensors.ribbon_cell(width, device)  # (2W, 2), (2W,)
    a1 = tensors.A1.to(device)
    T = torch.stack([hoppings(pos, pos + m * a1, xi, t1, t2, phi) for m in (-1, 0, 1)])  # (3, 2W, 2W)
    kx = torch.linspace(0, 2 * math.pi, n_k, dtype=torch.float64, device=device) / math.sqrt(3)  # (K,) one zone
    phase = torch.exp(1j * kx[:, None] * torch.tensor([-1.0, 0.0, 1.0], device=device) * math.sqrt(3))  # (K, 3)
    H = torch.einsum("km,mij->kij", phase, T) + torch.diag(M * xi).to(torch.complex128)  # (K, 2W, 2W)
    E, V = torch.linalg.eigh(H)  # (K, 2W), (K, 2W, 2W)
    weight = V.abs() ** 2  # (K, site, band)
    rows = len(pos) // 4  # bottom and top quarter of the ribbon
    bottom = weight[:, :rows].sum(1)  # (K, band)
    top = weight[:, -rows:].sum(1)
    return {"kx": kx, "E": E, "bottom": bottom, "top": top}


def bottom_edge_mode(bands: dict, window: float) -> tuple[float, float]:
    """Group velocity v = dE/dk and momentum k_c of the bottom-edge state inside the gap (|E| < window).

    A straight-line fit E ≈ v (k - k_c) through the states that live on the bottom edge; the sign of v is the
    direction light travels there. Returns (0, 0) when there is no such state (a trivial gap).
    """
    kx, E, bottom = bands["kx"], bands["E"], bands["bottom"]
    on_edge = (bottom > 0.6) & (E.abs() < window)  # (K, band)
    if on_edge.sum() < 3:
        return 0.0, 0.0
    k = kx[:, None].expand_as(E)[on_edge]
    e = E[on_edge]
    v = (((k - k.mean()) * (e - e.mean())).sum() / ((k - k.mean()) ** 2).sum()).item()
    return v, (k.mean() - e.mean() / v).item()


def flake_hamiltonian(pos, xi, t1: float, t2: float, phi: float, M: float, detuning: torch.Tensor) -> torch.Tensor:
    """H_ij of the finite flake: couplings plus on-site M ξ_i and a random detuning. (N, N) complex Hermitian."""
    return hoppings(pos, pos, xi, t1, t2, phi) + torch.diag(M * xi + detuning).to(torch.complex128)


def edge_packet(
    H: torch.Tensor, pos: torch.Tensor, center: torch.Tensor, k_c: float, window: float
) -> tuple[torch.Tensor, float]:
    """A spot of light near the edge with the edge mode's momentum e^{i k_c x}, filtered to frequencies
    |E| < window (inside the bulk gap). Returns ψ_0 (N,) and the fraction of the spot that was kept."""
    spot = tensors.gaussian(pos, center, 1.5) * torch.exp(1j * k_c * pos[:, 0])
    E, V = torch.linalg.eigh(H)
    c = V.conj().T @ spot  # mode amplitudes
    c = torch.where(E.abs() < window, c, torch.zeros_like(c))
    kept = (c.abs() ** 2).sum().item()
    if kept < 1e-8:  # no states in the window (a trivial gap): fall back to the bare spot
        return spot, 0.0
    psi = V @ c
    return psi / psi.norm(), kept


def detector(pos, notch, R: float, direction: float, width: float, height: float, strip: float = 3.0) -> torch.Tensor:
    """Resonators downstream of the notch and within `strip` of the flake's outer boundary: boolean (N,)."""
    x, y = pos[:, 0], pos[:, 1]
    to_boundary = torch.stack([x, width - x, y - y.min(), height - y]).min(0).values
    return (direction * (x - notch[0]) > R + 1) & (to_boundary < strip)


def propagate(Hs: torch.Tensor, psi0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """|ψ_i(t)|² for each lattice in the batch, ψ(t) = e^{-iHt} ψ_0 via one eigh each.   Hs: (S, N, N) → (S, T, N)"""
    E, V = torch.linalg.eigh(Hs)
    return evolve(E, V, psi0, times).abs() ** 2


def edge_transport(
    t1: float, t2: float, phi: float, masses: tuple, disorder: float, R: float, v: float, k_c: float, window: float,
    width: float, height: float, n_times: int, device: torch.device,
) -> dict:  # fmt: skip
    """Launch edge light upstream of a notch in a disordered flake and follow it in one lattice per mass.

    Returns positions, intensity |ψ|² (S, T, N), times (T,), the fraction past the notch along the edge (S, T),
    the notch centre and the fraction of the input spot that lay inside the gap.
    """
    pos, xi = tensors.flake(width, height, device)
    notch = torch.tensor([width / 2, 0.0], dtype=torch.float64, device=device)
    pos, xi = tensors.carve_defect(pos, xi, notch, R)
    detuning = tensors.disorder(len(pos), disorder, seed=1, device=device)
    Hs = torch.stack([flake_hamiltonian(pos, xi, t1, t2, phi, m, detuning) for m in masses])  # (S, N, N)

    s = 1.0 if v >= 0 else -1.0  # direction of travel along the bottom edge
    source = torch.tensor([width / 2 - s * width / 4, 1.0], dtype=torch.float64, device=device)
    psi0, kept = edge_packet(Hs[0], pos, source, k_c, window)
    t_max = (0.5 * width + math.pi * R + 0.3 * height) / max(abs(v), 0.2)  # long enough to pass the notch
    times = torch.linspace(0, t_max, n_times, dtype=torch.float64, device=device)
    intensity = propagate(Hs, psi0, times)  # (S, T, N)
    transmitted = intensity[:, :, detector(pos, notch, R, s, width, height)].sum(-1)  # (S, T)
    return {
        "pos": pos,
        "intensity": intensity,
        "times": times,
        "transmitted": transmitted,
        "notch": notch,
        "kept": kept,
    }
