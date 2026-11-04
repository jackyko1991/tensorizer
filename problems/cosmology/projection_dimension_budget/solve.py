"""Solvers: the (n, R) exclusion map bound by bound, the largest allowed radius, force laws, the quantum count."""

import math

import torch

from . import equation, tensors


def exclusion_map(n: torch.Tensor, R: torch.Tensor, rows: list[dict]) -> dict:
    """Masks per bound family (N, R), the first family that excludes each point (-1 = allowed), and M_D on the grid."""
    masks = equation.exclusion_masks(n[:, None], R[None, :], rows)
    first = torch.full((len(n), len(R)), -1, dtype=torch.long, device=R.device)
    for i, name in reversed(list(enumerate(tensors.STAGES))):  # earlier stages win where several exclude
        first = torch.where(masks[name], torch.full_like(first, i), first)
    md = equation.add_md_tev(n[1:, None].clamp_min(1), R[None, :])  # (N-1, R) M_D in TeV for n >= 1
    return {"masks": masks, "first": first, "md": md, "excluded": first >= 0}


def radius_budget(n: torch.Tensor, rows: list[dict]) -> dict:
    """R_max(n) after each stage of bounds (cumulative) and the M_D it corresponds to."""
    per_stage = []
    for k in range(len(tensors.STAGES)):
        used = [r for r in rows if r["bound"] in tensors.STAGES[: k + 1]]
        per_stage.append(equation.r_max(n, used))
    rmax = per_stage[-1]
    md = torch.where(n >= 1, equation.add_md_tev(n.clamp_min(1), rmax.clamp_max(1.0)), torch.nan)
    return {"per_stage": torch.stack(per_stage), "rmax": rmax, "md_at_rmax": md}


def force_curves(device) -> dict:
    """r^2 g(r) / (G_4 M) on R^3 x T^n, n = 1, 2, 3: (R/r)^n inside the compact space, 1 outside (r <= 10 R)."""
    r = torch.logspace(-2, 1, 50, dtype=torch.float64, device=device)
    out = {}
    for n, K in ((1, 2000), (2, 150), (3, 30)):
        out[n] = r**2 * equation.image_force(r, n, K) / equation.g4_over_gd(n)
    return {"r": r, "curves": out}


def gauss_check(n_max: int, generator) -> torch.Tensor:
    """Flux of the pull through spheres of radius 0.5, 1, 2 divided by |S^(n+2)|, for n = 0..n_max (should be 1)."""
    radii = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    return torch.stack([equation.gauss_flux(n, radii, 2000, generator) / equation.sphere_area(n + 2)
                        for n in range(n_max + 1)])  # fmt: skip


def quantum_budget(d: int, outcomes: int, qubits: int, device) -> dict:
    """Step 1's count: d + K for the Born-rule race, 2^N for N entangled qubits, and the spatial reading."""
    N = torch.arange(1, 81, dtype=torch.float64, device=device)
    Nq = torch.tensor(float(qubits), dtype=torch.float64)
    return {"N": N, "log10_hilbert": equation.hilbert_log2_dim(N) * math.log10(2),
            "log10_spatial": torch.log10(equation.spatial_reading_extra_dims(N)),
            "race": equation.race_dims(d, outcomes),
            "qubits": qubits, "hilbert_q": int(2 ** qubits),
            "spatial_q": int(equation.spatial_reading_extra_dims(Nq).item())}  # fmt: skip
