"""Metropolis Monte Carlo for the whole batch of couplings at once (checkerboard over links), measurements, and the
glueball correlator with jackknife errors."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import half_trace, plaquettes, qmul, staple, wilson_loops


def proposal(shape: tuple, eps: torch.Tensor, group: str, gen: torch.Generator) -> torch.Tensor:
    """Random group element near 1, R = normalise(1, ε·n) with Gaussian n; R and R⁻¹ are equally likely, so the
    Metropolis step obeys detailed balance. eps (B,) one step size per coupling.   → (*shape, 4)"""
    n = torch.randn(*shape, 4, generator=gen, dtype=torch.float64).to(eps.device)
    if group == "U(1)":
        n[..., 1:3] = 0.0  # stay on the U(1) circle (cos θ, 0, 0, sin θ)
    n[..., 1:] *= eps.view(-1, *([1] * len(shape)))  # shape = (B, L, …): step ε_b for coupling b
    n[..., 0] = 1.0
    return n / n.norm(dim=-1, keepdim=True)


def sweep(links: torch.Tensor, beta: torch.Tensor, group: str, gen: torch.Generator, hits: int = 2):
    """One sweep: for every direction μ and parity, all those links get `hits` Metropolis trials at once.
    Returns the new links and the acceptance rate per coupling (B,)."""
    links = links.clone()
    dim, size = links.shape[1], links.shape[2]
    par = tensors.parity(dim, size, links.device)
    eps = (1.5 / torch.sqrt(beta * (dim - 1))).clamp(max=10.0)  # step ∝ 1/√(β·staple size): acceptance ≈ ½
    bshape = (-1,) + (1,) * dim
    accepted = torch.zeros_like(beta)
    for mu in range(dim):
        for p in (0, 1):
            a = staple(links, mu)  # (B, L…, 4) fixed while the μ-links of parity p change
            old = links[:, mu]
            s_old = half_trace(old, a)
            for _ in range(hits):
                new = qmul(proposal(old.shape[:-1], eps, group, gen), old)
                new = new / new.norm(dim=-1, keepdim=True)
                s_new = half_trace(new, a)
                d_action = -beta.view(bshape) * (s_new - s_old)  # ΔS = −β ½Tr[(U′ − U) A]
                rnd = torch.rand(s_new.shape, generator=gen, dtype=torch.float64).to(links.device)
                ok = (rnd < torch.exp(-d_action)) & (par == p)
                old = torch.where(ok[..., None], new, old)
                s_old = torch.where(ok, s_new, s_old)
                accepted += ok.flatten(1).sum(dim=1) / ok[0].numel()
            links[:, mu] = old
    return links, accepted / (dim * hits)


def simulate(
    group: str,
    dim: int,
    size: int,
    betas: torch.Tensor,
    n_sweeps: int,
    n_therm: int,
    seed: int,
    measure_every: int,
    r_max: int = 3,
    n_snapshots: int = 0,
) -> dict:
    """Hot start, n_therm sweeps of thermalisation, then measure every sweep:
    plaquette (S, B); glueball operator O(t) = Σ_x spatial ½ Tr P on each time slice (S, B, L), axis 0 is time;
    Wilson loops every measure_every sweeps (M, B, R, T); n_snapshots plaquette slices (K, B, L, L)."""
    gen = torch.Generator().manual_seed(seed)
    links = tensors.hot_start(len(betas), dim, size, group, gen, betas.device)
    plaq, ops, loops, snaps, snap_sweep, acc = [], [], [], [], [], []
    snap_at = {round(k * (n_sweeps - 1) / max(1, n_snapshots - 1)) for k in range(n_snapshots)}
    spatial = [i for i, (mu, nu) in enumerate((m, n) for m in range(dim) for n in range(m + 1, dim)) if mu > 0]
    for s in steps(n_sweeps, "sweeps"):
        links, a = sweep(links, betas, group, gen)
        acc.append(a)
        p = plaquettes(links)  # (B, P, L…)
        if s in snap_at:
            snaps.append(p[:, 0].reshape(len(betas), size, size, -1)[..., 0])  # plane (0, 1) through x₂ = 0
            snap_sweep.append(s)
        if s < n_therm:
            continue
        plaq.append(p.flatten(1).mean(dim=1))
        if dim > 2:
            ops.append(p[:, spatial].sum(dim=1).flatten(2).sum(dim=2))  # (B, L): sum over spatial sites
        if (s - n_therm) % measure_every == 0:
            loops.append(wilson_loops(links, r_max))
    return {
        "plaquette": torch.stack(plaq),
        "operator": torch.stack(ops) if ops else None,
        "wilson": torch.stack(loops),
        "snapshots": torch.stack(snaps) if snaps else None,
        "snapshot_sweeps": snap_sweep,
        "acceptance": torch.stack(acc).mean(dim=0),
    }


def jackknife(samples: torch.Tensor, estimator, n_bins: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and error of estimator(bin means) by leave-one-bin-out jackknife (bins absorb autocorrelation).
    samples (S, …); estimator maps a (…) tensor of means to the quantity."""
    n = samples.shape[0] // n_bins * n_bins
    bins = samples[:n].reshape(n_bins, n // n_bins, *samples.shape[1:]).mean(dim=1)  # (n_bins, …)
    total = bins.sum(dim=0)
    leave_out = torch.stack([estimator((total - bins[i]) / (n_bins - 1)) for i in range(n_bins)])
    centre = estimator(bins.mean(dim=0))
    err = torch.sqrt((n_bins - 1) * ((leave_out - leave_out.mean(dim=0)) ** 2).mean(dim=0))
    return centre, err


def correlator_samples(op: torch.Tensor, t_max: int) -> torch.Tensor:
    """Per-configuration ingredients of C(Δ) = ⟨O(t)O(t+Δ)⟩ − ⟨O⟩²: [Ō, mean_t O(t)O(t+Δ) for Δ = 0 … t_max].
    op (S, B, L) → (S, B, t_max + 2)"""
    prods = [(op * torch.roll(op, -d, dims=-1)).mean(dim=-1) for d in range(t_max + 1)]
    return torch.stack([op.mean(dim=-1)] + prods, dim=-1)


def effective_mass(means: torch.Tensor) -> torch.Tensor:
    """m_eff(Δ) = ln C(Δ)/C(Δ+1) from the averaged ingredients (…, t_max + 2) → (…, t_max)."""
    c = means[..., 1:] - means[..., :1] ** 2  # connected correlator C(Δ)
    return torch.log(c[..., :-1].clamp_min(1e-30) / c[..., 1:].clamp_min(1e-30))


def running(out: dict, i0: int, n_therm: int, frames: list[int], t_max: int, measure_every: int = 5) -> dict:
    """The estimates as they would have looked after each frame's sweep: ⟨½TrP⟩ (K, B), W(R, T) at β_{i0}
    (K, R, T) and the glueball m_eff(Δ) with jackknife errors (K, Δ); NaN before enough data exist."""
    plaq, wil, mass, err = [], [], [], []
    nan = float("nan")
    for s in frames:
        n = s - n_therm + 1  # measurements so far (one per sweep after thermalisation)
        n_w = (s - n_therm) // measure_every + 1
        plaq.append(out["plaquette"][:n].mean(dim=0) if n > 0 else torch.full_like(out["plaquette"][0], nan))
        wil.append(out["wilson"][:n_w, i0].mean(dim=0) if n > 0 else torch.full_like(out["wilson"][0, i0], nan))
        if out["operator"] is not None and n >= 40:
            m, e = jackknife(correlator_samples(out["operator"][:n, i0], t_max), effective_mass)
        else:
            m = e = torch.full((t_max,), nan, dtype=torch.float64)
        mass.append(m)
        err.append(e)
    return {
        "plaquette": torch.stack(plaq),
        "wilson": torch.stack(wil),
        "mass": torch.stack(mass),
        "mass_err": torch.stack(err),
    }
