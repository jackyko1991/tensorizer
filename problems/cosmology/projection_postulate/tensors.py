"""Concrete tensors: random states of the one reality Psi, hidden blocks, Born weights, a triple slit, the data.

Psi is uniform on the unit sphere of C^D: a complex Gaussian vector z divided by its norm. The states for several
D are nested (the first D coordinates of one long Gaussian vector), so a single cumulative sum gives ||P Psi||^2 for
every D at once; each D on its own is still exactly uniform.
"""

import csv
import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"


def complex_gaussian(shape, generator, device) -> torch.Tensor:
    """Standard complex normal z = (x + i y) / sqrt(2), E|z|^2 = 1."""
    x = torch.randn(*shape, 2, generator=generator, dtype=torch.float64).to(device) / math.sqrt(2)
    return torch.complex(x[..., 0], x[..., 1])


def dimension_ladder(d: int, D_max: int) -> torch.Tensor:
    """(n_D,) roughly geometric list of total dimensions D from just above d up to D_max."""
    lo = max(d + 1, 4)
    n = max(2, int(round(1.5 * math.log2(D_max / lo))) + 1)
    return torch.unique(torch.logspace(math.log10(lo), math.log10(D_max), n).round().long())


def projected_states(Ds: torch.Tensor, d: int, trials: int, generator, device, chunk: int = 1024) -> dict:
    """Uniform Psi in C^D for every D in Ds, seen through the first d coordinates.

    Returns q (T, n_D) = ||P Psi||^2 (complex observer subspace C^d), psi (T, d) = P Psi / ||P Psi|| (the same for
    every D), shadow3 (T, n_D, 3) = the real 3D shadow (Re z1, Im z1, Re z2) / ||z|| and dir3 (T, 3) its direction.
    Trials are drawn in chunks only to bound memory.
    """
    D_max = int(Ds.max())
    idx = (Ds - 1).to(device)
    q, psi, shadow3 = [], [], []
    for start in range(0, trials, chunk):
        z = complex_gaussian((min(chunk, trials - start), D_max), generator, device)  # (t, D_max)
        total = torch.cumsum(z.abs() ** 2, dim=1)[:, idx]  # (t, n_D) ||z_{1..D}||^2 for every D at once
        q.append((z[:, :d].abs() ** 2).sum(1, keepdim=True) / total)  # (t, n_D)
        psi.append(z[:, :d] / z[:, :d].norm(dim=1, keepdim=True))
        s3 = torch.stack([z[:, 0].real, z[:, 0].imag, z[:, 1].real], -1)  # (t, 3)
        shadow3.append(s3[:, None, :] / total.sqrt()[..., None])
    psi_all = torch.cat(psi)
    s3_all = torch.cat(shadow3)
    dir3 = s3_all[:, 0] / s3_all[:, 0].norm(dim=-1, keepdim=True)
    return {"q": torch.cat(q), "psi": psi_all, "shadow3": s3_all, "dir3": dir3}


def hidden_energies(trials: int, K: int, m_max: int, field: str, generator, device) -> torch.Tensor:
    """(T, K, m_max) E_k = ||hidden block k||^2 when the block has m = 1..m_max dimensions (nested blocks).

    complex: each hidden coordinate |h|^2 ~ Exp(1), so E ~ Gamma(m, 1). real: h^2 ~ chi^2_1, E ~ 2 Gamma(m / 2).
    """
    if field == "complex":
        h2 = complex_gaussian((trials, K, m_max), generator, device).abs() ** 2
    else:
        h2 = torch.randn(trials, K, m_max, generator=generator, dtype=torch.float64).to(device) ** 2
    return torch.cumsum(h2, dim=-1)


def born_weights(K: int, device) -> torch.Tensor:
    """(K,) example outcome weights |c_k|^2 proportional to K, K-1, ..., 1 (e.g. 0.4, 0.3, 0.2, 0.1 for K = 4)."""
    w = torch.arange(K, 0, -1, dtype=torch.float64, device=device)
    return w / w.sum()


def slit_amplitudes(n_pix: int, tau: float, device, spacing: float = 1.0, width: float = 0.3, half_window: float = 2.5):
    """(3, X) far-field amplitude of each of three slits on X screen bins (odd X: bin X//2 is the centre).

    phi_s(x) = A sinc(width x) exp(2 pi i y_s x), slits at y_s = -spacing, 0, +spacing. Each slit alone sends a
    fraction tau of the source onto the screen; the rest is blocked by the mask (its own outcome).
    """
    x = torch.linspace(-half_window, half_window, n_pix, dtype=torch.float64, device=device)
    env = torch.sinc(width * x)  # sin(pi u) / (pi u)
    env = env * math.sqrt(tau) / env.norm()
    y = spacing * torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, device=device)
    return env * torch.exp(2j * math.pi * y[:, None] * x)


def born_rule_tests() -> list[dict]:
    """Cited triple-path (Sorkin) interference bounds: kappa value, 1 sigma, regime, source."""
    with open(DATA / "born_rule_tests.csv", encoding="utf-8") as f:
        return [row for row in csv.DictReader(r for r in f if not r.startswith("#"))]
