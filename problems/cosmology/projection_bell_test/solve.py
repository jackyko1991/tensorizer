"""Solvers: correlation curves and CHSH values of the joint and local projection races, signalling, the Bell data."""

import math

import torch

from . import equation, tensors

TSIRELSON = 2 * math.sqrt(2)
# step 1 (projection_postulate): three-path interference allows |m - 1| <= 6e-3 (heralded single photons) and
# <= 1.1e-4 (classical light, if every detection is one race)
SORKIN_BANDS = {"single photons": 6e-3, "classical light": 1.1e-4}
LOCAL_M = [1.0, 2.0, math.inf]


def correlations(m_max: int, D: int, trials: int, generator, device) -> dict:
    """E(theta) for quantum mechanics, the joint race (closed form for m = 1..m_max, Monte Carlo from Psi in C^D
    for m = 1) and the local races (m = 1, 2, oo), plus S(theta) along the CHSH family
    a = 0, b = t, a' = 2t, b' = 3t."""
    psi = equation.singlet(device)
    theta = torch.linspace(0, math.pi, 181, dtype=torch.float64, device=device)
    a, b = tensors.planar_pairs(theta, device)
    w = equation.joint_born(psi, a, b).flatten(-2)  # (Th, 4)
    m = torch.arange(1, m_max + 1, dtype=torch.float64, device=device)
    E_joint = equation.correlation(
        equation.race_probabilities(w.expand(m_max, -1, -1), m[:, None]).view(m_max, -1, 2, 2)
    )
    th_mc = theta[::10]
    E_mc = equation.joint_race_correlation(w[::10], tensors.hidden_block_energies(trials, 4, 1, D, generator, device))
    n = tensors.shared_directions(20_000, D, generator, device)
    E_local = torch.stack([equation.local_correlation(a, b, n, mm) for mm in LOCAL_M])  # (L, Th)

    def S_of(E_curve):  # S(t) = |3 E(t) - E(3t)| from a curve sampled on [0, pi], E even and 2 pi periodic
        t3 = torch.remainder(3 * theta, 2 * math.pi)
        t3 = torch.where(t3 > math.pi, 2 * math.pi - t3, t3)
        idx = (t3 / math.pi * (len(theta) - 1)).round().long()
        return (3 * E_curve - E_curve[..., idx]).abs()

    return {"theta": theta, "E_qm": -torch.cos(theta), "E_joint": E_joint, "th_mc": th_mc, "E_mc": E_mc,
            "E_local": E_local, "S_qm": S_of(-torch.cos(theta)), "S_joint": S_of(E_joint), "S_local": S_of(E_local),
            "mc_err": (E_mc + torch.cos(th_mc)).abs().max().item()}  # fmt: skip


def random_chsh(B: int, D: int, generator, device) -> dict:
    """S over B random measurement settings for the joint Born race (= quantum mechanics) and the local races."""
    settings = tensors.random_settings(B, generator, device)
    psi = equation.singlet(device)
    S_joint = equation.chsh(lambda a, b: equation.correlation(equation.joint_born(psi, a, b)), *settings)
    n = tensors.shared_directions(4000, D, generator, device)
    S_local = torch.stack([equation.chsh(lambda a, b, mm=mm: equation.local_correlation(a, b, n, mm), *settings)
                           for mm in LOCAL_M])  # fmt: skip
    opt = tensors.optimal_settings(device)
    S_local_opt = [
        equation.chsh(lambda a, b, mm=mm: equation.local_correlation(a, b, n, mm), *opt).item() for mm in LOCAL_M
    ]
    return {"S_joint": S_joint, "S_local": S_local, "S_local_opt": S_local_opt}


def beyond_born(m_max: int, generator, device) -> dict:
    """Joint race with m != 1: CHSH at the optimal settings, signalling for a partly entangled state, and the
    deviations allowed by the triple-slit bound of step 1."""
    m = torch.cat([torch.linspace(0.5, 1, 11), torch.linspace(1, m_max, 40)[1:]]).double().to(device)
    opt = tensors.optimal_settings(device)
    S = equation.joint_race_chsh(equation.singlet(device), m, *opt)
    sig = equation.max_signalling(equation.partly_entangled(0.3, device), m, generator, device)
    h = torch.tensor([1 - 1e-3, 1 + 1e-3], dtype=torch.float64, device=device)
    dS = equation.joint_race_chsh(equation.singlet(device), h, *opt).diff().item() / 2e-3
    gs = torch.Generator().manual_seed(5)
    sig1 = equation.max_signalling(equation.partly_entangled(0.3, device), h[1:], gs, device).item() / 1e-3
    # shape test: does E(theta) of a race with m = 1 + delta differ from a damped cosine V cos(theta)?
    theta = torch.linspace(0, math.pi, 91, dtype=torch.float64, device=device)
    w = equation.joint_born(equation.singlet(device), *tensors.planar_pairs(theta, device)).flatten(-2)
    delta = torch.tensor(list(SORKIN_BANDS.values()), dtype=torch.float64, device=device)
    E = equation.correlation(equation.race_probabilities(w.expand(len(delta), -1, -1), 1 + delta[:, None]).view(
        len(delta), -1, 2, 2))  # fmt: skip
    c = -torch.cos(theta)
    V = (E * c).sum(-1) / (c * c).sum()  # least-squares visibility
    shape = (E - V[:, None] * c).abs().amax(-1)  # what no visibility can absorb
    bands = {k: (v, abs(dS) * v, sig1 * v, shape[i].item()) for i, (k, v) in enumerate(SORKIN_BANDS.items())}
    return {"m": m, "S": S, "signal": sig, "dS_dm": dS, "dsignal_dm": sig1, "bands": bands}


def data_table() -> list[dict]:
    """The cited Bell tests with their distance from the local bound 2 and the quantum bound 2 sqrt 2."""
    out = []
    for r in tensors.bell_tests():
        row = dict(r)
        if r["S"]:
            S, s = float(r["S"]), float(r["sigma"])
            row.update(S=S, sigma=s, above_local=(S - 2) / s, below_tsirelson=(TSIRELSON - S) / s,
                       visibility=S / TSIRELSON)  # fmt: skip
        out.append(row)
    return out
