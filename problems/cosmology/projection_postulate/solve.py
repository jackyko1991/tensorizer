"""Solvers: concentration of measure over a ladder of D, the race over m, dimension counts, the data bound."""

import math

import torch

from . import equation, tensors


def concentration(Ds: torch.Tensor, d: int, trials: int, generator, device) -> dict:
    """||P Psi||^2 samples for every D against Beta(d, D - d): mean, variance, KS distance; the 3D shadow."""
    sh = tensors.projected_states(Ds, d, trials, generator, device)
    q = sh["q"].T  # (n_D, T)
    a, b = equation.beta_params(d, Ds.to(device), "complex")
    mean, var = equation.beta_mean_var(a, b)
    ks = equation.ks_distance(q, a, b)
    a3, b3 = equation.beta_params(3, 2 * Ds.to(device), "real")
    ks3 = equation.ks_distance((sh["shadow3"] ** 2).sum(-1).T, a3, b3)
    one = torch.ones(1, dtype=torch.float64, device=device)
    ks_height = equation.ks_distance(((sh["dir3"][:, 2] + 1) / 2)[None, :], one, one).item()  # uniform on S^2
    return {**sh, "Ds": Ds, "a": a, "b": b, "mean": mean, "var": var, "ks": ks, "ks3": ks3, "ks_height": ks_height,
            "q_mean": q.mean(1), "q_var": q.var(1)}  # fmt: skip


def race(K: int, m_max: int, trials: int, generator, device) -> dict:
    """The race for m = 1..m_max hidden dimensions per outcome: quadrature and Monte Carlo, complex and real."""
    w = tensors.born_weights(K, device)  # (K,) |c_k|^2
    m = torch.arange(1, m_max + 1, dtype=torch.float64, device=device)
    P_c = equation.race_probabilities(w.expand(m_max, -1), m)  # (M, K) complex hidden dims: Gamma(m)
    P_r = equation.race_probabilities(w.expand(m_max, -1), m / 2)  # real hidden dims: Gamma(m / 2)
    f_c = equation.race_frequencies(w, tensors.hidden_energies(trials, K, m_max, "complex", generator, device))
    f_r = equation.race_frequencies(w, tensors.hidden_energies(trials, K, m_max, "real", generator, device))
    a_line = torch.logspace(math.log10(0.5), math.log10(m_max), 60, dtype=torch.float64, device=device)
    tv_line = equation.total_variation(equation.race_probabilities(w.expand(60, -1), a_line), w)
    sigma = (P_c * (1 - P_c) / trials).sqrt()
    return {"w": w, "m": m, "P_c": P_c, "P_r": P_r, "f_c": f_c, "f_r": f_r, "a_line": a_line, "tv_line": tv_line,
            "tv_c": equation.total_variation(P_c, w), "tv_r": equation.total_variation(P_r, w),
            "tv_mc_c": equation.total_variation(f_c, w), "tv_mc_r": equation.total_variation(f_r, w),
            "max_pull": ((f_c - P_c).abs() / sigma.clamp_min(1e-12)).max().item()}  # fmt: skip


def dimensions(d: int, K: int, device) -> dict:
    """P5 counts: hidden dimensions for Born statistics, and 2^N for N entangled two-level systems."""
    N = torch.logspace(0, 80, 161, dtype=torch.float64, device=device)  # 1 ... 1e80 particles
    table_N = torch.tensor([2.0, 10.0, 50.0, 300.0, 1e23, 1e80], dtype=torch.float64, device=device)
    return {"N": N, "log10_D": equation.log10_min_dim_entanglement(N), "table_N": table_N,
            "table_log10_D": equation.log10_min_dim_entanglement(table_N),
            "D_born": equation.min_dim_born(d, K, 1)}  # fmt: skip


def sorkin_bound(m_max: int, device) -> dict:
    """kappa of the race model in a triple slit, and the hidden-dimension range the measured kappa allows.

    kappa(a) is computed for the default binning (41 screen bins, each slit sends 5% to the screen) over a range of
    the shape parameter a (complex m = a, real m = 2a), and its slope at a = 1 for 9 binnings; the smallest slope
    gives the most conservative (widest) allowed band |a - 1| <= (|kappa| + 2 sigma) / slope.
    """
    W = equation.slit_born_weights(tensors.slit_amplitudes(41, 0.05, device))  # (7, 42)
    da = torch.logspace(-6, math.log10(m_max - 1), 24, dtype=torch.float64, device=device)
    a = torch.cat([1 - da[da < 0.5].flip(0), torch.ones(1, dtype=torch.float64, device=device), 1 + da])
    kappa = equation.sorkin_kappa(
        equation.race_probabilities(W.expand(len(a), -1, -1), a[:, None], n_grid=1500), pixel=20
    )
    slopes = []
    h = torch.tensor([1 - 1e-3, 1 + 1e-3], dtype=torch.float64, device=device)
    for n_pix in (21, 41, 81):
        for tau in (0.02, 0.05, 0.1):  # 9 binnings: a loop over modelling choices, each batched inside
            Wb = equation.slit_born_weights(tensors.slit_amplitudes(n_pix, tau, device))
            k = equation.sorkin_kappa(
                equation.race_probabilities(Wb.expand(2, -1, -1), h[:, None], n_grid=1500), n_pix // 2
            )
            slopes.append(((k[1] - k[0]) / (h[1] - h[0])).item())
    s_min = min(abs(s) for s in slopes)
    rows = tensors.born_rule_tests()
    limits = [abs(float(r["kappa"])) + (1.0 if r["kind"] == "bound" else 2.0) * float(r["sigma"]) for r in rows]
    allowed = [lim / s_min for lim in limits]  # |a - 1| allowed, 95% (bounds taken as quoted)
    m_int = torch.arange(1, m_max + 1, dtype=torch.float64, device=device)
    k_int = equation.sorkin_kappa(
        equation.race_probabilities(W.expand(2 * m_max, -1, -1), torch.cat([m_int, m_int / 2])[:, None], n_grid=1500),
        pixel=20,
    )
    return {"a": a, "kappa": kappa, "slopes": slopes, "s_min": s_min, "rows": rows, "limits": limits,
            "allowed": allowed, "m_int": m_int, "kappa_complex": k_int[:m_max],
            "kappa_real": k_int[m_max:]}  # fmt: skip
