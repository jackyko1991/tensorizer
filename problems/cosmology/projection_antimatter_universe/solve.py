"""Solvers: evaluate each criterion against the cited data and collect the verdict table."""

import math

import torch

from . import equation, tensors

BOX_GPC = 30.0  # side of the domain picture, a bit more than the observable universe (diameter 28.6 Gpc)


def nearby(c: dict) -> dict:
    """Criterion 1: antinuclei in cosmic rays and gamma rays from clusters."""
    f_max = equation.cluster_antimatter_limit(c)
    return {"antihelium": c["antihelium_to_helium"], "f_cluster": f_max,
            "photons_100": equation.photons_above(100.0)}  # fmt: skip


def domain_sweep(c: dict, speed_factor: float, z_max: float, d_mpc: torch.Tensor) -> dict:
    """Criterion 2: predicted background for every domain size d, the observed background, and the minimum d."""
    E = tensors.energies()
    J = equation.annihilation_intensity(E, d_mpc, c, speed_factor, z_max)  # (D, E)
    res = equation.minimum_domain_size(c, speed_factor, z_max)
    ratio = res["d_min_mpc"] / d_mpc  # (D,) worst predicted / observed: exactly 1/d
    return {"E": E, "J": J, "comptel": equation.comptel_intensity(E, c), "fermi": equation.fermi_intensity(E, c),
            "d": d_mpc, "ratio": ratio, "d_min_mpc": res["d_min_mpc"], "E_worst_mev": res["E_worst_mev"]}  # fmt: skip


def domain_pictures(d_mpc: torch.Tensor, n_pix: int, generator) -> torch.Tensor:
    """(D, N, N) matter/antimatter maps over a BOX_GPC box for the domain sizes d (one noise, so domains grow)."""
    noise = torch.randn(n_pix, n_pix, generator=generator, dtype=torch.float64)
    d_pix = d_mpc / (1e3 * BOX_GPC / n_pix)
    return equation.domains(noise, d_pix)


def fine_tuning(c: dict, generator) -> dict:
    """Criterion 3: how unlikely a random single worldline makes our patch. N_B from eta and the horizon volume."""
    eta = 273.9e-10 * c["planck_omega_b_h2"]
    R = c["horizon_gpc"] * 1e3 * equation.MPC_CM
    N_gamma = c["n_gamma_cm3"] * 4 / 3 * math.pi * R**3
    N_B = eta * N_gamma
    t = lambda x: torch.tensor([x], dtype=torch.float64)  # noqa: E731
    today = equation.log10_tail(t(N_B), t(N_B)).item()  # all ~10^80 crossings the same sign (no antimatter seen)
    early = equation.log10_tail(t(N_B), t(N_gamma)).item()  # excess 10^80 among ~10^89 pairs (early universe)
    M = torch.tensor([10, 100, 1000])
    Q = equation.random_net_charge(M, 4000, generator)
    return {"N_B": N_B, "N_gamma": N_gamma, "log10_today": today, "log10_early": early, "M": M,
            "var": Q.double().var(1)}  # fmt: skip


def mirror(c: dict, dims: int, winding: int, generator) -> dict:
    """Criterion 4: one winding worldline across the CPT mirror, and the neutrino and r tests."""
    w = torch.tensor([winding])
    eta = tensors.winding_conformal_loop(w, 1200, wiggle=0.35, generator=generator)[0]  # (S + 1,)
    s = torch.linspace(0, 2 * math.pi, len(eta), dtype=torch.float64)
    k = torch.arange(1, 4, dtype=torch.float64)
    coef = torch.randn(dims - 1, 3, 2, generator=generator, dtype=torch.float64) / k[:, None]
    basis = torch.stack([torch.cos(k[:, None] * s), torch.sin(k[:, None] * s)], -1)
    space = torch.einsum("ikc,ksc->si", coef, basis)  # (S + 1, dims - 1): the other coordinates; only eta matters
    levels = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    q_ours, q_mirror = equation.charges_across_mirror(eta[None], levels, period=2.0)
    zero = torch.zeros(1, dtype=torch.float64)
    m = torch.logspace(-4, 0, 200, dtype=torch.float64)
    nu = {
        "sum_no0": equation.sum_mnu(zero, "normal", c).item(),
        "sum_io0": equation.sum_mnu(zero, "inverted", c).item(),
        "mb_no0": equation.m_beta(zero, "normal", c).item(),
        "mb_io0": equation.m_beta(zero, "inverted", c).item(),
        "m": m, "sum_no": equation.sum_mnu(m, "normal", c), "sum_io": equation.sum_mnu(m, "inverted", c),
    }  # fmt: skip
    return {"eta": eta, "x": space[:, 0], "levels": levels, "q_ours": q_ours[0], "q_mirror": q_mirror[0],
            "w": winding, **nu}  # fmt: skip


def verdicts(c: dict, near: dict, dom: dict, ft: dict, mi: dict) -> list[dict]:
    """The verdict table: what each criterion requires, its status, and what would falsify it."""
    no_ok = mi["sum_no0"] < c["desi_dr2_sum_mnu"]
    io_ok = mi["sum_io0"] < c["desi_dr2_sum_mnu"]
    return [
        {"criterion": "1 nothing mixed in nearby", "status": "satisfied",
         "detail": f"antiHe/He < {near['antihelium']:.1e}; Coma antiproton fraction < {near['f_cluster']:.1e}",
         "falsify": "one antihelium nucleus (AMS-02) or annihilation gamma rays from a cluster"},
        {"criterion": "2 no domains inside the horizon", "status": "constrained",
         "detail": f"domains must be >= {dom['d_min_mpc'] / 1e3:.0f} Gpc (horizon radius {c['horizon_gpc']} Gpc)",
         "falsify": "a diffuse MeV excess or CMB distortion pointing to nearer walls; untestable beyond the horizon"},
        {"criterion": "3 not fine-tuned", "status": "fails without a mechanism",
         "detail": f"log10 P = {ft['log10_early']:.1e} (early) to {ft['log10_today']:.1e} (today)",
         "falsify": "not a test but a cost: only a mechanism (a CPT boundary at t = 0 plus winding) removes it"},
        {"criterion": "4a lightest neutrino massless", "status": "constrained (consistent)",
         "detail": f"sum m = {mi['sum_no0']:.4f} eV (NO, allowed: {no_ok}), {mi['sum_io0']:.4f} eV (IO, allowed: "
                   f"{io_ok}) vs DESI DR2 < {c['desi_dr2_sum_mnu']} eV",
         "falsify": "a measured lightest mass > 0 (e.g. sum m above the floor + m_light, or KATRIN/Project 8 signal)"},
        {"criterion": "4b dark matter = 4.8e8 GeV right-handed neutrino", "status": "untestable now",
         "detail": f"m_N = {c['bft_dm_mass_gev']:.1e} GeV, far beyond colliders; no direct-detection signal expected",
         "falsify": "dark matter identified as something else (e.g. a WIMP or axion signal)"},
        {"criterion": "4c no primordial gravitational waves", "status": "consistent",
         "detail": f"r = 0 vs BICEP/Keck r < {c['bicep_keck_r']}",
         "falsify": f"primordial B-modes with r >> {c['future_sigma_r']:.0e} (LiteBIRD, Simons Observatory)"},
    ]  # fmt: skip
