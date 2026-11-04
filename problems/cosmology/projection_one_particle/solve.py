"""Solvers: slice a batch of worldlines, survey the open parameters, compare with the observed asymmetry."""

import torch

from . import equation, tensors


def slice_counts(X: torch.Tensor, levels: torch.Tensor, closed: bool) -> dict:
    """Particles (forward crossings), antiparticles (backward), net charge Q and total N on every slice (B, L)."""
    up, down = equation.crossings(X[..., 0], levels, closed)
    return {"up": up, "down": down, "Q": up - down, "N": up + down}


def batch(B, S, modes, roughness, dims, levels, generator, device) -> dict:
    """Closed smooth loops: mean counts per slice against Rice's formula, and the Morse (turning point) count."""
    X = tensors.smooth_worldlines(B, S, modes, roughness, dims, True, generator, device)
    c = slice_counts(X, levels, closed=True)
    rice = equation.rice_mean_crossings(levels, tensors.mode_sigmas(modes, roughness, device))
    morse_ok = torch.equal(c["N"], equation.morse_count(X[..., 0], levels))
    return {**c, "rice": rice, "mean_N": c["N"].double().mean(0), "sem_N": c["N"].double().std(0) / B**0.5,
            "morse_ok": morse_ok, "max_abs_Q": c["Q"].abs().max().item()}  # fmt: skip


def survey(B, S, dims, levels, A_target: float, generator, device) -> list[dict]:
    """Which worldline configurations give an asymmetry |Q| / N >= A_target on a slice with N >= 2 crossings?

    The open parameters are scanned: smooth (Fourier) or random (Brownian) worldlines, closed or open, the number of
    harmonics and the roughness. The loop runs over these modelling choices; each case is a batch of B worldlines.
    """
    cases = [("random", c, 0, 0.0) for c in (True, False)]
    cases += [("smooth", c, k, r) for c in (True, False) for k in (2, 4, 8, 16, 32) for r in (0.5, 1.0, 2.0)]
    rows = []
    for kind, closed, modes, rough in cases:
        X = tensors.worldlines(kind, B, S, max(modes, 1), rough, dims, closed, generator, device)
        c = slice_counts(X, levels, closed)
        many = c["N"] >= 2  # slices that cut the worldline at least twice
        A = c["Q"].abs().double() / c["N"].clamp_min(1).double()
        ok = many & (A >= A_target)
        rows.append({"kind": kind, "closed": closed, "modes": modes, "roughness": rough,
                     "slices": int(many.sum()), "reproduce": int(ok.sum()),
                     "max_A": A[many].max().item() if many.any() else 0.0,
                     "max_N": int(c["N"].max()), "max_abs_Q": int(c["Q"].abs().max())})  # fmt: skip
    return rows


def representative(S, modes, roughness, dims, levels, device, seed: int = 14) -> dict:
    """One closed loop for the picture: the worldline, its slices, and its turning points."""
    g = torch.Generator().manual_seed(seed)
    X = tensors.smooth_worldlines(1, S, modes, roughness, dims, True, g, device)
    c = slice_counts(X, levels, closed=True)
    births, deaths = equation.turning_points(X[..., 0], closed=True)
    return {"X": X[0], **{k: v[0] for k, v in c.items()}, "births": births[0], "deaths": deaths[0]}


def observations(device) -> dict:
    """The observed asymmetry: AMS-02 positron fraction, eta from Planck, the baryon count, BESS antihelium."""
    pf = tensors.positron_fraction(device)
    c = tensors.cited()
    eta = equation.eta_from_omega_b(c["omega_b_h2"])
    N_B = equation.baryons_in_observable_universe(eta, c["n_gamma_cm3"], c["hubble_radius_gly"])
    A = equation.asymmetry_from_fraction(pf["fraction"])
    r = equation.efficiency_ratio(pf["fraction"])
    return {**pf, "A": A, "A_min": A.min().item(), "A_max": A.max().item(), "r": r, "eta": eta, "N_B": N_B,
            "N_gamma": N_B / eta, "cited": c,
            "hidden_over_dm": c["omega_b_h2"] / c["omega_c_h2"]}  # fmt: skip


def winding_demo(device, generator) -> dict:
    """Extension: loops that wind w times round a compact time carry net charge w on every slice."""
    w = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64)
    t = tensors.winding_worldlines(w, 2000, 1.0, 0.3, generator, device)
    levels = torch.linspace(0.05, 0.95, 19, dtype=torch.float64, device=device)
    up, down = equation.crossings_periodic(t, levels, 1.0)
    return {"w": w, "Q": up - down, "N": up + down}
