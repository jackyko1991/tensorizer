"""Main loop: parameters → tensors → simulate photons → G(τ), FLIM and FLCS → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

N_BUILD, N_HOLD = 60, 12  # animation frames while G(τ) builds up, then frames that show the fits
STEPS_PER_FRAME = 10  # Brownian steps between two frames of the molecule movie (50 µs)
REF_PHOTONS = 100_000  # photons in each pure-species reference decay


def measure(params: dict, dev: torch.device, n_snap: int = 1, n_record: int = 0) -> dict:
    """Run the whole virtual experiment and its analysis; returns every tensor the printout and the plot need."""
    gen = torch.Generator(dev).manual_seed(tensors.SEED)
    sp = tensors.species_table(params, dev)
    w0, kappa = tensors.W0, params["kappa"]
    z0 = kappa * w0
    A = equation.shape_matrix(w0, z0, dev)  # (3, 3)
    box = tensors.box_size(w0, z0, dev)  # (3,)
    x, S, D_n = tensors.molecules(sp["N"], sp["D"], box, w0, z0, gen)  # (n, 3), (n, s), (n,)
    q = equation.peak_brightness(sp["eps"])  # (s,) photons/s at the centre
    n_steps = round(params["trace"] / tensors.DT)

    # 1) simulate: Brownian motion → detected rate → Poisson photons with TCSPC microtimes
    rates, positions = solve.brownian_rates(x, S, D_n, q, A, box, tensors.DT, n_steps, gen, n_record, STEPS_PER_FRAME)
    bg = params["background"] * 1e3
    tcspc = (tensors.IRF_T0, tensors.IRF_SIGMA, tensors.PERIOD)
    counts, bins, chans, _source = solve.draw_photons(rates, bg, tensors.DT, sp["tau"], gen, *tcspc, tensors.N_CHANNELS)
    n_s = len(sp["keys"])

    # 2) FLIM: the microtime histogram of the sample plus one reference decay per pure species, fitted together
    #    (shared lifetimes, IRF reconvolution, flat background). The references pin down lifetimes that are too close
    #    to separate from the mixture alone; the mixture row then gives the photon fraction of each species.
    t = tensors.channels(dev)  # (C,) ns
    h = solve.histogram(chans, tensors.N_CHANNELS)  # (C,)
    refs = solve.reference_histograms(sp["tau"], REF_PHOTONS, REF_PHOTONS // 100, gen, *tcspc, tensors.N_CHANNELS)
    tau_sorted, a, b = solve.fit_flim(t, torch.cat([refs, h[None]]), n_s, *tcspc)
    order = sp["tau"].argsort().argsort()  # fits come sorted by lifetime: put them back in species order
    tau_fl, photons, bg_photons = tau_sorted[order], a[-1, order], b[-1]

    # 3) FLCS: lifetime filters from the fitted pure patterns (+ background pattern), one trace per species
    flat = torch.full((len(t), 1), 1.0 / len(t), device=dev, dtype=torch.float64)
    M = torch.cat([equation.decay_pattern(t, tau_fl, *tcspc), flat], 1)  # (C, s+1)
    F = equation.flcs_filters(M, h)  # (s+1, C)
    traces = torch.cat([counts.sum(1, keepdim=True).T.double(), solve.filtered_traces(bins, chans, F[:n_s], n_steps)])

    # 4) correlations on log-spaced lags: row 0 = all photons (plain FCS), rows 1… = FLCS per species
    edges = solve.lag_edges(min(round(0.05 / tensors.DT), n_steps // 20), device=dev)
    lags, G = solve.binned_correlation(traces, edges)  # (L,), (1+s, L)
    lag = lags * tensors.DT  # s
    duration = n_steps * tensors.DT
    corr = equation.background_correction(len(bins) / duration, bg_photons.item() / duration)
    G[0] *= corr
    tauD_all, a_all = solve.fit_fcs(lag, G[0], kappa, n_s)  # plain FCS: n_s components mixed in one curve
    flcs = [solve.fit_fcs(lag, G[1 + k], kappa, 1) for k in range(n_s)]  # FLCS: one clean component each

    # 5) snapshots for the animation: G(τ) and the histogram from the first part of the trace
    fractions = torch.logspace(-1, 0, n_snap).tolist() if n_snap > 1 else [1.0]
    G_snap, h_snap = [], []
    for f in fractions:
        T_f = round(f * n_steps)
        G_f = solve.binned_correlation(traces[:, :T_f], edges)[1]
        G_f[0] *= corr
        G_snap.append(G_f)
        h_snap.append(solve.histogram(chans[bins < T_f], tensors.N_CHANNELS))

    true_tauD = w0**2 / (4 * sp["D"].double())
    return {
        "sp": sp,
        "w0": w0,
        "z0": z0,
        "box": box,
        "S": S,
        "positions": positions,
        "A": A,
        "fine_counts": counts[: n_record - n_record % STEPS_PER_FRAME].reshape(-1, STEPS_PER_FRAME, n_s + 1).sum(1),
        "t": t,
        "h": h,
        "refs": refs,
        "tau_fl": tau_fl,
        "photons": photons,
        "true_photons": counts[:, :n_s].sum(0),
        "bg_photons": bg_photons,
        "M": M,
        "F": F,
        "lag": lag,
        "G": G,
        "tauD_all": solve.match(tauD_all, true_tauD),
        "a_all": solve.match(a_all, true_tauD),
        "tauD_flcs": torch.cat([f[0] for f in flcs]),
        "N_flcs": 1 / torch.cat([f[1] for f in flcs]),
        "true_tauD": true_tauD,
        "fractions": fractions,
        "G_snap": G_snap,
        "h_snap": h_snap,
        "kappa": kappa,
        "duration": duration,
    }


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    if fast:
        params = {**params, "trace": min(params["trace"], 0.3)}
    r = measure(params, dev, n_snap=4 if fast else 24, n_record=(N_BUILD + N_HOLD) * STEPS_PER_FRAME)
    sp, w0 = r["sp"], r["w0"]

    D_flcs = equation.diffusion_coefficient(w0, r["tauD_flcs"])
    D_all = equation.diffusion_coefficient(w0, r["tauD_all"])
    eps = sp["eps"].double()
    N = sp["N"].double()
    N_app_true = (eps * N).sum() ** 2 / (eps**2 * N).sum()  # what plain FCS sees for a brightness-weighted mix
    print(f"trace {r['duration']:.2f} s · {int(r['h'].sum())} photons · κ = {r['kappa']:.1f}")
    print(f"plain FCS: N_app = {1 / r['a_all'].sum().item():.2f} (expected {N_app_true.item():.2f})")
    print(f"{'species':<17} {'D in':>7} {'D FCS':>7} {'D FLCS':>7} {'N in':>5} {'N FLCS':>6} {'τ in':>5} {'τ FLIM':>6}")
    for k, name in enumerate(sp["names"]):
        D_k, N_k, tau_k = sp["D"][k].item(), sp["N"][k].item(), sp["tau"][k].item()
        print(
            f"{name:<17} {D_k:7.1f} {D_all[k].item():7.1f} {D_flcs[k].item():7.1f} "
            f"{N_k:5.2f} {r['N_flcs'][k].item():6.2f} {tau_k:5.2f} {r['tau_fl'][k].item():6.2f}"
        )
    frac_fit = r["photons"] / r["photons"].sum()
    frac_true = r["true_photons"].double() / r["true_photons"].sum()
    print("photon fractions  FLIM " + " / ".join(f"{f:.2f}" for f in frac_fit.tolist()), end="")
    print("   true " + " / ".join(f"{f:.2f}" for f in frac_true.tolist()))
    return plot.animate(r, N_BUILD, N_HOLD, tensors.DT * STEPS_PER_FRAME, out_dir / "point_fcs_flim.gif")


if __name__ == "__main__":
    cli(run, __file__)
