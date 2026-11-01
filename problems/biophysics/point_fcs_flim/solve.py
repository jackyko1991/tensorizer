"""Simulate the confocal experiment (Brownian motion → photons with microtimes) and analyse it (G(τ), FLIM, FLCS)."""

import math

import torch

from problems._common.progress import steps

from . import equation

CHUNK = 2000  # Brownian steps generated at once


# --- simulation -------------------------------------------------------------------------------------------------


def brownian_rates(x, S, D_n, q, A, box, dt, n_steps, gen, n_record=0, record_every=10):
    """Step all molecules together and return the detected rate per species, I_ts (T, s).

    x (n, 3) start positions, S (n, s) membership, D_n (n,) diffusion coefficients, q (s,) peak brightness.
    Each step adds a Gaussian displacement with variance 2 D dt per axis. A chunk of steps is drawn at once and
    summed with cumsum; positions are wrapped back into the periodic box (focus at the centre).
    Also returns positions every `record_every` steps for the first n_record steps (for the animation).
    """
    rates, recorded = [], []
    sigma = torch.sqrt(2 * D_n * dt)[None, :, None]  # (1, n, 1)
    for k in steps(math.ceil(n_steps / CHUNK), "Brownian chunks"):
        K = min(CHUNK, n_steps - k * CHUNK)
        dx = torch.randn(K, *x.shape, generator=gen, device=x.device) * sigma  # (K, n, 3)
        path = torch.remainder(x + torch.cumsum(dx, 0) + box / 2, box) - box / 2
        W = equation.detection(path, A)  # (K, n)
        rates.append(equation.species_rates(W, S, q))  # (K, s)
        if k * CHUNK < n_record:
            recorded.append(path[: n_record - k * CHUNK : record_every])
        x = path[-1]
    rec = torch.cat(recorded) if recorded else x[None]
    return torch.cat(rates), rec


def draw_photons(rates, background, dt, tau, gen, t0, sigma, period, n_channels):
    """Poisson photon counts per bin and a TCSPC microtime for every photon.

    rates (T, s) photons/s; background in photons/s. Returns counts (T, s+1) (last column = background) and, per
    photon, its time bin, microtime channel and source (0 … s-1 species, s background).
    A species-s photon arrives Exp(τ_s) after excitation, smeared by the Gaussian IRF; background is flat in time.
    """
    T, n_s = rates.shape
    lam = torch.cat([rates, torch.full((T, 1), background, device=rates.device)], 1) * dt
    counts = torch.poisson(lam, generator=gen).long()  # (T, s+1)
    bins, chans, source = [], [], []
    for s in range(n_s + 1):
        b = torch.repeat_interleave(torch.arange(T, device=rates.device), counts[:, s])
        lifetime = tau[s] if s < n_s else None  # the last column is background
        bins.append(b)
        chans.append(microtime_channels(len(b), lifetime, gen, t0, sigma, period, n_channels, rates.device))
        source.append(torch.full((len(b),), s, device=rates.device))
    return counts, torch.cat(bins), torch.cat(chans), torch.cat(source)


def microtime_channels(n, tau, gen, t0, sigma, period, n_channels, device):
    """TCSPC channels of n photons: delay Exp(τ) after the pulse plus Gaussian IRF jitter, wrapped into one laser
    period; τ = None means background, spread evenly over the period."""
    if tau is None:
        micro = torch.rand(n, generator=gen, device=device, dtype=torch.float64) * period
    else:
        delay = torch.empty(n, device=device, dtype=torch.float64).exponential_(generator=gen) * tau
        irf = t0 + sigma * torch.randn(n, generator=gen, device=device, dtype=torch.float64)
        micro = torch.remainder(delay + irf, period)
    return (micro / period * n_channels).long().clamp(max=n_channels - 1)


def reference_histograms(tau, n_photons, n_background, gen, t0, sigma, period, n_channels):
    """Decays of the pure species, measured separately (a dye solution, a protein sample) as in FLCS practice:
    (s, C), each with n_photons fluorescence photons plus n_background background photons."""
    out = []
    for lifetime in tau.tolist():
        chans = torch.cat(
            [
                microtime_channels(n_photons, lifetime, gen, t0, sigma, period, n_channels, tau.device),
                microtime_channels(n_background, None, gen, t0, sigma, period, n_channels, tau.device),
            ]
        )
        out.append(histogram(chans, n_channels))
    return torch.stack(out)


# --- analysis ---------------------------------------------------------------------------------------------------


def histogram(chans, n_channels, weights=None):
    """TCSPC histogram h_c (C,)."""
    return torch.bincount(chans, weights=weights, minlength=n_channels).double()


def filtered_traces(bins, chans, F, T):
    """I_k(t) = Σ_{photons p in bin t} f_k(c_p): one intensity trace per filter, (K, T)."""
    out = torch.zeros(F.shape[0], T, device=F.device, dtype=F.dtype)
    return out.index_add_(1, bins, F[:, chans])


def lag_edges(max_lag, per_decade=12, device=None):
    """Integer lag windows [e_l, e_{l+1}) spaced logarithmically from 1 step to max_lag steps."""
    e = torch.logspace(0, math.log10(max_lag), int(per_decade * math.log10(max_lag)) + 1, device=device)
    return torch.unique(torch.round(e).long())


def binned_correlation(trace, edges):
    """G averaged over each log-spaced lag window: trace (K, T) → lags (L,) in steps and G (K, L).

    A binned stand-in for the multi-tau correlator: all lags come from one FFT, then neighbouring lags are averaged
    so the long lags (many, noisy, slowly varying) get more averaging than the short ones.
    """
    G = equation.autocorrelation(trace)
    csum = torch.cumsum(G, -1)
    lo, hi = edges[:-1], edges[1:]
    G_bin = (csum[..., hi - 1] - csum[..., lo - 1]) / (hi - lo)
    lags = (lo + hi - 1).double() / 2  # mean lag in each window
    return lags, G_bin


def _combos(grid, n_comp):
    """Starting guesses: every value of the grid (1 component) or every pair a < b (2 components)."""
    if n_comp == 1:
        return grid[:, None]
    i, j = torch.triu_indices(len(grid), len(grid), 1, device=grid.device)
    return torch.stack([grid[i], grid[j]], -1)


def _project(Phi, y, w):
    """Best linear amplitudes a for y_h ≈ Phi a_h (weights w_h) for every data set h, and the total weighted χ².

    Phi (..., L, K) shared basis, y and w (H, L) → χ² (...) and a (..., H, K). l is summed in the normal equations.
    """
    A = torch.einsum("...lk,hl,...lm->...hkm", Phi, w, Phi)
    b = torch.einsum("...lk,hl,hl->...hk", Phi, w, y)
    ridge = 1e-10 * A.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * torch.eye(A.shape[-1], device=y.device)
    a = torch.linalg.solve(A + ridge, b[..., None])[..., 0]
    r = y - torch.einsum("...lk,...hk->...hl", Phi, a)
    return torch.einsum("...hl,hl,...hl->...", r, w, r), a


def fit_times(basis, y, w, grid, n_comp, n_iter=100):
    """Fit y_h ≈ Σ_k a_hk φ_k(τ): amplitudes a (H, K) are linear (solved exactly), the shared times τ nonlinear.

    1) every combination of τ on a log grid at once (a batched least-squares problem) → best start;
    2) refine log τ with autograd and L-BFGS (two close exponentials leave a long, flat χ² valley that plain
       gradient steps crawl along). Returns τ sorted ascending (n_comp,) and the amplitudes a in the same order.
    """
    chi2, _ = _project(basis(_combos(grid, n_comp)), y, w)
    log_tau = _combos(grid, n_comp)[chi2.argmin()].log().clone().requires_grad_(True)
    scale = chi2.min().item()  # keep the objective O(1) for the line search
    opt = torch.optim.LBFGS([log_tau], max_iter=n_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _project(basis(log_tau.exp()), y, w)[0] / scale
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        tau = log_tau.exp().sort().values
        _, a = _project(basis(tau), y, w)
    return tau, a


def fit_fcs(lag, G, kappa, n_comp):
    """G(τ) = Σ_s a_s g(τ; τ_{D,s}) by least squares: returns τ_D (S,) ascending and amplitudes a (S,).

    For a single species a = G(0) = 1/N.
    """
    grid = torch.logspace(-6.5, 0, 50, device=G.device, dtype=G.dtype)
    tau_D, a = fit_times(
        lambda tau: equation.diffusion_model(lag, tau, kappa), G[None], torch.ones_like(G)[None], grid, n_comp
    )
    return tau_D, a[0]


def fit_flim(t, h, n_comp, t0, sigma, period):
    """Global IRF-reconvolution fit of several histograms h (H, C) with shared lifetimes:

    h_hc ≈ Σ_s a_hs p_c(τ_s) + b_h / C. Returns τ (S,) ascending, photons per species a (H, S) and background b (H,).
    Weights 1/h (Poisson variance); the flat background is one more linear column.
    """
    flat = torch.full((len(t), 1), 1.0 / len(t), device=h.device, dtype=h.dtype)

    def basis(tau):
        p = equation.decay_pattern(t, tau, t0, sigma, period)  # (..., C, S)
        return torch.cat([p, flat.expand(*p.shape[:-1], 1)], -1)

    grid = torch.logspace(math.log10(0.2), math.log10(15), 60, device=h.device, dtype=h.dtype)
    tau, a = fit_times(basis, h, 1 / h.clamp(min=1), grid, n_comp)
    return tau, a[:, :-1], a[:, -1]


def match(fitted_sorted, true):
    """Put components fitted in ascending order into the order of the true species (by rank)."""
    return fitted_sorted[true.argsort().argsort()]
