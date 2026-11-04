"""The search: Welch PSD, whitening, band-pass, Q-transform map, ridge fit with autograd, batched matched filter."""

import cmath
import math

import torch

from problems._common.progress import steps

from . import tensors
from .equation import (
    T_SUN,
    chirp_frequency,
    inner,
    isco_frequency,
    peak_frequency,
    snr_series,
    spa_template,
)


def welch_psd(x: torch.Tensor, fs: float, seg_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided PSD by Welch's method: Hann-windowed segments with 50 % overlap, averaged.   (…, n) → (…, F), (F,)"""
    seg = x.unfold(-1, seg_len, seg_len // 2)  # (…, S, L) overlapping segments
    w = torch.hann_window(seg_len, periodic=True, dtype=x.dtype)
    spec = torch.fft.rfft((seg - seg.mean(-1, keepdim=True)) * w, dim=-1)  # (…, S, F)
    psd = spec.abs().pow(2).mean(-2) * 2.0 / (fs * w.pow(2).sum())  # 2/(fs Σw²): one-sided, density per Hz
    psd[..., 0] /= 2  # DC and Nyquist are not doubled
    psd[..., -1] /= 2
    return psd, torch.fft.rfftfreq(seg_len, 1 / fs).to(x.dtype)


def interpolate(f_new: torch.Tensor, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of y(f) (…, F) onto f_new (F2,): (…, F2)."""
    i = torch.searchsorted(f, f_new).clamp(1, len(f) - 1)
    w = ((f_new - f[i - 1]) / (f[i] - f[i - 1])).clamp(0, 1)
    return y[..., i - 1] + w * (y[..., i] - y[..., i - 1])


def whiten(x: torch.Tensor, psd: torch.Tensor, fs: float, alpha: float = 0.1) -> torch.Tensor:
    """x̃(f)/√(S(f) fs/2): unit-variance white noise where the PSD model is right.   x (…, n), psd (…, n//2+1)."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x * tensors.tukey(n, alpha), dim=-1)
    return torch.fft.irfft(spec / (psd * fs / 2).sqrt(), n=n, dim=-1)


def bandpass(x: torch.Tensor, fs: float, lo: float, hi: float, taper: float = 5.0) -> torch.Tensor:
    """Zero-phase band-pass in the frequency domain with raised-cosine edges `taper` Hz wide.   (…, n) → (…, n)"""
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / fs).to(x.dtype)
    rise = ((f - lo + taper) / taper).clamp(0, 1)
    fall = ((hi + taper - f) / taper).clamp(0, 1)
    mask = 0.5 * (1 - torch.cos(torch.pi * rise)) * 0.5 * (1 - torch.cos(torch.pi * fall))
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=n, dim=-1)


def template_bank(f, mc_msun, psd, pn_order: int = 2, f_low: float = 20.0, cutoff: str = "peak"):
    """Normalised SPA templates for every chirp mass, all at once: f (F,), mc (K,) M☉, psd (F,) → (K, F), ⟨h|h⟩ = 1.

    cutoff 'peak': stop at the frequency where an equal-mass merger peaks (M ω₂₂ ≈ 0.36); 'isco': at f_ISCO.
    """
    mc = mc_msun[:, None] * T_SUN  # (K, 1) seconds
    f_high = peak_frequency(mc) if cutoff == "peak" else isco_frequency(mc)
    h = spa_template(f, mc, f_low, f_high, pn_order)  # (K, F)
    psd_safe = torch.where(psd > 0, psd, torch.full_like(psd, float("inf")))
    norm = inner(h, h, psd_safe, f[1] - f[0]).sqrt()  # (K,)
    return h / norm[:, None]


def injection(f, mc_msun: float, t_c: float, snr: float, psd, n: int, fs: float, pn_order: int = 2) -> torch.Tensor:
    """Time series of a template with ⟨s|s⟩^½ = snr that merges at t_c: (n,)."""
    h = template_bank(f, torch.tensor([mc_msun], dtype=f.dtype), psd, pn_order)[0]
    h = snr * h * torch.exp(-2j * math.pi * f * t_c)  # time shift by t_c
    return torch.fft.irfft(h * fs, n=n)  # continuous-FT convention: x̃ = Δt·DFT(x)


def matched_filter(x, mc_msun, psd, fs, pn_order: int = 2, f_low: float = 20.0, cutoff: str = "peak", edge=1.0):
    """Filter every detector against every template in one batch.   x (D, n), psd (D, n//2+1) on the rfft grid.

    Returns |z| (D, K, n), the peak SNR (D, K) and its time (D, K) (refined by a parabola through the
    three highest samples), searched away from the first and last `edge` seconds.
    """
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / fs).to(x.dtype)
    df = f[1] - f[0]
    data_f = torch.fft.rfft(x * tensors.tukey(n, 0.1), dim=-1) / fs  # x̃(f) ≈ Δt·DFT
    banks = [template_bank(f, mc_msun, psd[d], pn_order, f_low, cutoff) for d in range(x.shape[0])]
    h = torch.stack(banks)  # (D, K, F): normalised with each detector's own PSD
    psd_safe = torch.where(psd > 0, psd, torch.full_like(psd, float("inf")))
    rho = torch.stack([snr_series(data_f[d : d + 1], h[d], psd_safe[d : d + 1], df, n)[0] for d in range(len(h))])
    rho = rho.abs()  # (D, K, n)
    lo, hi = int(edge * fs), n - int(edge * fs)
    peak, j = rho[..., lo:hi].max(-1)
    j = j + lo
    y0, y1, y2 = (rho.gather(-1, (j + s)[..., None])[..., 0] for s in (-1, 0, 1))
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # vertex of the parabola through 3 samples
    return {"rho": rho, "snr_peak": peak, "t_peak": (j + shift) / fs}


def qtransform(white: torch.Tensor, fs: float, freqs: torch.Tensor, q: float = 8.0) -> torch.Tensor:
    """Normalised energy of Gaussian band-pass filters centred on each frequency (a Q-transform): (Fq, n).

    The filter at f₀ has width σ_f = f₀/q, i.e. a time resolution σ_t = q/(2π f₀); every row is divided by its median.
    """
    n = white.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / fs).to(white.dtype)
    spec = torch.fft.rfft(white)  # (F,)
    window = torch.exp(-0.5 * ((f[None] - freqs[:, None]) / (freqs[:, None] / q)) ** 2)  # (Fq, F)
    tiles = torch.fft.ifft(spec[None] * window, n=n, dim=-1)  # (Fq, n) analytic signal of each band
    energy = tiles.abs().pow(2)
    return energy / energy.median(-1, keepdim=True).values


def ridge(energy, t, freqs, t_lo: float, t_hi: float, threshold: float = 6.0):
    """Frequency of the loudest tile in every time column of [t_lo, t_hi] above threshold: (t_k, f_k) points."""
    cols = (t >= t_lo) & (t <= t_hi)
    e = energy[:, cols]  # (Fq, T')
    best, i = e.max(0)
    keep = best > threshold
    return t[cols][keep], freqs[i[keep]]


def fit_track(t_pts, f_pts, tc0: float, mc0: float = 25.0, n_steps: int = 400, lr: float = 0.02, clip: float = 0.3,
              free_tc: bool = True):  # fmt: skip
    """Fit f(t) = f_Newton(t_c − t; M_c) to ridge points with autograd (Adam on ln M_c, and t_c if free_tc).

    With free_tc=False, t_c is held at the matched-filter time (M_c and t_c are degenerate on a short ridge: a
    later t_c with a heavier M_c fits almost as well).

    Minimises the mean of r_k² with r_k = ln f_model − ln f_k, in two rounds: after the first, points with
    |r_k| > clip (stray noise tiles) are dropped. Returns (M_c in M☉, t_c, loss history, mask of points kept).
    """
    log_mc = torch.tensor(math.log(mc0), dtype=t_pts.dtype, requires_grad=True)
    tc = torch.tensor(tc0, dtype=t_pts.dtype, requires_grad=free_tc)
    keep = torch.ones_like(t_pts, dtype=torch.bool)
    history = []
    for _ in range(2):
        groups = [{"params": [log_mc], "lr": lr}] + ([{"params": [tc], "lr": lr * 0.01}] if free_tc else [])
        opt = torch.optim.Adam(groups)
        for _ in steps(n_steps, "ridge fit"):
            opt.zero_grad()
            tau = (tc - t_pts[keep]).clamp(min=1e-4)
            r = chirp_frequency(tau, log_mc.exp() * T_SUN).log() - f_pts[keep].log()
            loss = r.pow(2).mean()
            loss.backward()
            opt.step()
            history.append(loss.item())
        with torch.no_grad():
            tau = (tc - t_pts).clamp(min=1e-4)
            keep = (chirp_frequency(tau, log_mc.exp() * T_SUN).log() - f_pts.log()).abs() < clip
    return log_mc.exp().item(), tc.item(), history, keep


def time_offset(white_bp: torch.Tensor, fs: float, t0: float, half: float = 0.15, max_lag: float = 0.012):
    """H1–L1 delay (s) from the cross-correlation of H1 with −L1 around t0 (L1 is rotated ≈ 180° relative to H1)."""
    a = int((t0 - half) * fs)
    b = int((t0 + half) * fs)
    h, ell = white_bp[0, a:b], -white_bp[1]
    lags = torch.arange(-int(max_lag * fs), int(max_lag * fs) + 1)
    idx = torch.arange(a, b)[None] - lags[:, None]  # (Lags, T): L1 sample that lines up with each H1 sample
    cc = (h[None] * ell[idx]).sum(-1) / (h.norm() * ell[a:b].norm())
    k = int(cc.argmax().clamp(1, len(cc) - 2))
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    shift = (0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)).item()
    return (lags[k].item() + shift) / fs, lags.double() / fs, cc


def analyse(strain: dict, pn_order: int = 2, f_low: float = 20.0, cutoff: str = "peak", n_templates: int = 81,
            mc_lo: float = 15.0, mc_hi: float = 55.0, q: float = 5.66, seg: float = 4.0) -> dict:  # fmt: skip
    """The whole pipeline on both detectors; returns every intermediate tensor the plot needs."""
    x, fs = strain["x"], strain["fs"]
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / fs).to(x.dtype)
    psd_w, fw = welch_psd(x, fs, int(seg * fs))  # (2, Fw)
    psd = interpolate(f, fw, psd_w)  # (2, F) on the full-length grid
    white = whiten(x, psd, fs)
    white_bp = bandpass(white, fs, 35.0, 350.0)
    bank = tensors.chirp_mass_grid(mc_lo, mc_hi, n_templates)
    mf = matched_filter(x, bank, psd, fs, pn_order, f_low, cutoff)
    network = mf["snr_peak"].pow(2).sum(0).sqrt()  # (K,) combined SNR (peak per detector, quadrature sum)
    k = int(network.argmax())
    t_h, t_l = mf["t_peak"][0, k].item(), mf["t_peak"][1, k].item()
    dt_cc, lags, cc = time_offset(white_bp, fs, t_h)
    freqs = torch.logspace(math.log10(30.0), math.log10(400.0), 60, dtype=x.dtype)
    qmaps = torch.stack([qtransform(white[0], fs, freqs, q), qtransform(-white[1], fs, freqs, q)])  # (2, Fq, n)
    both = qmaps[0] + torch.roll(qmaps[1], round(dt_cc * fs), dims=-1)  # L1 moved onto H1's clock
    t = strain["t"]
    t_pts, f_pts = ridge(both[:, ::8], t[::8], freqs, t_h - 0.15, t_h - 0.01, 8.0)  # one column per 4 ms
    mc_track, tc_track, hist, kept = fit_track(t_pts, f_pts, t_h)
    # the best template in the time domain, whitened like the data, for the overlay
    h_best = template_bank(f, bank[k : k + 1], psd[0], pn_order, f_low, cutoff)[0]
    h_best = h_best * torch.exp(-2j * math.pi * f * t_h)
    phase = _best_phase(x[0], h_best, psd[0], fs)
    h_t = torch.fft.irfft(mf["snr_peak"][0, k] * h_best * cmath.exp(1j * phase) * fs, n=n)
    template_white = bandpass(whiten(h_t[None], psd[:1], fs), fs, 35.0, 350.0)[0]
    return {
        "f": f, "psd": psd, "fw": fw, "psd_w": psd_w, "white_bp": white_bp, "t": t, "bank": bank,
        "snr_bank": mf["snr_peak"], "network": network, "k": k, "mc_bank": bank[k].item(),
        "snr": mf["snr_peak"][:, k].tolist(), "t_event": t_h, "dt_ms": (t_h - t_l) * 1e3, "dt_cc_ms": dt_cc * 1e3,
        "lags": lags, "cc": cc, "rho_best": mf["rho"][:, k], "freqs": freqs, "qmap": qmaps[0], "t_pts": t_pts,
        "kept": kept, "f_pts": f_pts, "mc_track": mc_track, "tc_track": tc_track, "fit_history": hist,
        "template_white": template_white,
    }  # fmt: skip


def _best_phase(x, h, psd, fs) -> float:
    """Phase that maximises Re⟨d|h e^{iφ}⟩ for a template already shifted to the peak time."""
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / fs).to(x.dtype)
    d = torch.fft.rfft(x * tensors.tukey(n, 0.1)) / fs
    psd_safe = torch.where(psd > 0, psd, torch.full_like(psd, float("inf")))
    z = 4 * (f[1] - f[0]) * (d * h.conj() / psd_safe).sum()
    return math.atan2(z.imag.item(), z.real.item())
