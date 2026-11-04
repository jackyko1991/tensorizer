"""Main loop: LIGO strain → Welch PSD → whitening → Q-map and ridge fit → batched template bank → M_c, SNR, delay."""

from pathlib import Path

import torch

from problems._common.runtime import cli

from . import plot, solve, tensors
from .equation import inner


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: strains are ~1e-21 and PSDs ~1e-46, and the template phases reach thousands of radians.
    strain = tensors.load_strain()
    x, fs = strain["x"], strain["fs"]
    n = x.shape[-1]
    print(f"GW150914: {n / fs:.0f} s of H1 and L1 strain at {fs:.0f} Hz from GPS {strain['gps_start']:.0f} (GWOSC)")

    res = solve.analyse(
        strain,
        pn_order=params["pn_order"],
        f_low=params["f_low"],
        cutoff=params["cutoff"],
        n_templates=params["n_templates"],
        mc_lo=params["mc_min"],
        mc_hi=params["mc_max"],
        q=params["q"],
        seg=params["welch_segment"],
    )

    # --- checks on the real data ---
    big_x = torch.fft.fft(x[0])
    print(f"Parseval on H1: Σ|x|² / (Σ|X|²/N) = {(x[0].pow(2).sum() / (big_x.abs().pow(2).sum() / n)).item():.12f}")
    f100 = int(100 * n / fs)
    asd = res["psd"][:, f100].sqrt()
    print(f"Welch ASD at 100 Hz: H1 {asd[0]:.2e}, L1 {asd[1]:.2e} strain/√Hz (design-era O1 ≈ 1e-23)")
    w = res["white_bp"]
    psd_w, fw = solve.welch_psd(solve.whiten(x, res["psd"], fs)[:, 1024:-1024], fs, int(fs))
    band = (fw > 30) & (fw < 400)
    flat = (psd_w[:, band].std(-1) / psd_w[:, band].mean(-1)).tolist()
    print(f"whitened PSD 30–400 Hz: relative scatter H1 {flat[0]:.2f}, L1 {flat[1]:.2f} (flat ⇒ only Welch noise)")
    f = res["f"]
    h = solve.template_bank(f, res["bank"], res["psd"][0], params["pn_order"], params["f_low"], params["cutoff"])
    norms = inner(h, h, torch.where(res["psd"][0] > 0, res["psd"][0], torch.inf), f[1] - f[0])
    print(f"template bank: {len(h)} templates, ⟨h|h⟩ between {norms.min():.12f} and {norms.max():.12f}")

    # --- the detection ---
    k = res["k"]
    print(
        f"\nbest template: M_c = {res['mc_bank']:.1f} M☉ (detector frame, PN order {params['pn_order']}, "
        f"cut at {params['cutoff']}), SNR H1 {res['snr'][0]:.1f}, L1 {res['snr'][1]:.1f}, "
        f"network {res['network'][k]:.1f}"
    )
    print(f"merger time: GPS {strain['gps_start'] + res['t_event']:.3f} (GWOSC: {tensors.GPS_EVENT})")
    print(
        f"H1 − L1 delay: {res['dt_ms']:.1f} ms from the SNR peaks, {res['dt_cc_ms']:.1f} ms from the "
        f"cross-correlation of H1 with −L1 (published 6.9 +0.5/−0.4 ms; light travel time ≤ 10 ms)"
    )
    print(
        f"ridge fit (Newtonian df/dt ∝ M_c^(5/3) f^(11/3), {int(res['kept'].sum())} Q-tiles): "
        f"M_c = {res['mc_track']:.1f} M☉"
    )
    print(
        "published: detector-frame M_c ≈ 30 M☉ and network SNR 24 (Abbott et al. 2016); GWTC-1: source-frame "
        "M_c = 28.6 (+1.7/−1.5) M☉ at z ≈ 0.09, i.e. (1+z)M_c ≈ 31 M☉"
    )
    print(f"whitened strain peak |h| ≈ {w[0].abs().max():.1f} σ in H1")
    return plot.animate(res, out_dir / "gw150914_chirp.gif")


if __name__ == "__main__":
    cli(run, __file__)
