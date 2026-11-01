"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    e = tensors.energies(dev)  # (E,) keV
    mu = tensors.attenuation_table(params["iodine_mg_ml"], dev)  # (E, M)
    mu_chosen = tensors.interpolate_loglog(e, mu, params["energy_kev"])  # (M,)
    x, y, masks = tensors.phantom(params["n_pixels"], dev)  # masks (M, Y, X)
    dl = tensors.path_weights(y)  # (Y,)
    spectrum = tensors.tube_spectrum(params["kvp"], e)  # (E,)

    # Table energies plus the chosen energy as the last row.
    mu_all = torch.cat([mu, mu_chosen[None]])  # (E+1, M)
    e_all = torch.cat([e, torch.tensor([float(params["energy_kev"])], device=dev)])
    lengths, mono, poly = solve.projections(mu, mu_chosen, masks, dl, spectrum)
    mu_maps = torch.einsum("em,myx->eyx", mu_all, masks)  # μ(y, x) at every energy
    profiles = torch.stack([solve.ray_profiles(m, masks, dl) for m in mu_all])  # (E+1, Y, X)
    contrast = solve.contrast_vs_energy(mu)
    hardening = solve.beam_hardening(mu[:, 2], spectrum, e)

    centre = len(x) // 2
    print(
        f"energy {params['energy_kev']:.0f} keV:  μ(tissue) = {mu_chosen[2]:.3f} /cm, μ(bone) = {mu_chosen[3]:.3f} /cm"
    )
    print(f"half-value layer in soft tissue: {solve.half_value_layer(float(mu_chosen[2])):.2f} cm")
    print(
        f"central ray crosses {', '.join(f'{n} {lengths[m, centre]:.1f} cm' for m, n in enumerate(tensors.MATERIALS))}"
    )
    print(f"central ray: I/I0 = {mono[-1, centre]:.2e} (mono), {poly[centre]:.2e} ({params['kvp']:.0f} kVp spectrum)")
    for k, ek in enumerate(e.tolist()):
        c = contrast[k]
        print(f"  {ek:5.0f} keV  contrast  lung {c[0]:+.3f}  bone {c[1]:+.3f}  iodine {c[2]:+.3f}")
    d, _, mu_eff, e_mean = hardening
    print(
        f"beam hardening: μ_eff {mu_eff[0]:.3f} → {mu_eff[-1]:.3f} /cm, "
        f"mean energy {e_mean[0]:.0f} → {e_mean[-1]:.0f} keV"
    )

    depth = y - y[0] + (y[1] - y[0]) / 2
    return plot.animate(
        x, depth, mu_maps, profiles, mono, poly, e_all, contrast, hardening, out_dir / "xray_attenuation.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
