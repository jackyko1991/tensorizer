"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

DZ = 0.04  # depth step (cm)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    I_mev = params["i_ev"] * 1e-6
    spread = params["energy_spread"] / 100.0
    z1, z2 = params["target_start"], max(params["target_end"], params["target_start"] + 1.0)

    # 1) A batch of pencil beams: slow down, find the range, compute the depth-dose curves.
    E0 = tensors.beam_energies(params["e_min"], params["e_max"], params["n_energies"], dev)  # (B,)
    z_max = max(1.15 * float(equation.bragg_kleeman(E0.max(), 0.0022, 1.77)) + 2.0, z2 + 8.0)
    z, E = solve.slow_down(E0, I_mev, z_max, DZ)  # (Z+1,), (Z+1, B)
    ranges = solve.csda_range(z, E)
    z_mid, dose = solve.depth_dose(z, E, ranges, spread)  # (B, Z)
    times = solve.flight_times(z, E)
    alpha, p = solve.fit_bragg_kleeman(E0, ranges) if len(E0) > 1 else (0.0022, 1.77)

    # 2) Spread-out Bragg peak: energy layers whose ranges cover the target, non-negative weights.
    E_layers = tensors.layer_energies(tensors.layer_ranges(z1, z2, params["n_layers"], dev), alpha, p)
    zl, El = solve.slow_down(E_layers, I_mev, z_max, DZ)
    _, layer_dose = solve.depth_dose(zl, El, solve.csda_range(zl, El), spread)  # (K, Z)
    w = solve.sobp_weights(layer_dose, z_mid, z1, z2)
    plateau = equation.sobp(layer_dose, w)
    xray = equation.xray_depth_dose(z_mid)
    centre = (z_mid - 0.5 * (z1 + z2)).abs().argmin()
    xray = xray / xray[centre]  # same dose as the protons in the middle of the tumour

    print(f"stopping power at 150 MeV: {float(equation.stopping_power(E0.new_tensor(150.0), I_mev)):.2f} MeV/cm")
    for e, r, zp in zip(E0.tolist(), ranges.tolist(), z_mid[dose.argmax(1)].tolist()):
        print(f"  E0 = {e:6.1f} MeV  range {r:6.2f} cm  dose peak at {zp:6.2f} cm")
    print(f"Bragg–Kleeman fit: R = {alpha:.5f} E^{p:.3f}")
    beyond = z_mid > z2 + 1.0
    print(
        f"SOBP over {z1:.1f}–{z2:.1f} cm with {len(w)} layers: "
        f"flatness {100 * solve.flatness(plateau, z_mid, z1, z2):.2f} %"
    )
    near = (z_mid - 1.0).abs().argmin()  # 1 cm deep, past the photon build-up
    print(f"dose at 1 cm depth: protons {float(plateau[near]):.2f}, X-rays {float(xray[near]):.2f}  (target = 1)")
    print(
        f"mean dose beyond the target: protons {float(plateau[beyond].mean()):.3f}, "
        f"X-rays {float(xray[beyond].mean()):.3f}"
    )

    layers = layer_dose * w[:, None]  # (K, Z) each layer's weighted contribution
    return plot.animate(
        z,
        times,
        E0,
        ranges,
        z_mid,
        dose,
        (alpha, p),
        tensors.reference_ranges(dev),
        layers,
        xray,
        (z1, z2),
        out_dir / "proton_bragg_peak.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
