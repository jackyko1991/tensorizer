"""Main loop: Nernst grid over (T, Q) → Debye–Hückel activities → EMF → Poisson–Boltzmann ion clouds → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    E0, n, z, a = params["E0"], tensors.N_ELECTRONS, tensors.Z_ION, params["a_ion"]
    T = torch.tensor(params["T_C"] + 273.15, device=dev)  # the working temperature (K)

    # 1) ideal Nernst: EMF for every temperature t and reaction quotient q at once
    T_batch = tensors.temperatures(dev)  # (T,)
    log_Q = tensors.log_ratios(dev)  # (Q,)
    E_tq = equation.nernst(E0, T_batch, log_Q, n)  # (T, Q)

    # 2) Debye–Hückel: the Zn²⁺ half-cell molality is swept, the Cu²⁺ half-cell is fixed
    m_zn = tensors.zinc_molalities(60, dev)  # (C,)
    m_cu = torch.tensor(params["m_Cu"], device=dev)
    I_zn, I_cu = tensors.ionic_strength(m_zn), tensors.ionic_strength(m_cu)
    A, B = equation.debye_A(T), equation.debye_B(T)
    lg_lim_zn, lg_lim_cu = equation.log10_gamma_limiting(z, I_zn, A), equation.log10_gamma_limiting(z, I_cu, A)
    lg_ext_zn, lg_ext_cu = (equation.log10_gamma_extended(z, ion, A, B, a) for ion in (I_zn, I_cu))
    E_ideal = equation.emf(E0, T, m_zn, m_cu, 0.0, 0.0, n)  # (C,)
    E_lim = equation.emf(E0, T, m_zn, m_cu, lg_lim_zn, lg_lim_cu, n)
    E_ext = equation.emf(E0, T, m_zn, m_cu, lg_ext_zn, lg_ext_cu, n)

    # 3) the ion cloud: linearised Poisson–Boltzmann around a Zn²⁺ ion, one radial grid per concentration
    kap = equation.kappa(I_zn, T)  # (C,) nm⁻¹
    M = 120 if fast_mode() else params["n_points"]
    r, h = tensors.radial_grids(kap, a, M)  # (C, M), (C,)
    phi = solve.solve_pb(r, h, kap, a)  # (C, M) in units of q/4πε₀ε_r (nm⁻¹)
    phi_exact = equation.yukawa(r, kap, a)
    l_B = equation.bjerrum_length(T)
    lg_pb = equation.ln_gamma_from_potential(phi[:, 0], a, z, l_B) / equation.LN10  # (C,) log₁₀ γ from the cloud

    s = equation.nernst_slope(T, n)
    print(f"Daniell cell Zn|Zn²⁺||Cu²⁺|Cu   E° = {E0:.3f} V   n = {n}   T = {params['T_C']:.0f} °C")
    print(f"Nernst slope ln10·RT/nF = {1e3 * s.item():.2f} mV per decade of Q")
    print("E at Q = 0.01, 1, 100:  " + "   ".join(
        f"{round(Tk.item() - 273.15):d} °C: " + ", ".join(f"{E_tq[t, q].item():.4f}" for q in (20, 40, 60))
        for t, Tk in enumerate(T_batch)
    ))  # fmt: skip
    print(f"ε_r(water) = {equation.permittivity_water(T).item():.2f}   l_B = {l_B.item():.3f} nm   "
          f"A = {A.item():.4f} (kg/mol)^½   B = {B.item():.3f} nm⁻¹ (kg/mol)^½")  # fmt: skip
    print(f"{'m(Zn²⁺)':>9} {'I':>7} {'κ⁻¹ (nm)':>9} {'γ lim':>7} {'γ ext':>7} {'γ PB':>7} {'E ideal':>8} {'E ext':>8}")
    for c in range(0, len(m_zn), 12):
        print(
            f"{m_zn[c].item():9.1e} {I_zn[c].item():7.4f} {1 / kap[c].item():9.3f} {10 ** lg_lim_zn[c].item():7.4f} "
            f"{10 ** lg_ext_zn[c].item():7.4f} {10 ** lg_pb[c].item():7.4f} "
            f"{E_ideal[c].item():8.4f} {E_ext[c].item():8.4f}"
        )
    err = ((phi - phi_exact).abs() / phi_exact.abs().max(1, keepdim=True).values).max().item()
    print(f"Poisson–Boltzmann on {M}-point grids vs Yukawa: max error {err:.1e} of the surface potential")
    print(
        f"γ from the PB cloud vs extended Debye–Hückel: max |Δ log₁₀γ| = {(lg_pb - lg_ext_zn).abs().max().item():.1e}"
    )

    volts = equation.potential_scale(T, z)
    return plot.animate(
        log_Q, T_batch, E_tq, m_zn, m_cu, E_ideal, E_lim, E_ext, I_zn, lg_pb, A, B, a, z, T, r, phi, kap, volts,
        out_dir / "nernst_debye_huckel.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
