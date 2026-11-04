"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation as eq
from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # float64 everywhere (cgs numbers span 60 orders of magnitude): the CPU is the right device here.
    poly = solve.polytropes(torch.tensor([1.0, 1.5, 3.0], dtype=torch.float64))
    m_ch = eq.chandrasekhar_mass(poly["omega"][2], mu_e=2.0).item()
    print(f"Lane–Emden: ξ₁ = {', '.join(f'{v:.5f}' for v in poly['xi1'].tolist())} for n = 1, 3/2, 3;  "
          f"Chandrasekhar mass (n = 3, μ_e = 2) = {m_ch / eq.M_SUN:.3f} M☉")  # fmt: skip

    # --- brown dwarfs: the hottest a contracting body of mass M ever gets ---
    M_bd = tensors.log_ramp(0.004, 0.3, 300) * eq.M_SUN
    T_max = solve.brown_dwarf_max_temperature(M_bd, poly)
    m_d = solve.crossing(M_bd, T_max, tensors.T_IGNITE["D"])
    m_h = solve.crossing(M_bd, T_max, tensors.T_IGNITE["H"])
    print(f"brown dwarfs (one-zone): deuterium burning above {m_d / eq.M_JUP:.1f} M_J, hydrogen burning above "
          f"{m_h / eq.M_SUN:.3f} M☉ = {m_h / eq.M_JUP:.0f} M_J   (evolution models: ≈13 M_J, ≈0.075 M☉)")  # fmt: skip

    # --- white dwarfs and neutron stars: one batch of accreting shells ---
    wd_masses = torch.linspace(0.5, 1.4, 19).tolist()
    wd_rates = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    ns_rates = tensors.log_ramp(1e-11, 3e-8, 14).tolist()
    survey = tensors.shell_cases(wd_masses, wd_rates, ns_rates, m_ch, params["core_temp"] * 1e6, ("solar", "helium"))
    ign = solve.ignition(survey)
    print(f"ignition survey: {len(survey['M'])} accreting remnants in one batch, {int(ign['ignited'].sum())} ignite")

    comp = params["ns_fuel"]
    mine = tensors.shell_cases([params["wd_mass"]], [10 ** params["wd_rate"]], [10 ** params["ns_rate"]], m_ch,
                               params["core_temp"] * 1e6, (comp,))  # fmt: skip
    me = solve.ignition(mine)
    t_nova = (me["dM"][0] / mine["Mdot"][0]).item() / eq.YEAR
    t_burst = (me["dM"][1] / mine["Mdot"][1]).item() / 3600
    print(f"nova: {params['wd_mass']:.2f} M☉ white dwarf at 10^{params['wd_rate']:g} M☉/yr ignites after "
          f"ΔM = {me['dM'][0] / eq.M_SUN:.1e} M☉ (base T = {me['T'][0]:.1e} K, ρ = {me['rho'][0]:.1e} g/cm³), "
          f"recurrence {t_nova:.1e} yr")  # fmt: skip
    print(f"X-ray burst ({comp} fuel): ignition column y = {me['y'][1]:.1e} g/cm², ΔM = "
          f"{me['dM'][1] / eq.M_SUN:.1e} M☉, recurrence {t_burst:.1f} h, α = "
          f"{eq.burst_alpha(mine['M'][1], mine['R'][1], mine['X'][1]).item():.0f}")  # fmt: skip
    dm_spin = eq.spin_up_mass(torch.tensor([3e-3], dtype=torch.float64), tensors.NS_MASS).item()
    print(f"recycling a pulsar to P = 3 ms takes ≈ {dm_spin / eq.M_SUN:.2f} M☉ of accreted gas")

    # --- black holes: no surface, only accretion light up to Eddington ---
    M_bh = torch.tensor([10 * eq.M_SUN], dtype=torch.float64)
    eta = eq.accretion_efficiency(torch.tensor([params["bh_spin"]], dtype=torch.float64))
    L_edd = eq.eddington_luminosity(M_bh)
    mdot_edd = eq.eddington_rate(M_bh, eta).item() * eq.YEAR / eq.M_SUN
    print(f"black hole 10 M☉, a = {params['bh_spin']:g}: η = {eta.item():.3f}, L_Edd = {L_edd.item():.2e} erg/s "
          f"at Ṁ_Edd = {mdot_edd:.1e} M☉/yr; no surface, nothing to ignite")  # fmt: skip

    data = {
        "M_bd": M_bd, "T_max": T_max, "m_d": m_d, "m_h": m_h,
        "bd_m0": params["bd_mass"] * eq.M_SUN, "bd_rate": 10 ** params["bd_rate"] * eq.M_SUN / eq.YEAR,
        "survey": survey, "ign": ign, "wd_masses": wd_masses, "wd_rates": wd_rates, "ns_rates": ns_rates,
        "mine": mine, "me": me, "comp": comp, "m_ch": m_ch,
        "M_bh": M_bh, "eta": eta, "L_edd": L_edd, "bh_spin": params["bh_spin"],
    }  # fmt: skip
    if math.isnan(t_nova) or math.isnan(t_burst):
        raise ValueError("no ignition found on the column grid")
    gif = plot.animate(data, out_dir / "dead_star_reignition.gif")
    for pvd in plot.export_vtk(data, out_dir):  # interactive 3D scenes, one per remnant
        size = sum(f.stat().st_size for f in pvd.parent.glob(f"{pvd.stem}*")) / 1e6
        print(f"interactive results: {pvd.relative_to(out_dir).as_posix()} and its layers ({size:.2f} MB)")
    return gif


if __name__ == "__main__":
    cli(run, __file__)
