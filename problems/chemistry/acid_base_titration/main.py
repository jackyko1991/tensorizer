"""Main loop: acids and volumes → charge balance solved for pH → equivalence points, buffer check → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    c_a, c_b, v0, pka_weak = params["c_acid"], params["c_base"], params["V0"], params["pKa_weak"]
    names, pkas = tensors.acids(pka_weak)
    log_beta = tensors.log_beta(tensors.pka_table(pkas, dev))  # (A, 4)
    v_eq = c_a * v0 / c_b  # mL of base per proton

    V = tensors.volumes(params["V_max"], 201 if fast_mode() else 1601, dev)  # (N,)
    C_A, Na = tensors.totals(V, c_a, v0, c_b)  # (N,), (N,)
    pH = solve.solve_ph(log_beta, C_A, Na, tensors.KW)  # (A, N): every acid and volume at once
    dpdv = solve.slope(pH, V)  # (A, N)
    found = solve.equivalence_points(dpdv, V)
    residual = equation.charge_balance(pH, log_beta, C_A, Na, tensors.KW).abs().max().item()
    print(f"{len(names)} acids × {len(V)} volumes solved together; largest charge-balance residual {residual:.1e} M")
    print(f"expected equivalence volumes: {v_eq:g} mL per proton (c_a V₀ / c_b)")

    # exact pH at a few key volumes: half-equivalence and equivalence points
    V_key = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64, device=dev) * v_eq
    pH_key = solve.solve_ph(log_beta, *tensors.totals(V_key, c_a, v0, c_b), tensors.KW)  # (A, 4)
    for a, name in enumerate(names):
        peaks = ", ".join(f"{v:.2f}" for v in found[a]) or "none"
        print(f"{name:<20} start pH {pH[a, 0].item():5.2f}   dpH/dV peaks at {peaks} mL")
        print(
            f"{'':<20} pH at ½·V_eq = {pH_key[a, 0].item():.3f}   at V_eq = {pH_key[a, 1].item():.3f}   "
            f"at 2·V_eq = {pH_key[a, 3].item():.3f}"
        )

    # buffer region of the weak acid: Henderson–Hasselbalch vs the exact curve, from 10 % to 90 % of V_eq
    buffer = (V >= 0.1 * v_eq) & (V <= 0.9 * v_eq)
    V_buf = V[buffer]
    pH_hh = equation.henderson_hasselbalch(pka_weak, c_b * V_buf, c_a * v0 - c_b * V_buf)
    print(f"buffer region of the weak acid: |pH − HH| ≤ {(pH[1, buffer] - pH_hh).abs().max().item():.3f}")

    # species fractions of H₃PO₄ over the whole pH scale
    pH_grid = torch.linspace(0.0, 14.0, 141 if fast_mode() else 561, dtype=torch.float64, device=dev)
    alpha = equation.fractions(pH_grid[None], log_beta[2:3])[0]  # (P, 4)
    print(f"α fractions of H₃PO₄ sum to 1 within {(alpha.sum(-1) - 1).abs().max().item():.1e}")

    return plot.figure(V, pH, dpdv, found, names, v_eq, pka_weak, V_buf, pH_hh, pH_grid, alpha, pkas[2], out_dir)


if __name__ == "__main__":
    cli(run, __file__)
