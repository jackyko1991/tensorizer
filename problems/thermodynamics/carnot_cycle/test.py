"""Physics checks, written first (TDD): what any correct Carnot-cycle code must satisfy.

Run with `pytest problems/thermodynamics/carnot_cycle/test.py`. The efficiency equals 1 − T_c/T_h whatever the
gas, the net work is Q_h − Q_c (first law), and the two adiabats do equal and opposite work.
"""

import math

from . import solve, tensors


def test_efficiency_and_first_law():
    for gas in ["monatomic", "diatomic"]:
        gamma = tensors.GAMMA[gas]
        legs = tensors.cycle(n=1.0, T_h=600.0, T_c=300.0, ratio=2.5, gamma=gamma)
        r = solve.solve(legs, 1.0, 600.0, 300.0, gamma)
        assert math.isclose(r["efficiency"], 1 - 300.0 / 600.0, rel_tol=1e-4)  # Carnot efficiency 1 − T_c/T_h
        assert math.isclose(r["W"], r["Q_h"] - r["Q_c"], rel_tol=1e-4)  # first law over a closed cycle
        # the two adiabats cancel: equal and opposite work
        assert math.isclose(r["W_legs"][1].item(), -r["W_legs"][3].item(), rel_tol=1e-6)
