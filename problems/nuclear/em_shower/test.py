"""Physics checks, written first (TDD): what any correct electromagnetic-shower code must satisfy.

Run with `pytest problems/nuclear/em_shower/test.py`. Each test is a physical law, not an implementation detail: the
shower maximum follows the Heitler model, energy in flight plus deposited equals E0 and the stochastic profile peaks
near t_max.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def run_shower(e0: float, e_crit: float, stochastic: bool, seed: int = 0):
    t_max = equation.heitler_t_max(e0, e_crit)
    depths = tensors.depth_axis(t_max + 12.0, 20, CPU)
    result = solve.shower(tensors.primary(e0, CPU), depths, 20, e_crit, stochastic, seed=seed)
    return depths, result


def test_heitler_maximum():
    for e0, e_crit in ((10000.0, 7.4), (1024.0 * 10.0, 10.0), (50000.0, 20.0)):
        depths, result = run_shower(e0, e_crit, stochastic=False)
        total = result["counts"].sum(dim=1)
        k = int(total.argmax())
        n_max, t_max = equation.heitler_n_max(e0, e_crit), equation.heitler_t_max(e0, e_crit)
        assert n_max / 2 < total[k].item() <= n_max  # Heitler: N_max = E0/Ec
        assert abs(depths[k].item() - t_max) <= 1.0
    depths, result = run_shower(1024.0 * 10.0, 10.0, stochastic=False)  # E0/Ec a power of 2: exact
    total = result["counts"].sum(dim=1)
    assert total.max().item() == 1024 and depths[int(total.argmax())].item() == 10.0


def test_energy_is_conserved():
    for stochastic in (False, True):
        _, result = run_shower(10000.0, 7.4, stochastic)
        balance = result["in_flight"] + result["deposited"]
        # energy conservation: in flight + deposited = E0
        assert torch.allclose(balance, torch.full_like(balance, 10000.0), rtol=1e-12)
    _, result = run_shower(10000.0, 7.4, stochastic=False)
    assert result["counts"][-1].sum() == 0  # fixed lengths: everything has been absorbed by the end


def test_stochastic_profile():
    depths, result = run_shower(10000.0, 7.4, stochastic=True, seed=1)
    total = result["counts"].sum(dim=1).double()
    t_mean = (total * depths).sum() / total.sum()  # centre of the longitudinal profile
    assert equation.heitler_t_max(10000.0, 7.4) - 1 < t_mean < equation.heitler_t_max(10000.0, 7.4) + 6
    assert total.max() < equation.heitler_n_max(10000.0, 7.4)
